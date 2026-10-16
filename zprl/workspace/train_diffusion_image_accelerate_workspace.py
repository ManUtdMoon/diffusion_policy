if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import contextlib
import copy
import os
import pathlib
import random
from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
import hydra
import numpy as np
from omegaconf import OmegaConf, ListConfig
import torch
from torch.utils.data import DataLoader
import tqdm
import wandb

from zprl.common.action_mse_util import action_mse_per_sample
from zprl.common.checkpoint_util import TopKCheckpointManager
from zprl.common.json_logger import JsonLogger
from zprl.common.training_budget import configure_training_budget
from zprl.common.pytorch_util import dict_apply
from zprl.dataset.base_dataset import BaseImageDataset
from zprl.env_runner.base_image_runner import BaseImageRunner
from zprl.model.common.lr_scheduler import get_scheduler
from zprl.model.diffusion.ema_model import EMAModel
from zprl.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _NullLogger:
    def log(self, data):
        return data


def _mean_scalar(accelerator: Accelerator, value, device: torch.device) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    gathered = accelerator.gather_for_metrics(tensor)
    return gathered.mean().item()


def _split_total_batch_size(total_batch_size: int, world_size: int, name: str) -> int:
    if total_batch_size < world_size:
        raise ValueError(
            f"{name}.batch_size={total_batch_size} is smaller than world_size={world_size}"
        )
    if total_batch_size % world_size != 0:
        raise ValueError(
            f"{name}.batch_size={total_batch_size} must be divisible by world_size={world_size}"
        )
    return total_batch_size // world_size


def _action_mse_log(metric_samples, prefix):
    result = dict()
    for name, values in metric_samples.items():
        key = prefix if name == "" else f"{prefix}_{name}"
        result[key] = torch.cat(values).mean().item()
    return result


def _configure_cuda_devices(gpu_ids):
    if gpu_ids is None:
        return
    if not isinstance(gpu_ids, (list, tuple, ListConfig)) or not gpu_ids \
            or any(type(index) is not int for index in gpu_ids):
        raise ValueError('training.gpu_ids must be a nonempty list of CUDA device indices')
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if len(gpu_ids) != local_world_size:
        raise ValueError('training.gpu_ids must have one entry per local training process')
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('training.gpu_ids must not contain duplicate devices')
    device_count = torch.cuda.device_count()
    if any(index < 0 or index >= device_count for index in gpu_ids):
        raise ValueError(f'training.gpu_ids must be within the {device_count} visible CUDA devices')
    index = gpu_ids[local_rank]
    torch.cuda.set_device(index)
    os.environ['ACCELERATE_TORCH_DEVICE'] = f'cuda:{index}'
    # DDP must follow the model device instead of using LOCAL_RANK as a GPU ID.
    os.environ['ACCELERATE_BYPASS_DEVICE_MAP'] = 'true'


class TrainDiffusionImageAccelerateWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        if cfg.training.gradient_accumulate_every != 1:
            raise ValueError('Accelerate workspace currently requires gradient_accumulate_every=1')

        _configure_cuda_devices(cfg.training.get('gpu_ids'))
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=cfg.training.ddp_find_unused_parameters)
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(
            seconds=cfg.training.get('ddp_timeout_seconds', 600)))
        accelerator = Accelerator(
            mixed_precision=cfg.training.mixed_precision,
            kwargs_handlers=[ddp_kwargs, process_group_kwargs]
        )
        device = accelerator.device
        set_seed(cfg.training.seed, device_specific=True)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path,
                    include_keys=self.include_keys)
                self.epoch += 1

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        action_mse_groups = OmegaConf.to_container(
            cfg.task.action_mse_groups, resolve=True)

        train_dataloader_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_kwargs = OmegaConf.to_container(cfg.val_dataloader, resolve=True)

        train_shuffle = train_dataloader_kwargs.pop('shuffle', True)
        val_shuffle = val_dataloader_kwargs.pop('shuffle', False)
        train_total_batch_size = train_dataloader_kwargs['batch_size']
        val_total_batch_size = val_dataloader_kwargs['batch_size']
        train_dataloader_kwargs['batch_size'] = _split_total_batch_size(
            train_total_batch_size, accelerator.num_processes, 'dataloader')
        val_dataloader_kwargs['batch_size'] = _split_total_batch_size(
            val_total_batch_size, accelerator.num_processes, 'val_dataloader')

        train_dataloader = DataLoader(
            dataset,
            shuffle=train_shuffle,
            **train_dataloader_kwargs
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=val_shuffle,
            **val_dataloader_kwargs
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
            self.ema_model.to(device)

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            ema.optimization_step = self.global_step

        env_runner: BaseImageRunner = None
        if accelerator.is_main_process and cfg.training.get('enable_rollout', True):
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        wandb_run = None
        if accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                    "world_size": accelerator.num_processes,
                    "train_total_batch_size": train_total_batch_size,
                    "train_batch_size_per_device": train_dataloader_kwargs['batch_size'],
                    "val_total_batch_size": val_total_batch_size,
                    "val_batch_size_per_device": val_dataloader_kwargs['batch_size'],
                }
            )

        topk_manager = None
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        model = self.model
        optimizer = self.optimizer
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader)
        raw_model = accelerator.unwrap_model(model)

        batches_per_epoch = len(train_dataloader)
        total_train_batches = configure_training_budget(
            cfg.training, batches_per_epoch)

        lr_scheduler_kwargs = OmegaConf.to_container(
            OmegaConf.select(cfg, "training.lr_scheduler_kwargs", default={}),
            resolve=True)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_train_batches,
            last_epoch=self.global_step-1,
            **lr_scheduler_kwargs
        )

        if accelerator.is_main_process:
            wandb.config.update(
                {
                    "train_dataloader_len_after_prepare": batches_per_epoch,
                    "num_training_steps": total_train_batches,
                    "num_epochs": cfg.training.num_epochs,
                }
            )

        train_sampling_batch = None

        optimizer.zero_grad()
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        logger_context = JsonLogger(log_path) if accelerator.is_main_process else contextlib.nullcontext(_NullLogger())
        with logger_context as json_logger:
            while self.global_step < total_train_batches:
                step_log = dict()

                model.train()
                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=not accelerator.is_local_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        loss = model(batch)
                        raw_loss = loss.detach()

                        accelerator.backward(loss)
                        grad_norm = None
                        if cfg.training.get('gradient_clip_val') is not None:
                            grad_norm = accelerator.clip_grad_norm_(
                                model.parameters(), cfg.training.gradient_clip_val)
                        optimizer.step()
                        lr_scheduler.step()
                        if cfg.training.use_ema:
                            ema.step(raw_model)
                        optimizer.zero_grad()

                        loss_cpu = _mean_scalar(accelerator, raw_loss, device)
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        self.global_step += 1
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        if grad_norm is not None:
                            step_log['grad_norm'] = _mean_scalar(accelerator, grad_norm, device)

                        is_last_batch = (batch_idx == (len(train_dataloader)-1)) \
                            or self.global_step >= total_train_batches
                        if not is_last_batch:
                            if accelerator.is_main_process:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        if self.global_step >= total_train_batches:
                            break

                train_loss = float(np.mean(train_losses))
                step_log['train_loss'] = train_loss

                model.eval()
                if self.ema_model is not None:
                    self.ema_model.eval()

                if cfg.training.get('enable_rollout', True) and (
                    (self.epoch > 0 and self.epoch % cfg.training.rollout_every == 0)
                    or self.global_step >= total_train_batches
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        policy = self.ema_model if cfg.training.use_ema else raw_model
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)
                    accelerator.wait_for_everyone()

                if (self.epoch % cfg.training.val_every) == 0 \
                        or self.global_step >= total_train_batches:
                    with torch.no_grad(), accelerator.autocast():
                        val_mse_samples = dict()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                                disable=not accelerator.is_local_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                policy = self.ema_model if cfg.training.use_ema else raw_model
                                result = policy.predict_action(batch["obs"])
                                mse_samples = action_mse_per_sample(
                                    result["action_pred"], batch["action"], action_mse_groups)
                                mse_samples = accelerator.gather_for_metrics(mse_samples)
                                for name, values in mse_samples.items():
                                    val_mse_samples.setdefault(name, list()).append(values.cpu())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_mse_samples) > 0:
                            step_log.update(_action_mse_log(
                                val_mse_samples, "val_action_mse"))

                if (self.epoch % cfg.training.sample_every) == 0:
                    accelerator.wait_for_everyone()
                    if train_sampling_batch is not None:
                        with torch.no_grad(), accelerator.autocast():
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            policy = self.ema_model if cfg.training.use_ema else raw_model
                            result = policy.predict_action(batch["obs"])
                            mse_samples = action_mse_per_sample(
                                result["action_pred"], batch["action"], action_mse_groups)
                            mse_samples = accelerator.gather(mse_samples)
                            mse_samples = {
                                name: [values.cpu()]
                                for name, values in mse_samples.items()
                            }
                            step_log.update(_action_mse_log(
                                mse_samples, "train_action_mse"))

                if (self.epoch % cfg.training.checkpoint_every) == 0 \
                        or self.global_step >= total_train_batches:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                model.train()
                if self.ema_model is not None:
                    self.ema_model.train()

                if accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.epoch += 1
                accelerator.wait_for_everyone()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionImageAccelerateWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
