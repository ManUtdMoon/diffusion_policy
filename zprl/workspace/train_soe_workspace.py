import os

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from soe.soe_util import SoeBasePolicy
from zprl.common.action_mse_util import action_mse_per_sample
from zprl.common.checkpoint_util import TopKCheckpointManager
from zprl.common.json_logger import JsonLogger
from zprl.common.pytorch_util import dict_apply, optimizer_to
from zprl.model.common.lr_scheduler import get_scheduler
from zprl.workspace.train_flow_match_vib_unet_image_workspace import (
    TrainFlowMatchVibUnetImageWorkspace)


class TrainSoeWorkspace(TrainFlowMatchVibUnetImageWorkspace):
    """Joint base/VIB training with epoch monitoring and base-only evaluation."""

    include_keys = ['global_step', 'epoch', 'optimizer_step']

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.optimizer_step = 0

    def run(self):
        cfg = self.cfg
        assert not cfg.training.resume, 'SOE rounds train from scratch'
        assert cfg.training.gradient_accumulate_every == 1
        assert cfg.training.max_train_steps is None, 'Use max_grad_steps for SOE'
        if cfg.training.debug:
            cfg.training.max_grad_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        action_mse_groups = OmegaConf.to_container(
            OmegaConf.select(cfg, 'task.action_mse_groups', default=OmegaConf.create({})),
            resolve=True)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        max_grad_steps = cfg.training.max_grad_steps
        if max_grad_steps is not None:
            assert max_grad_steps > 0
            cfg.training.num_epochs = max_grad_steps // len(train_dataloader) + 1
        num_training_steps = max_grad_steps if max_grad_steps is not None else (
            len(train_dataloader) * cfg.training.num_epochs)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            **OmegaConf.to_container(cfg.training.lr_scheduler_kwargs, resolve=True))
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) \
            if cfg.training.use_ema else None

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk)
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)
        wandb_run = None
        try:
            wandb_run = wandb.init(
                dir=self.output_dir, config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging)
            wandb.config.update({'output_dir': self.output_dir})
            train_sampling_batch = None
            with JsonLogger(os.path.join(self.output_dir, 'logs.json.txt')) as json_logger:
                for local_epoch_idx in range(cfg.training.num_epochs):
                    self.model.train()
                    if cfg.training.freeze_encoder:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)
                    train_losses = []
                    for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader,
                            desc=f'Training epoch {self.epoch}', leave=False,
                            mininterval=cfg.training.tqdm_interval_sec)):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        loss, info = self.compute_batch_loss(batch)
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.optimizer_step += 1
                        lr_scheduler.step()
                        if ema is not None:
                            ema.step(self.model)
                        train_losses.append(loss.item())
                        step_log = {
                            'train_loss': loss.item(),
                            'global_step': self.global_step,
                            'optimizer_step': self.optimizer_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(info)
                        reached_limit = (max_grad_steps is not None and
                            self.optimizer_step >= max_grad_steps)
                        is_last_batch = batch_idx == len(train_dataloader) - 1 or reached_limit
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                        if reached_limit:
                            break

                    is_final_epoch = self.epoch == cfg.training.num_epochs - 1 or reached_limit
                    step_log['train_loss'] = float(np.mean(train_losses))
                    policy = self.ema_model if cfg.training.use_ema else self.model
                    policy.eval()
                    eval_policy = SoeBasePolicy(policy)
                    if self.epoch % cfg.training.rollout_every == 0 or is_final_epoch:
                        step_log.update(env_runner.run(eval_policy))
                    if len(val_dataloader) and self.epoch % cfg.training.val_every == 0:
                        val_losses, val_mse_samples = [], {}
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(val_dataloader):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, _ = self.compute_batch_loss(batch, model=policy)
                                val_losses.append(loss.item())
                                result = eval_policy.predict_action(batch['obs'])
                                mse_samples = action_mse_per_sample(
                                    result['action_pred'], batch['action'], action_mse_groups)
                                for name, values in mse_samples.items():
                                    val_mse_samples.setdefault(name, []).append(values.cpu())
                                if cfg.training.max_val_steps is not None and \
                                        batch_idx + 1 >= cfg.training.max_val_steps:
                                    break
                        step_log['val_loss'] = float(np.mean(val_losses))
                        for name, values in val_mse_samples.items():
                            key = 'val_action_mse' if name == '' else f'val_action_mse_{name}'
                            step_log[key] = torch.cat(values).mean().item()
                    if self.epoch % cfg.training.sample_every == 0:
                        result = eval_policy.predict_action(train_sampling_batch['obs'])
                        step_log['train_action_mse_error'] = torch.nn.functional.mse_loss(
                            result['action_pred'], train_sampling_batch['action']).item()
                        mse_samples = action_mse_per_sample(
                            result['action_pred'], train_sampling_batch['action'], action_mse_groups)
                        for name, values in mse_samples.items():
                            if name != '':
                                step_log[f'train_action_mse_{name}'] = values.mean().item()

                    if self.epoch % cfg.training.checkpoint_every == 0 or is_final_epoch:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint(use_thread=False)
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()
                        metric_dict = {key.replace('/', '_'): value
                            for key, value in step_log.items()}
                        if topk_manager.monitor_key in metric_dict:
                            path = topk_manager.get_ckpt_path(metric_dict)
                            if path is not None:
                                self.save_checkpoint(path=path, use_thread=False)
                        if is_final_epoch:
                            self.save_checkpoint(tag=f'step_{self.optimizer_step:06d}', use_thread=False)
                    policy.train()
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1
                    self.epoch += 1
                    if is_final_epoch:
                        break
        finally:
            env_runner.close()
            if wandb_run is not None:
                wandb_run.finish()
