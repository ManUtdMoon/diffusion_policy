import queue
import threading
import time
from collections import deque

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.remote_policy import RemoteImagePolicy


QP_ACTION_INDICES = np.array([0, 1, 2, 8, 9, 10], dtype=np.int64)


class QpChunkRuntime:
    """
    Async chunk runtime for RTG.

    This class wires policy inference, chunk alignment, and RTG bookkeeping.
    The actual position QP solve is intentionally not implemented yet.
    """

    LAMBDA_SMOOTH = 1.0
    LAMBDA_OLD = 1.0
    LAMBDA_NEW = 1.0

    def __init__(
        self,
        server_addr,
        n_action_steps,
        min_exec_horizon=None,
        timeout_ms=60000,
        delay_buffer_size=10,
        initial_delay_steps=5,
        qp_overlap_decay=5.0,
    ):
        self.server_addr = server_addr
        self.timeout_ms = timeout_ms
        self.n_action_steps = int(n_action_steps)
        self.min_exec_horizon = (
            int(min_exec_horizon) if min_exec_horizon is not None else self.n_action_steps
        )
        self.initial_delay_steps = int(initial_delay_steps)
        self.delay_history = deque([self.initial_delay_steps], maxlen=int(delay_buffer_size))
        self.qp_overlap_decay = float(qp_overlap_decay)

        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker = None

        self.current_chunk = None
        self.exec_idx = 0
        self.global_step = 0
        self.in_flight = False
        self.request_counter = 0
        self.last_action = None
        self.policy_latencies_sec = []

    def reset(self, obs):
        self.close()
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.delay_history.clear()
        self.delay_history.append(self.initial_delay_steps)
        self.exec_idx = 0
        self.global_step = 0
        self.in_flight = False
        self.request_counter = 0
        self.last_action = None
        self.policy_latencies_sec = []

        bootstrap_client = RemoteImagePolicy(
            server_addr=self.server_addr,
            timeout_ms=self.timeout_ms,
        )
        try:
            bootstrap_client.reset()
            action_dict = bootstrap_client.predict_action(self._obs_to_torch(obs))
        finally:
            bootstrap_client.close()

        self.current_chunk = self._extract_chunk(action_dict)

        self.worker = threading.Thread(
            target=self._worker_loop,
            args=(self.request_queue, self.response_queue),
            daemon=True,
        )
        self.worker.start()

    def step(self, obs):
        if self.current_chunk is None:
            raise RuntimeError("QpChunkRuntime.reset(obs) must be called before step().")

        self._drain_responses()
        if (not self.in_flight) and (self.exec_idx >= self.min_exec_horizon):
            self._enqueue_request(obs)

        action = self._get_current_action()
        self.last_action = action.copy()
        self.exec_idx += 1
        self.global_step += 1
        return action

    def close(self):
        if self.worker is not None:
            self.request_queue.put(None)
            self.worker.join(timeout=1.0)
            self.worker = None
        self.in_flight = False

    def get_log(self):
        return {
            "policy_latency_sec": list(self.policy_latencies_sec),
        }

    def _enqueue_request(self, obs):
        if self.exec_idx >= self.current_chunk.shape[0]:
            raise RuntimeError(
                f"Cannot enqueue RTG request at exec_idx={self.exec_idx}, "
                f"chunk_len={self.current_chunk.shape[0]}."
            )

        request_id = self.request_counter
        self.request_counter += 1
        self.in_flight = True
        self.request_queue.put({
            "request_id": request_id,
            "obs": dict(obs),
            "request_step": self.global_step,
            "request_exec_idx": self.exec_idx,
            "old_chunk": self.current_chunk.copy(),
        })

    def _drain_responses(self):
        while True:
            try:
                response = self.response_queue.get_nowait()
            except queue.Empty:
                return

            self.in_flight = False
            if "error" in response:
                raise RuntimeError(response["error"])

            observed_delay_steps = self.global_step - response["request_step"]
            self.delay_history.append(observed_delay_steps)
            new_chunk = response["chunk"]
            switch_idx = observed_delay_steps

            if switch_idx >= new_chunk.shape[0]:
                continue

            self.current_chunk = new_chunk
            self.exec_idx = switch_idx
            self.policy_latencies_sec.append(response["policy_latency_sec"])

    def _get_current_action(self):
        if self.exec_idx < self.current_chunk.shape[0]:
            return self.current_chunk[self.exec_idx]
        if self.last_action is not None:
            print(
                "[QpChunkRuntime] current chunk exhausted; "
                f"holding last action at global_step={self.global_step}, "
                f"exec_idx={self.exec_idx}, chunk_len={self.current_chunk.shape[0]}, "
                f"in_flight={self.in_flight}"
            )
            return self.last_action
        raise RuntimeError(
            f"Current chunk exhausted at exec_idx={self.exec_idx}, "
            f"chunk_len={self.current_chunk.shape[0]}."
        )

    def _worker_loop(self, request_queue, response_queue):
        policy_client = RemoteImagePolicy(
            server_addr=self.server_addr,
            timeout_ms=self.timeout_ms,
        )
        try:
            while True:
                request = request_queue.get()
                if request is None:
                    return
                try:
                    t0 = time.perf_counter()
                    action_dict = policy_client.predict_action(
                        self._obs_to_torch(request["obs"])
                    )
                    new_chunk = self._extract_chunk(action_dict)

                    qp_chunk = self._postprocess_chunk(
                        old_chunk=request["old_chunk"],
                        new_chunk=new_chunk,
                        request_exec_idx=request["request_exec_idx"],
                    )
                    policy_latency = time.perf_counter() - t0

                    response_queue.put({
                        "request_id": request["request_id"],
                        "request_step": request["request_step"],
                        "request_exec_idx": request["request_exec_idx"],
                        "chunk": qp_chunk,
                        "policy_latency_sec": policy_latency,
                    })
                except Exception as e:
                    response_queue.put({
                        "request_id": request["request_id"],
                        "error": repr(e),
                    })
        finally:
            policy_client.close()

    def _postprocess_chunk(self, old_chunk, new_chunk, request_exec_idx):
        if new_chunk.shape[1] != 16:
            raise ValueError(f"RTG expects wallet action dim 16, got {new_chunk.shape[1]}.")
        if old_chunk.shape[1] != 16:
            raise ValueError(f"RTG expects wallet action dim 16, got {old_chunk.shape[1]}.")

        s = int(request_exec_idx)
        h = int(new_chunk.shape[0])
        if s >= h:
            raise RuntimeError(f"RTG request_exec_idx={s} must be < chunk_len={h}.")

        d = int(max(self.delay_history))
        weights_old = self._make_rtg_weights(h=h, s=s, d=d)
        new_pos = new_chunk[:, QP_ACTION_INDICES]
        old_pos_overlap = old_chunk[s:, QP_ACTION_INDICES]
        old_pos_padded = np.zeros_like(new_pos)
        copy_len = min(old_pos_overlap.shape[0], h)
        old_pos_padded[:copy_len] = old_pos_overlap[:copy_len]
        position_chunk = self._solve_position_qp(
            old_pos_padded=old_pos_padded,
            new_pos=new_pos,
            weights_old=weights_old,
        )

        qp_chunk = new_chunk.copy()
        qp_chunk[:, QP_ACTION_INDICES] = position_chunk
        return qp_chunk.astype(np.float32)

    def _solve_position_qp(self, old_pos_padded, new_pos, weights_old):
        h, pos_dim = new_pos.shape
        old = old_pos_padded.reshape(h * pos_dim)
        new = new_pos.reshape(h * pos_dim)
        w_old = np.repeat(weights_old, pos_dim)
        w_new = 1.0 - w_old

        old_weight = self.LAMBDA_OLD * w_old
        new_weight = self.LAMBDA_NEW * w_new
        denom = old_weight + new_weight
        if np.any(denom <= 0):
            raise RuntimeError("RTG position solve has zero total weight.")

        position_chunk = (old_weight * old + new_weight * new) / denom
        return position_chunk.reshape(h, pos_dim).astype(np.float32)

    def _make_rtg_weights(self, h, s, d):
        weights_old = np.zeros(h, dtype=np.float64)
        overlap_end = max(min(h - s, h), 0)
        delay_end = min(d, overlap_end)
        weights_old[:delay_end] = 1.0

        transition_start = delay_end
        transition_end = max(overlap_end, transition_start)
        transition_len = transition_end - transition_start
        if transition_len > 0:
            u = np.arange(transition_len, dtype=np.float64) / max(transition_len - 1, 1)
            weights_old[transition_start:transition_end] = np.exp(
                -self.qp_overlap_decay * u
            )
        return weights_old

    def _obs_to_torch(self, obs):
        return dict_apply(
            obs,
            lambda x: torch.from_numpy(x).unsqueeze(0)
            if isinstance(x, np.ndarray) else x,
        )

    def _extract_chunk(self, action_dict):
        np_action_dict = dict_apply(
            action_dict,
            lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x,
        )
        if "action_pred_all" in np_action_dict:
            chunk = np_action_dict["action_pred_all"].squeeze(0)
        else:
            chunk = np_action_dict["action"].squeeze(0)
        return chunk.astype(np.float32)
