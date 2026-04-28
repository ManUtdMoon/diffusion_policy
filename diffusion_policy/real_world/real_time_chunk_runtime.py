import queue
import threading
import time
from collections import deque

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.remote_policy import RemoteImagePolicy


class RealTimeChunkRuntime:
    """
    Minimal no-RTC async chunk runtime.

    The runner owns the environment. This class owns chunk execution state and
    a background policy client used to request future action chunks.
    """

    def __init__(
        self,
        server_addr,
        n_action_steps,
        min_exec_horizon=None,
        timeout_ms=60000,
        delay_buffer_size=10,
        initial_delay_steps=2,
    ):
        self.server_addr = server_addr
        self.timeout_ms = timeout_ms
        self.n_action_steps = int(n_action_steps)
        self.min_exec_horizon = (
            int(min_exec_horizon) if min_exec_horizon is not None else self.n_action_steps
        )
        self.initial_delay_steps = int(initial_delay_steps)
        self.delay_history = deque([self.initial_delay_steps], maxlen=int(delay_buffer_size))

        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker = None
        self.closed = False

        self.current_chunk = None
        self.current_nchunk = None
        self.exec_idx = 0
        self.global_step = 0
        self.in_flight = False
        self.request_counter = 0
        self.last_action = None
        self.policy_latencies_sec = []

    def reset(self, obs):
        self.close()
        self.closed = False
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
            t0 = time.perf_counter()
            action_dict = bootstrap_client.predict_action(self._obs_to_torch(obs))
            latency = time.perf_counter() - t0
        finally:
            bootstrap_client.close()

        self.current_chunk, self.current_nchunk = self._extract_chunks(action_dict)

        self.worker = threading.Thread(
            target=self._worker_loop,
            args=(self.request_queue, self.response_queue),
            daemon=True,
        )
        self.worker.start()

    def step(self, obs):
        if self.current_chunk is None:
            raise RuntimeError("RealTimeChunkRuntime.reset(obs) must be called before step().")

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
            self.closed = True
            self.request_queue.put(None)
            self.worker.join(timeout=1.0)
            self.worker = None
        self.in_flight = False

    def get_log(self):
        return {
            "policy_latency_sec": list(self.policy_latencies_sec),
        }

    def _enqueue_request(self, obs):
        request_id = self.request_counter
        self.request_counter += 1
        self.in_flight = True
        request = {
            "request_id": request_id,
            "obs": dict(obs),
            "request_step": self.global_step,
            "request_exec_idx": self.exec_idx,
            "request_time": time.perf_counter(),
        }
        self.request_queue.put(request)

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
            new_nchunk = response["nchunk"]
            switch_idx = observed_delay_steps

            if switch_idx >= new_chunk.shape[0]:
                continue

            self.current_chunk = new_chunk
            self.current_nchunk = new_nchunk
            self.exec_idx = switch_idx
            self.policy_latencies_sec.append(response["latency_sec"])

    def _get_current_action(self):
        if self.exec_idx < self.current_chunk.shape[0]:
            return self.current_chunk[self.exec_idx]
        if self.last_action is not None:
            print(
                "[RealTimeChunkRuntime] current chunk exhausted; "
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
                    latency = time.perf_counter() - t0
                    chunk, nchunk = self._extract_chunks(action_dict)
                    response_queue.put({
                        "request_id": request["request_id"],
                        "request_step": request["request_step"],
                        "request_exec_idx": request["request_exec_idx"],
                        "chunk": chunk,
                        "nchunk": nchunk,
                        "latency_sec": latency,
                    })
                except Exception as e:
                    response_queue.put({
                        "request_id": request["request_id"],
                        "error": repr(e),
                    })
        finally:
            policy_client.close()

    def _obs_to_torch(self, obs):
        return dict_apply(
            obs,
            lambda x: torch.from_numpy(x).unsqueeze(0)
            if isinstance(x, np.ndarray) else x,
        )

    def _extract_chunks(self, action_dict):
        np_action_dict = dict_apply(
            action_dict,
            lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x,
        )
        if "action_pred_all" in np_action_dict:
            chunk = np_action_dict["action_pred_all"].squeeze(0)
        else:
            chunk = np_action_dict["action"].squeeze(0)
        nchunk = None
        if "naction_pred_all" in np_action_dict:
            nchunk = np_action_dict["naction_pred_all"].squeeze(0)
        return chunk.astype(np.float32), nchunk
