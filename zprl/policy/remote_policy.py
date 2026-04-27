from typing import Dict
import uuid

import dill
import torch
import zmq

from zprl.common.pytorch_util import dict_apply


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    return x


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return x


class RemoteImagePolicy:
    def __init__(self,
            server_addr: str,
            timeout_ms: int = 60000):
        self.server_addr = server_addr
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(server_addr)
        self._device = torch.device('cpu')
        self._dtype = torch.float32

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def eval(self):
        return self

    def train(self):
        return self

    def reset(self):
        self._request('reset', {})

    def close(self):
        self.socket.close(linger=0)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], rtc_context=None) -> Dict[str, torch.Tensor]:
        payload = {
            'obs': dict_apply(obs_dict, _to_cpu)
        }
        if rtc_context is not None:
            payload['rtc_context'] = dict_apply(rtc_context, _to_cpu)
        response = self._request('predict_action', payload)
        action_dict = response['action_dict']
        return dict_apply(action_dict, lambda x: _to_device(x, self.device))

    def _request(self, request_type: str, payload: dict):
        request_id = str(uuid.uuid4())
        request = {
            'type': request_type,
            'request_id': request_id,
            'payload': payload,
        }
        try:
            self.socket.send(dill.dumps(request))
            raw_response = self.socket.recv()
        except zmq.error.Again as e:
            raise TimeoutError(
                f"Timed out waiting for policy server {self.server_addr} "
                f"while handling {request_type}") from e

        response = dill.loads(raw_response)
        if response.get('request_id') != request_id:
            raise RuntimeError(
                f"Policy server returned mismatched request_id: "
                f"{response.get('request_id')} != {request_id}")
        if response.get('type') == 'error':
            raise RuntimeError(response.get('error'))
        return response.get('payload', {})
