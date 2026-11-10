"""vLLM executor for the resident dav-c100 four-NPU / CPU expert model.

The four workers own KV buffers sized from the checkpoint context (or DEEPSEEK_MAX_SEQ_LEN). vLLM owns request
scheduling, detokenization, stopping and the OpenAI API. A single sequence,
unchunked prefill, and disabled prefix caching prevent private KV reuse
across requests. No A2-only NPU worker or model runner is constructed.
"""
from concurrent.futures import Future
import json
from pathlib import Path
import time
import uuid

from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

from settings import RUN


class CPUOffloadExecutor(Executor):
    def _init_executor(self):
        if self.scheduler_config.max_num_seqs != 1:
            raise ValueError('Resident reference backend supports max_num_seqs=1')
        if self.cache_config.enable_prefix_caching:
            raise ValueError('Private rolling KV requires disabled prefix caching')
        if self.scheduler_config.enable_chunked_prefill:
            raise ValueError('Reference prefill must be unchunked')
        if self.parallel_config.tensor_parallel_size != 4:
            raise ValueError('Resident worker group has tensor_parallel_size=4')
        self.requests = {}
        health = self._rpc({'action': 'ping'})
        if self.model_config.max_model_len > health['context']:
            raise ValueError('Requested context exceeds resident KV capacity')
        print('LEGACY_VLLM_EXECUTOR_READY', health, flush=True)

    def _rpc(self, command):
        command = dict(command, id=uuid.uuid4().hex)
        target = RUN / 'command.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(command), encoding='utf-8')
        temporary.replace(target)
        result = RUN / f'result_{command["id"]}.json'
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if result.exists():
                try:
                    response = json.loads(result.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.01)
                    continue
                result.unlink(missing_ok=True)
                if 'error' in response:
                    raise RuntimeError(response['error'])
                return response
            time.sleep(0.01)
        raise TimeoutError('Resident four-rank model did not return within 1200 seconds')

    def get_kv_cache_specs(self):
        # KV is allocated and retained by the resident workers, outside vLLM's pool.
        return [{}]

    def determine_available_memory(self):
        return [0]

    def initialize_from_config(self, kv_cache_configs):
        return None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None,
                       non_block=False, **options):
        if method == 'get_supported_tasks':
            output = [('generate',)]
        elif method == 'get_kv_connector_handshake_metadata':
            output = [None]
        elif method in ('shutdown', 'reset_mm_cache', 'reset_encoder_cache'):
            output = [None]
        else:
            raise NotImplementedError(f'Reference executor does not support RPC {method}')
        if non_block:
            future = Future()
            future.set_result(output)
            return future
        return output

    def execute_model(self, scheduler_output, non_block=False):
        try:
            output = self._execute(scheduler_output)
        except Exception as error:
            if not non_block:
                raise
            future = Future()
            future.set_exception(error)
            return future
        if non_block:
            future = Future()
            future.set_result(output)
            return future
        return output

    def _execute(self, scheduled):
        for req_id in scheduled.finished_req_ids:
            self.requests.pop(req_id, None)
        for request in scheduled.scheduled_new_reqs:
            params = request.sampling_params
            if params.logprobs is not None or params.prompt_logprobs is not None:
                raise ValueError('Reference executor does not yet expose logprobs')
            if params.structured_outputs is not None:
                raise ValueError('Reference executor does not yet support constrained decoding')
            self.requests[request.req_id] = {
                'tokens': list(request.prompt_token_ids),
                'prompt_len': len(request.prompt_token_ids), 'params': params,
            }
        ids = list(scheduled.num_scheduled_tokens)
        if not ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        if len(ids) != 1:
            raise ValueError('Private KV backend cannot execute concurrent sequences')
        req_id = ids[0]
        state = self.requests[req_id]
        cached = scheduled.scheduled_cached_reqs
        if req_id in cached.req_ids:
            position = cached.num_computed_tokens[cached.req_ids.index(req_id)]
        else:
            position = 0
        count = scheduled.num_scheduled_tokens[req_id]
        tokens = state['tokens'][position:position+count]
        if len(tokens) != count:
            raise RuntimeError('Scheduler token history is inconsistent')
        params = state['params']
        generated_count = len(state['tokens']) - state['prompt_len']
        response = self._rpc({
            'action': 'forward', 'position': position, 'tokens': tokens,
            'history': state['tokens'],
            'output_history': state['tokens'][state['prompt_len']:],
            'sampling': {
                'temperature': params.temperature, 'top_k': params.top_k,
                'top_p': params.top_p, 'repetition_penalty': params.repetition_penalty,
                'presence_penalty': params.presence_penalty,
                'frequency_penalty': params.frequency_penalty,
                'mask_eos': generated_count < params.min_tokens,
                'seed': params.seed, 'generated_count': generated_count,
                'min_p': params.min_p, 'logit_bias': params.logit_bias,
                'allowed_token_ids': params.allowed_token_ids,
            },
        })
        token = response['token_id']
        state['tokens'].append(token)
        return ModelRunnerOutput(req_ids=[req_id], req_id_to_index={req_id: 0},
                                 sampled_token_ids=[[token]])

    def check_health(self):
        self._rpc({'action': 'ping'})

    def shutdown(self):
        self.requests.clear()
