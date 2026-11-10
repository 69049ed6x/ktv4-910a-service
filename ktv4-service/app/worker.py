"""Four persistent ranks: CPU routed experts and NPU projections/routing."""
import json
import os
import time
import traceback
from datetime import timedelta

from settings import CACHE, RUN, atomic_json

# CANN tuning IPC must be isolated between ranks and worker launches.
work = CACHE / 'cann_workers' / ('rank' + os.environ['LOCAL_RANK'] + '_' + str(os.getpid()))
work.mkdir(parents=True, exist_ok=True)
os.environ['ASCEND_WORK_PATH'] = str(work)
os.environ.pop('CANN_KNOWLEDGE_BANK_PROCESS_NUM', None)

import torch
import torch_npu
import torch.distributed as dist
from legacy_reference import make_model, load_checkpoint
from runtime import initialize, forward_step


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', timeout=timedelta(seconds=3600))
    with torch.inference_mode():
        handshake = torch.ones(1, device='npu')
        dist.all_reduce(handshake)
        torch.npu.synchronize()
    assert handshake.item() == 4
    model = make_model()
    load_checkpoint(model)
    initialize(model)
    atomic_json(RUN / f'rank{rank}.json', {
        'rank': rank, 'pid': os.getpid(), 'session': os.environ['KTV4_SESSION'],
        'context': model.max_seq_len, 'layers': len(model.layers),
        'routed_experts_device': str(model.layers[0].ffn.experts.w13.device),
        'gate_device': str(model.layers[0].ffn.gate.weight.device),
        'shared_expert_device': str(model.layers[0].ffn.shared_experts.w2.weight.device),
    })
    previous = None
    command_file = RUN / 'command.json'
    while True:
        if not command_file.exists():
            time.sleep(.005)
            continue
        command = json.loads(command_file.read_text())
        if command['id'] == previous:
            time.sleep(.005)
            continue
        previous = command['id']
        dist.barrier()
        try:
            if command['action'] == 'ping':
                if rank == 0:
                    atomic_json(RUN / f'result_{previous}.json', {
                        'ready': True, 'tp': dist.get_world_size(),
                        'layers': len(model.layers), 'context': model.max_seq_len,
                    })
            elif command['action'] == 'forward':
                forward_step(model, command)
            else:
                raise ValueError('Unsupported worker action: ' + command['action'])
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            atomic_json(RUN / f'worker_error_rank{rank}.json', {'error': error, 'command': previous})
            if rank == 0:
                atomic_json(RUN / f'result_{previous}.json', {'error': error})


if __name__ == '__main__':
    main()
