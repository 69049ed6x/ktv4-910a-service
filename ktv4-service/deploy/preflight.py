"""Check the actual runtime and checkpoint before allocating model weights."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def inspect(args):
    root = Path(__file__).resolve().parent.parent
    errors = []
    checks = {}
    checks['python'] = platform.python_version()
    checks['architecture'] = platform.machine()
    if sys.platform != 'linux' or platform.machine() != 'aarch64':
        errors.append('This delivery targets Linux aarch64 Ascend servers.')
    if sys.version_info[:2] != (3, 11):
        errors.append('Use Python 3.11 to match the verified runtime.')
    weight = Path(args.model_path).resolve()
    checks['model_path'] = str(weight)
    try:
        config = json.loads((weight / 'config.json').read_text())
        checks['model_type'] = config.get('model_type')
        checks['architectures'] = config.get('architectures')
        checks['routed_experts'] = config.get('n_routed_experts')
        if config.get('model_type') != 'deepseek_v4':
            errors.append('Checkpoint model_type must be deepseek_v4.')
        index = weight / 'model.safetensors.index.json'
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else [p.name for p in weight.glob('*.safetensors')]
        if not shards:
            errors.append('No safetensors shards found.')
        missing = [name for name in shards if not (weight / name).is_file()]
        if missing:
            errors.append('Missing weight shards: ' + ', '.join(missing))
        checks['weight_shards'] = len(shards)
        checks['weight_bytes'] = sum((weight / name).stat().st_size for name in shards if (weight / name).is_file())
        if not any((weight / name).is_file() for name in ['tokenizer.json', 'tokenizer.model']):
            errors.append('Checkpoint tokenizer files are missing.')
    except (OSError, ValueError, KeyError) as exc:
        errors.append('Checkpoint inspection failed: ' + str(exc))
    memory = {}
    try:
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemTotal:'))}
    except OSError:
        pass
    checks['host_memory'] = memory
    # The running reference server has 755 GiB total and four CPU expert shards.
    checks['memory_note'] = 'Reference host: 755 GiB RAM; reserve several hundred GiB for CPU experts and weight loading.'
    expected = {'torch': '2.10.0+cpu', 'torch-npu': '2.10.0.post4', 'transformers': '5.5.4'}
    versions = {}
    for name, version in expected.items():
        try:
            versions[name] = importlib.metadata.version(name)
            if versions[name] != version:
                errors.append(f'{name}: expected {version}, found {versions[name]}')
        except importlib.metadata.PackageNotFoundError:
            errors.append('Missing distribution: ' + name)
    checks['versions'] = versions
    try:
        import torch
        import torch_npu  # noqa: F401
        count = torch.npu.device_count()
        checks['visible_npus'] = count
        if count < 4:
            errors.append('At least four visible NPUs are required.')
        checks['cpu_weight_int8_matmul_shape'] = list(torch._weight_int8pack_mm(
            torch.ones((1, 16), dtype=torch.float32),
            torch.ones((16, 16), dtype=torch.int8),
            torch.ones(16, dtype=torch.float32),
        ).shape)
    except Exception as exc:
        errors.append('torch / torch_npu check failed: ' + repr(exc))
    try:
        import vllm
        import vllm_ascend
        checks['vllm_source'] = str(Path(vllm.__file__).resolve())
        checks['vllm_ascend_source'] = str(Path(vllm_ascend.__file__).resolve())
        if not Path(vllm.__file__).resolve().is_relative_to((root / 'vendor/vllm').resolve()):
            errors.append('vLLM must import from this delivery, not another installation.')
        if not Path(vllm_ascend.__file__).resolve().is_relative_to((root / 'env210').resolve()):
            errors.append('vLLM-Ascend must import from the installed package in this delivery environment.')
        importlib.import_module('vllm_ascend.vllm_ascend_C')
        checks['native_extension_import'] = 'passed'
    except Exception as exc:
        errors.append('Patched libraries / native extension check failed: ' + repr(exc))
    try:
        from vllm.tokenizers import get_tokenizer
        tokenizer = get_tokenizer(weight, tokenizer_mode='deepseek_v4',
                                  trust_remote_code=True, local_files_only=True)
        checks['tokenizer_test_ids'] = tokenizer.encode('Hello')
        checks['chat_template_test_tokens'] = len(tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Hello'}], add_generation_prompt=True,
            thinking=False,
        ))
        import legacy_reference
        import runtime
        import executor
        importlib.import_module('vllm.entrypoints.openai.api_server')
        checks['service_imports'] = 'passed'
    except Exception as exc:
        errors.append('Tokenizer / service source imports failed: ' + repr(exc))
    result = {'passed': not errors, 'checks': checks, 'errors': errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if not errors else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--report')
    raise SystemExit(inspect(parser.parse_args()))
