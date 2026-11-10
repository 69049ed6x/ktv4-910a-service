"""Regenerate the per-file source catalog and checksums; skip installed artifacts."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re


MAIN = {
    'README.md': '交付入口：指向完整部署说明和逐文件索引，给出安装与聊天的最短命令。',
    'DEPENDENCIES.md': '依赖版本、源码提交、包清单及外部系统组件的说明。',
    '部署与文件说明.md': '主交付文档：硬件/软件前提、安装、迁移、聊天、代码调用关系、验证范围和目标机验收。',
    '文件逐项说明.md': '本文件：每个交付文件的用途索引；上游说明依据路径、模块文档和顶层符号生成。',
    'delivery_manifest.json': '当前交付的逐文件哈希、来源、用途和迁移差异；对清单自身不计算递归哈希。',
    'export_manifest.json': '从运行服务器导出的原始逐文件地址、大小和 SHA-256，以及未下载的二进制/目录记录。',
    'requirements.lock.txt': '原服务器完整环境版本快照；包括系统组件和可选包，不能直接作为整体 pip 安装输入。',
    'start.sh': '正式启动入口：加载 deployment.env，等待四卡/API 就绪，然后启动终端问答；支持 --no-chat。',
    'chat.sh': '问答入口：使用本项目 Python 调用 app/chat.py，连接已启动的 API，支持 --prompt。',
    'stop.sh': '停止入口：调用进程管理器，只停止本项目记录且身份匹配的 API、worker 和 supervisor。',
    'status.sh': '状态入口：输出本项目状态、PID 和 API 健康检查结果。',
    'setup.sh': '迁移安装入口：检查基础环境，创建 env210，安装固定 Python 依赖，构建两套修改后服务库和原生算子，保存配置并自检。',
    'app/settings.py': '统一定位项目根、权重目录、Python、cache/run、端口；提供原子 JSON 写入。',
    'app/service.py': '服务生命周期管理：启动四卡与 API，等待本次 session 就绪，验证进程身份，有限内存诊断缓冲，清理旧驱动日志和停止进程。',
    'app/env.sh': '运行环境：加载部署配置、CANN/ATB，设置缓存、离线 HF、CPU 线程、四卡可见性和 HCCL 通信参数。',
    'app/launch_workers.sh': '用 torch.distributed.run 启动四个持久 worker，master 端口 31220。',
    'app/launch_api.sh': '启动 vLLM OpenAI API，选择 CPUOffloadExecutor；固定四卡、单请求、不分块 prefill，关闭 prefix cache 和异步调度。',
    'app/worker.py': '每个 rank 加载模型及自己负责的 CPU 专家，初始化 HCCL/优化，写入真实 tensor 设备记录，持续接收 ping/forward 文件 RPC。',
    'app/executor.py': 'vLLM 与持久 worker 之间的执行器：管理请求历史/位置/采样参数，通过 command/result 文件转发计算并及时删除已消费结果。',
    'app/chat.py': '标准库 HTTP/SSE 流式聊天客户端；多轮历史、/clear、/exit、等待提示和中断；不发送输出 token 上限。',
    'app/runtime.py': '正式推理路径：一次性应用已验证优化，重置/增长 KV，执行前向、检查 logits、原样采样并在四卡广播结果。',
    'app/official_reference_model.py': '参考模型结构：参数、注意力、压缩 KV、路由 gate、专家和 Transformer 层；由兼容层构造并修正设备行为。',
    'app/legacy_reference.py': '当前 910A/dav-c100 兼容主实现：载入参考模型、兼容数学、权重分配/反量化、CPU 路由专家和 NPU 层构造。',
    'app/cpu_experts.py': 'CPU 常驻 INT8 路由专家：按专家编号分四份，加载 weight/scale/offset，完成多 token 专家计算并返回本 rank 部分输出。',
    'app/fast_cpu_experts.py': '单 token CPU 专家优化：减少 gather/index_add，使用 CPU weight-only INT8 GEMM，保持 FP32 激活。',
    'app/fast_moe.py': 'NPU 共享专家的权重存储压缩及 gate/up 投影融合；CPU 路由专家仍由 CPU 实现负责。',
    'app/fast_small_ops.py': '已启用的小算子优化：融合 RMSNorm/RoPE、CPU Sinkhorn 等数值与设备兼容实现。',
    'app/fast_attention.py': '直接访问现有缓存的注意力实现，处理压缩/滚动 KV 和有效索引，避免旧测试路径。',
    'app/legacy_indices.py': '稳定的窗口/压缩 KV 索引与 RoPE 设备辅助函数，减少 decode 形状变化和跨设备索引问题。',
    'app/dynamic_kv_cache.py': '按实际 token 位置增长 KV 与压缩器状态，在上下文上限内避免一次分配完整百万 token 缓存。',
    'deploy/configure.py': '生成 deployment.env；对路径进行 shell 引用，允许外部显式环境变量覆盖默认配置。',
    'deploy/preflight.py': '安装后的真实环境自检：版本、架构、权重分片、四卡可见性、CPU INT8 GEMM、插件 ABI、DeepSeek-V4 tokenizer 与 API 导入。',
    'deploy/build_catalog.py': '重新生成逐文件说明、交付校验值和原机差异；跳过 env210、cache、run 等安装后产物。',
    'deploy/requirements.runtime.lock': '当前问答服务所需普通 Python 包的固定版本，包含实际 API 导入覆盖到的依赖；不携带包二进制。',
    'deploy/requirements.build.lock': '从原环境固定的 pip/setuptools/CMake/Ninja/pybind11 等构建工具版本。',
    'config/portable_validation.json': '新增迁移入口的隔离检查结果及测试范围；使用原机已有依赖，未宣称全新安装/重编译已完成。',
    'config/portability_dependency_audit.json': '原机实际动态库地址、CANN/ATB 环境、设备/系统信息与第三方 Python 源码完整性证据。',
    'config/dependency_metadata.json': '原机安装包的名称、版本和声明依赖关系，用来构造迁移安装清单。',
    'config/dependency_install_plan.json': '需要安装的普通包、CPU torch 来源、CANN/setuptools 提供的组件及原机缺失的可选/开发依赖。',
    'config/portability_changes.json': '将交付文件与原服务器哈希比较，列出新增、修改和是否缺失原始文件，区分迁移改动与模型代码。',
    'config/runtime.json': '原服务器实际模块导入地址、模型路径、四卡/线程/优化配置的静态证据，不作为新机启动配置。',
    'config/packages.json': '原服务器已安装包和版本的结构化记录。',
    'config/source_manifest.json': '原服务器主要服务源码的哈希记录，保留为迁移前证据。',
    'config/cleanup.json': '原服务器清理旧项目的结果和 root 权限残留记录，属于证据。',
    'config/final_layout.json': '原服务器清理完成后的目录和可读空间统计，含权限限制信息，属于证据。',
    'config/delivery_checks.json': '原服务最后的健康检查、重复启动、源码校验、一次性脚本/结果文件清理检查。',
    'config/verification.json': '原服务整理阶段实际中文、算术、多轮及复测结果，属于原机验收证据。',
    'config/verification_after_removal.json': '删除旧目录后原服务的完整重启及问答验收证据。',
    'config/post_cleanup_verification.json': '原机最后正式版本重启、中文/算术自然结束、四 rank tensor 设备和日志设置的验收证据。',
    'config/server_README.md': '原服务器 README 原文，保留原路径和操作说明以供对照。',
    'config/server_DEPENDENCIES.md': '原服务器依赖和运行代码定位说明原文。',
}


def module_context(rel):
    if '/third_party/catlass/' in rel:
        return 'Ascend 算子构建所用 Catlass 张量/分块/流水线模板库'
    if '/third_party/abseil-cpp/' in rel:
        return 'Ascend 构建所用 Abseil C++ 基础工具库'
    if '/third_party/ascend_protobuf/' in rel:
        return 'Ascend 构建所用 Protobuf 消息/代码生成及语言绑定源码'
    if '/_cann_ops_custom/' in rel:
        return '原生算子构建生成的 CANN 厂商包描述/辅助源码；二进制需重新构建'
    domains = [
        ('/vllm/tokenizers/', 'vLLM tokenizer、消息编码和模板支持'),
        ('/vllm/entrypoints/', 'vLLM API/CLI 入口及请求处理支持'),
        ('/vllm/v1/core/', 'vLLM 调度、KV 分配和请求生命周期支持'),
        ('/vllm/v1/engine/', 'vLLM 引擎进程、协议和请求执行协调支持'),
        ('/vllm/v1/executor/', 'vLLM 分布式执行接口；本项目接入自定义执行器'),
        ('/vllm/transformers_utils/', '模型配置、tokenizer 和 Hugging Face 兼容支持'),
        ('/vllm/model_executor/models/', '上游模型实现，是否使用由模型/执行器选择；当前主模型在 app'),
        ('/vllm/model_executor/layers/', '上游模型层、量化和 MoE 通用实现'),
        ('/vllm/distributed/', '分布式通信、并行组和传输支持'),
        ('/vllm/config/', 'vLLM 配置结构和参数校验'),
        ('/vllm_ascend/patch/', 'Ascend 对上游库的兼容补丁与注册'),
        ('/vllm_ascend/ops/', 'Ascend Python 算子封装或兼容实现'),
        ('/vllm_ascend/models/', 'Ascend 标准模型实现；与当前 app 自定义 worker 主路径区分'),
        ('/vllm_ascend/worker/', 'Ascend 标准 worker/runner；本服务执行自定义 app/worker.py'),
        ('/vllm_ascend/attention/', 'Ascend 标准注意力后端和元数据'),
        ('/vllm_ascend/quantization/', 'Ascend 权重量化配置、加载和算子支持'),
        ('/csrc/', '原生 C/C++/AscendC/CUDA 算子、接口或构建支持'),
        ('/cmake/', 'CMake 构建配置、编译器选项和依赖发现'),
        ('/rust/', '上游 Rust 前端/分词/解析源码；当前启动采用 Python API 前端'),
        ('/requirements/', '上游依赖声明；部署使用 deploy 中固定版本清单'),
        ('/tools/', '上游开发/诊断工具，非当前模型前向入口'),
        ('/scripts/', '库的构建/安装/开发辅助脚本'),
    ]
    for needle, text in domains:
        if needle in rel:
            return text
    if rel.startswith('vendor/vllm-ascend/'):
        return 'vLLM-Ascend 插件的库支持或构建文件'
    if rel.startswith('vendor/vllm/'):
        return 'vLLM 库支持、构建或项目资料'
    return '项目交付支持文件'


def describe(rel, data):
    if rel in MAIN:
        return MAIN[rel], ''
    p = Path(rel)
    context = module_context(rel)
    symbols = ''
    if p.name.endswith('_origin.json'):
        return '固定上游仓库地址和提交，供源码来源追溯：' + p.name, ''
    if p.suffix == '.patch':
        return '服务器保留的本地修改补丁，相关修改已经包含在导出源码中：' + p.name, ''
    if p.name in ('AGENTS.md', 'CLAUDE.md'):
        return '上游贡献/编辑约定，不是服务运行代码。', ''
    if p.name in ('LICENSE', 'COPYING', 'NOTICE', 'DCO') or p.name.startswith(('LICENSE', 'COPYING', 'NOTICE')):
        return '对应上游/第三方源码的许可、版权或贡献授权声明，随源码保留。', ''
    if p.name in ('setup.py', 'pyproject.toml', 'CMakeLists.txt', 'MANIFEST.in', 'Cargo.toml', 'Cargo.lock', '.gitmodules'):
        return context + '；' + p.name + ' 描述软件包、源码选择、版本、构建目标或固定依赖。', ''
    text = data.decode('utf-8', errors='replace')
    if p.suffix in ('.py', '.pyi'):
        doc = ''
        try:
            tree = ast.parse(text)
            doc = ast.get_docstring(tree) or ''
            names = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            symbols = ', '.join(names[:8])
            if len(names) > 8:
                symbols += f' 等 {len(names)} 个顶层定义'
        except SyntaxError:
            pass
        detail = re.sub(r'\s+', ' ', doc).strip()[:220]
        purpose = context + '；模块 ' + p.stem
        if p.name == '__init__.py':
            purpose += ' 的包初始化、导出或注册'
        if detail:
            purpose += '。源码说明：' + detail
        elif symbols:
            purpose += '，提供下列类/函数'
        else:
            purpose += '；具体入口以源文件定义为准'
        return purpose + '。', symbols
    if p.suffix in ('.cc', '.cpp', '.c', '.cu', '.h', '.hpp', '.cuh', '.inc', '.m', '.mm'):
        if '/op_host/' in rel:
            role = '算子 host 端注册、形状/类型推导或 tiling 配置'
        elif '/op_kernel/' in rel or '/kernels/' in rel:
            role = '算子 kernel、设备计算或 kernel 模板实现'
        elif p.suffix in ('.h', '.hpp', '.cuh', '.inc'):
            role = '接口、模板、常量或内联实现头文件'
        else:
            role = '原生实现编译单元'
        names = re.findall(r'\b(?:class|struct|enum\s+class)\s+(\w+)', text)
        return context + '；' + p.stem + ' 的' + role + '。', ', '.join(dict.fromkeys(names))[:180]
    if p.suffix == '.rs':
        names = re.findall(r'\bpub(?:\([^)]*\))?\s+(?:struct|enum|trait|fn)\s+(\w+)', text)
        return context + '；Rust 模块 ' + p.stem + '。', ', '.join(dict.fromkeys(names))[:180]
    if p.suffix in ('.java', '.cs', '.kt', '.php', '.rb', '.swift', '.js', '.ts'):
        return context + '；' + p.stem + ' 的语言绑定、生成实现或辅助源码。', ''
    if p.suffix == '.proto':
        names = re.findall(r'\b(?:message|enum|service)\s+(\w+)', text)
        return context + '；' + p.name + ' 定义序列化消息/枚举/协议。', ', '.join(names[:8])
    if p.suffix == '.json':
        try:
            obj = json.loads(text)
            keys = list(obj)[:7] if isinstance(obj, dict) else []
            symbols = ', '.join(str(x) for x in keys)
        except ValueError:
            pass
        return context + '；' + p.name + ' 的结构化描述、算子元数据或配置，具体字段见右列。', symbols
    if p.suffix in ('.sh', '.bash', '.bat', '.ps1'):
        return context + '；' + p.stem + ' 的构建、安装、运行或开发命令。', ''
    if p.suffix in ('.cmake', '.bzl', '.bazel', '.mk', '.in', '.make') or p.name.startswith(('Makefile', 'BUILD', 'WORKSPACE', 'Dockerfile')):
        return context + '；' + p.name + ' 的构建规则、模板、外部依赖或容器环境。', ''
    if p.suffix in ('.md', '.rst', '.txt', '.info'):
        heading = next((line.lstrip('# ').strip() for line in text.splitlines() if line.strip() and not line.startswith('<!--')), '')
        return context + '；' + p.name + ' 的文档、清单或说明' + ('。开头：' + heading[:130] if heading else '。'), ''
    if p.suffix in ('.gz', '.zip', '.xz', '.bz2'):
        return context + '；' + p.name + ' 是构建时使用的第三方源码归档，不是已安装运行环境。', ''
    if p.suffix in ('.png', '.jpg', '.jpeg', '.svg'):
        return '上游说明/开发工具的图片或矢量资源：' + p.name + '，不参与模型计算。', ''
    if p.suffix in ('.yaml', '.yml', '.toml', '.cfg', '.ini', '.xml', '.properties', '.config', '.plist') or p.name.startswith('.'):
        return context + '；' + p.name + ' 的项目、工具、平台或构建配置。', ''
    return context + '；保留 ' + p.name + ' 作为对应子模块的源码、资源或构建输入，具体语义结合源文件确认。', ''


def main():
    base = Path(__file__).resolve().parent.parent
    root = Path(chr(92) * 2 + '?' + chr(92) + str(base)) if os.name == 'nt' and not str(base).startswith(chr(92) * 2 + '?') else base
    original = json.loads((root / 'export_manifest.json').read_text(encoding='utf-8'))
    source = {entry['path']: entry for entry in original['files']}
    skipped_roots = {'env210', 'run', 'cache'}
    def all_paths():
        return sorted(p for p in root.rglob('*') if p.is_file()
                      and p.relative_to(root).parts[0] not in skipped_roots
                      and not any(part in {'__pycache__', '.git'} for part in p.relative_to(root).parts))
    changed = []
    for p in all_paths():
        rel = p.relative_to(root).as_posix()
        if rel not in source:
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != source[rel]['sha256']:
            changed.append({'path': rel, 'source_sha256': source[rel]['sha256'], 'delivery_sha256': digest, 'reason': 'portable configuration loading' if rel.endswith('.sh') else 'delivery documentation; original retained under config/server_*'})
    changes = {'source_files': len(source), 'modified_original_files': changed, 'missing_original_files': sorted(name for name in source if not (root / name).is_file()), 'inference_implementation_changed': any(x['path'].startswith('app/') and x['path'].endswith('.py') for x in changed),
               'additional_files_role': 'portable installer, dependency metadata, validation and documentation/checksum catalogs'}
    assert not changes['missing_original_files'], changes['missing_original_files']
    (root / 'config/portability_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
    # Include the two catalogs themselves in their own file lists.
    for name in ['文件逐项说明.md', 'delivery_manifest.json']:
        if not (root / name).exists():
            (root / name).write_text('', encoding='utf-8')
    descriptions = {}
    for p in all_paths():
        rel = p.relative_to(root).as_posix()
        descriptions[rel] = describe(rel, p.read_bytes())
    groups = {
        '一、服务、部署入口与交付说明': [r for r in descriptions if not r.startswith(('vendor/', 'config/'))],
        '二、配置、来源与验收证据': [r for r in descriptions if r.startswith('config/')],
        '三、vLLM 库源码与构建文件': [r for r in descriptions if r.startswith('vendor/vllm/')],
        '四、vLLM-Ascend、算子与第三方构建源码': [r for r in descriptions if r.startswith('vendor/vllm-ascend/')],
    }
    lines = ['# 每一个交付文件的作用', '', f'共 {len(descriptions):,} 个文件，每个文件单独一行。原始服务器源码文件 {len(source):,} 个。', '',
             '主服务文件按实际实现核对职责；上游/第三方条目按目录职责、模块文档和顶层定义自动建立索引。未能从源码提炼具体行为的条目明确保留为路径归属说明，不声称逐一手工审查了全部上游实现。', '',
             '库中其他模型、测试资源、开发工具和原生构建材料不一定进入当前文本问答执行路径。所有原机绝对来源地址和 SHA-256 见 export_manifest.json；本包当前校验值见 delivery_manifest.json。安装后新增 env210/cache/run 不属于源码交付清单。', '']
    def cell(value):
        return str(value).replace('|', '&#124;').replace('\n', ' ').replace('\r', ' ').replace('`', "'")
    for heading, names in groups.items():
        lines.extend(['## ' + heading, '', f'{len(names):,} 个文件。', '', '| 文件（相对交付根目录） | 作用 | 代表性顶层定义 / JSON 字段 |', '|---|---|---|'])
        for rel in sorted(names):
            purpose, symbols = descriptions[rel]
            lines.append('| `' + rel + '` | ' + cell(purpose) + ' | ' + cell(symbols or '—') + ' |')
        lines.append('')
    (root / '文件逐项说明.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    records = []
    for p in all_paths():
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        entry = {'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'purpose': descriptions[rel][0], 'symbols': descriptions[rel][1]}
        if rel in source:
            entry.update(server_path=source[rel]['server_path'], server_sha256=source[rel]['sha256'], origin='server_unchanged' if entry['sha256'] == source[rel]['sha256'] else 'server_with_portability_change')
        else:
            entry.update(server_path=None, origin='delivery_added')
        if rel == 'delivery_manifest.json':
            entry.update(bytes=None, sha256=None, self_hash_note='Manifest does not recursively hash itself.')
        records.append(entry)
    manifest = {'files_count': len(records), 'source_server_root': original['source_root'], 'source_original_files': len(source), 'origin_counts': dict(Counter(r['origin'] for r in records)), 'files': records}
    (root / 'delivery_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(records), 'origin_counts': manifest['origin_counts'], 'modified_original_files': [x['path'] for x in changed], 'missing_original_files': changes['missing_original_files']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
