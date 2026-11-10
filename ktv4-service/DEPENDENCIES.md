# 依赖与来源

完整安装步骤见 [部署与文件说明.md](部署与文件说明.md)。

| 路径 | 用途 |
|---|---|
| `deploy/requirements.build.lock` | 与原机一致的 Python 构建工具版本 |
| `deploy/requirements.runtime.lock` | 当前文本问答的 Python 包及版本；由安装脚本以 `--no-deps` 安装 |
| `requirements.lock.txt` | 原机全环境快照，含 CANN 自带组件和可选扩展，不直接整体 pip 安装 |
| `config/dependency_install_plan.json` | 包选择、原机缺失的可选/开发依赖和 CANN 提供的组件 |
| `config/dependency_metadata.json` | 原环境声明的依赖关系 |
| `config/portability_dependency_audit.json` | 实际加载动态库、系统/设备环境及依赖源码完整性检查 |
| `config/vllm_origin.json`、`config/vllm-ascend_origin.json` | 上游仓库和固定提交 |
| `config/vllm.patch`、`config/vllm-ascend.patch` | 原服务器本地修改证据；源码已包含修改，不重复应用 |

关键组合：Python 3.11、CPU PyTorch 2.10.0+cpu、torch-npu 2.10.0.post4、Transformers 5.5.4、CANN 9.1.0、ATB CXX ABI 1。安装时不允许上游开发版依赖自动覆盖 torch 版本。

vLLM 提交为 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，vLLM-Ascend 提交为 `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。实际安装的 Ascend 插件 Python 文件与导出源码一致。检查 torch、torch_npu、transformers、safetensors、numba 的 Python 文件时，没有发现相对各包 RECORD 的修改。

驱动、固件、CANN/OPP、ATB 由目标服务器自行安装。完整原生算子源码位于 `vendor/vllm-ascend/csrc`，编译产物不随包分发。仓库内第三方构建源码保留，以保全源码构建上下文。
