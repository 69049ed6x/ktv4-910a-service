# DeepSeek-V4-Flash · 昇腾 910A 四卡服务

在当前服务器上直接启动并进入终端问答：

```bash
bash /mnt/nvme0n1/ktv4/start.sh
```

启动入口会自动加载四卡模型、等待所有权重加载完成、启动 OpenAI 兼容 API，再进入聊天。已有服务正常运行时会直接复用，不重复加载权重。首次加载需要等待；遇到未缓存的输入形状时，CANN 算子编译可能需要数分钟，聊天终端每15秒显示等待时间。

只启动后台服务：

```bash
bash /mnt/nvme0n1/ktv4/start.sh --no-chat
```

连接已启动的服务、查看状态、停止本项目全部服务：

```bash
bash /mnt/nvme0n1/ktv4/chat.sh
bash /mnt/nvme0n1/ktv4/status.sh
bash /mnt/nvme0n1/ktv4/stop.sh
```

终端直接输入提示词并回车。`/clear` 清空历史，`/exit` 退出聊天；退出聊天会保留后台服务。回答期间 Ctrl+C 关闭当前客户端请求，正在执行的后端前向可能仍需完成。输入时 Ctrl+D 退出。

客户端不发送 `max_tokens` 或 `max_completion_tokens`，正常生成到 EOS。输入和输出仍共享模型上下文容量；默认配置为1,048,576 token，完整百万 token 推理能力未经验证。默认关闭思考模式，保留多轮对话。当前服务一次处理一个请求。

单条问答：

```bash
bash /mnt/nvme0n1/ktv4/chat.sh --prompt '你好'
```

目录说明：

| 路径 | 内容 |
|---|---|
| `app/` | 当前运行的模型、CPU专家、兼容算子、API执行器、进程管理、聊天客户端 |
| `vendor/vllm/` | API、调度器、tokenizer等所使用的完整库源码及原有修改 |
| `vendor/vllm-ascend/` | Ascend插件与原生算子的构建源码 |
| `vendor/vllm_ascend_installed/` | 指向当前实际导入的已安装Ascend插件，含二进制算子 |
| `env210/` | 保留原位置的可运行Python 3.11环境及依赖 |
| `cache/` | 可复用的算子编译缓存；不属于历史日志 |
| `run/` | 当前进程状态、四卡就绪标记和请求通信文件；不保存聊天记录 |
| `config/` | 依赖、源码来源、库修改补丁、当前配置和验收结果 |

模型权重位于 `/mnt/nvme1n1/bf16/w8a8-mtp`，没有复制到项目内。该目录必须保留。`KTV4_MODEL_PATH` 可在服务启动前指定其他兼容权重；`DEEPSEEK_MAX_SEQ_LEN` 可调整上下文配置，不能超过权重配置上限。

当前计算分工：每层256个路由专家的INT8权重保存在CPU内存，四个进程各负责64个；路由专家用CPU计算。Gate与专家选择在四张NPU上各执行一份；前三层使用hash路由。共享专家权重与计算在NPU；路由专家的部分结果传回NPU后通过HCCL归约。启动后 `run/rank0.json` 至 `rank3.json` 记录真实驻留设备。

启动、API和CANN输出交给进程管理器的有限内存缓冲，不积累历史日志；启动或子进程失败时，最近的诊断信息写入当前 `run/state.json`。请求结果消费后立即删除。驱动可能生成的少量运行文件只留在 `run/`，启动和停止服务时清除本项目的旧驱动日志。

旧测试、旧构建、旧GGUF权重和废弃环境已删除。当前SSH账号没有sudo权限，`code/reference`、`code/llama.cpp-v4ref/gguf-py/gguf/__pycache__`、`AscendUsage`、`tmp` 下仍有root所有、无权删除的残留；这些目录不参与当前启动。完整删除结果见 `config/cleanup.json`，实际重启与问答验收见 `config/post_cleanup_verification.json`。

本交付针对当前ARM64服务器、现有昇腾驱动、CANN 9.1.0和ATB环境。`requirements.lock.txt` 是版本清单，不代表能仅靠公开pip包在任意新机器重建所有二进制组件。系统驱动/CANN、Python环境及权重都是运行依赖。详细位置见 `DEPENDENCIES.md`。
