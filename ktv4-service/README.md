# DeepSeek-V4-Flash 四卡服务源码交付

先看 **[部署与文件说明.md](部署与文件说明.md)**，每一个文件的作用见 **[文件逐项说明.md](文件逐项说明.md)**。

在匹配的 Linux aarch64 昇腾四卡服务器准备好驱动/固件、CANN 9.1.0、ATB、Python 3.11、系统构建工具和兼容的 W8A8 权重后：

```bash
bash setup.sh --model-path /data/models/w8a8-mtp
bash start.sh
```

安装脚本在线获取 Python 依赖，从本包构建修改后的服务库及自定义算子。本包没有权重、驱动、Python 环境和依赖二进制。

原始源码来自正在运行的 `qiusy@218.25.40.86:/mnt/nvme0n1/ktv4`，原机已实际完成模型问答验收。新服务器仍需完成依赖安装、算子编译和真实模型问答验收。部署约束、实际测试范围和完整校验清单均在主说明中。
