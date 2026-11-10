#!/usr/bin/env python3
"""Interactive streaming chat client for the resident DeepSeek service.

Only Python's standard library is required. This client does not load weights,
change the serving code, or start another inference process.
"""
import argparse
import http.client
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import readline  # noqa: F401; in-memory editing and input history on Linux.
except ImportError:
    pass


class ChatError(Exception):
    pass


class FirstTokenStatus:
    """Keep the terminal responsive while the synchronous HTTP reader waits."""

    def __init__(self, interval=15):
        self.interval = interval
        self.stopped = threading.Event()
        self.thread = None

    def __enter__(self):
        self.started = time.monotonic()
        print("[正在等待首段回答；排队或首次算子编译可能耗时数分钟。Ctrl+C 可中断等待。]",
              file=sys.stderr, flush=True)
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped.wait(self.interval):
            print("[仍在等待首段回答，已等待 {:.0f} 秒。]".format(time.monotonic() - self.started),
                  file=sys.stderr, flush=True)

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def __exit__(self, *exc):
        self.stop()


def error_message(error):
    if isinstance(error, dict):
        return str(error.get("message") or error.get("error") or error)
    return str(error)


def make_request(url, payload=None):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers.update({"Content-Type": "application/json", "Accept": "text/event-stream"})
    request = urllib.request.Request(url, data=data, headers=headers)
    return request


def open_request(request, timeout):
    try:
        # The local inference endpoint must not be routed through a shell proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read(65536).decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
            message = error_message(body.get("error", body))
        except (ValueError, AttributeError):
            message = raw.strip() or str(exc.reason)
        raise ChatError("HTTP {}：{}".format(exc.code, message)) from exc
    except urllib.error.URLError as exc:
        raise ChatError("无法连接推理服务：{}".format(exc.reason)) from exc


def check_service(base_url, model):
    with open_request(make_request(base_url + "/models"), 10) as response:
        data = json.load(response)
    models = [item.get("id") for item in data.get("data", [])]
    if model not in models:
        raise ChatError("服务未提供模型 {}；可用模型：{}".format(model, ", ".join(models)))


def sse_data(response):
    """Yield complete SSE data events, including multiline data fields."""
    parts = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(" ") else value)
    if parts:
        yield "\n".join(parts)


def generate(args, messages, emit):
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "stream": True,
        "chat_template_kwargs": {"thinking": False},
    }
    pieces = []
    finish_reason = None
    done = False
    with open_request(make_request(args.base_url + "/chat/completions", payload), args.timeout) as response:
        for data in sse_data(response):
            if data.strip() == "[DONE]":
                done = True
                break
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise ChatError("服务返回了无法解析的流式数据。") from exc
            if not isinstance(event, dict):
                raise ChatError("服务返回了异常的流式数据。")
            if event.get("error"):
                raise ChatError(error_message(event["error"]))
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                content = (choice.get("delta") or {}).get("content")
                if content:
                    pieces.append(content)
                    emit(content)
    if not done or finish_reason is None:
        raise ChatError("回答连接提前结束，本轮未加入对话记录。")
    answer = "".join(pieces)
    if not answer:
        raise ChatError("服务未返回回答文本，本轮未加入对话记录。")
    return answer, finish_reason


def help_text():
    print("直接输入提示词并回车即可提问；回答会逐段显示。")
    print("/clear 清空对话   /help 帮助   /exit 退出")
    print("回答期间按 Ctrl+C 中断本次请求；输入时按 Ctrl+D 退出。")
    print("首次算子编译可能等待数分钟；回答生成到模型结束，不另设输出 token 上限。")


def emit(text):
    print(text, end="", flush=True)


def run_chat(args):
    check_service(args.base_url, args.model)
    if args.check:
        print("连接成功，模型 {} 可用。".format(args.model), flush=True)
        return 0
    history = []
    if args.prompt is None:
        print("DeepSeek-V4-Flash 终端聊天（已连接）", flush=True)
        help_text()
    while True:
        if args.prompt is not None:
            prompt = args.prompt
        else:
            try:
                prompt = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                return 0
        if not prompt:
            if args.prompt is not None:
                raise ChatError("提示词不能为空。")
            continue
        if args.prompt is None and prompt.startswith("/"):
            command, _, value = prompt.partition(" ")
            if command in ("/exit", "/quit"):
                print("已退出。")
                return 0
            if command in ("/clear", "/reset"):
                history.clear()
                print("已清空对话。")
            elif command == "/help":
                help_text()
            else:
                print("未知命令，输入 /help 查看帮助。")
            continue
        request_messages = history + [{"role": "user", "content": prompt}]
        try:
            with FirstTokenStatus() as status:
                first = True

                def show_content(content):
                    nonlocal first
                    if first:
                        status.stop()
                        print("\n模型：", end="", flush=True)
                        first = False
                    emit(content)

                answer, reason = generate(args, request_messages, show_content)
        except KeyboardInterrupt:
            print("\n[已中断本次请求，本轮未加入对话记录。]", flush=True)
            if args.prompt is not None:
                return 130
            continue
        except (ChatError, OSError, http.client.HTTPException, UnicodeError, ValueError) as exc:
            print("\n[请求失败：{}]".format(exc), flush=True)
            if args.prompt is not None:
                return 1
            continue
        print(flush=True)
        history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
        if reason == "length":
            print("[服务端因长度条件结束了回答；客户端没有设置输出 token 上限。]", flush=True)
        if args.prompt is not None:
            return 0


def positive_timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("超时必须是正数。")
    return value


def temperature(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise argparse.ArgumentTypeError("temperature 必须在 0—2 之间。")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="连接当前 DeepSeek 服务，直接输入提示词并流式显示回答。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=temperature, default=0.6)
    parser.add_argument("--timeout", type=positive_timeout, default=1800)
    once = parser.add_mutually_exclusive_group()
    once.add_argument("--prompt", help="直接回答这一条提示词后退出。")
    once.add_argument("--check", action="store_true", help="仅检查服务连接，不发送生成请求。")
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(args.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        parser.error("--base-url 必须是 http 或 https 地址。")
    try:
        return run_chat(args)
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
        return 130
    except (ChatError, OSError, http.client.HTTPException, ValueError) as exc:
        print("连接失败：{}".format(exc), file=sys.stderr)
        print("请确认服务器上的推理服务已启动，默认地址为 http://127.0.0.1:8000/v1。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
