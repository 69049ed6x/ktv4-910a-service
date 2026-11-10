"""Manage this deployment only; keep child output in a bounded memory buffer."""
import argparse
from collections import deque
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import psutil
from settings import APP, CHECKPOINT, PORT, PYTHON, ROOT, RUN, atomic_json

STATE = RUN / 'state.json'


def read_state():
    try:
        return json.loads(STATE.read_text())
    except FileNotFoundError:
        return {'phase': 'stopped'}


def identify(proc):
    return {'pid': proc.pid, 'created': psutil.Process(proc.pid).create_time()}


def recorded_process(record, marker):
    if not record:
        return None
    try:
        proc = psutil.Process(record['pid'])
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        if abs(proc.create_time() - record['created']) > .01:
            return None
        if marker not in ' '.join(proc.cmdline()):
            raise RuntimeError(f'Refusing unrelated PID {proc.pid}')
        return proc
    except psutil.NoSuchProcess:
        return None


def terminate(record, marker):
    proc = recorded_process(record, marker)
    if proc is None:
        return
    processes = proc.children(recursive=True) + [proc]
    for item in reversed(processes):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=12)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=4)


def api_healthy():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def clear_driver_logs():
    # Only call while this deployment's API and worker processes are stopped.
    # CANN messages from new children go to the bounded supervisor pipe.
    for path in (RUN / 'ascend-log').rglob('*.log'):
        path.unlink(missing_ok=True)


def supervisor():
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    state = {'phase': 'loading_workers', 'session': uuid.uuid4().hex,
             'supervisor': identify(psutil.Process()), 'workers': None, 'api': None,
             'started': time.time()}
    rings = {name: deque(maxlen=160) for name in ('workers', 'api')}
    children = {}

    def save():
        atomic_json(STATE, state)

    def launch(name, script):
        env = dict(os.environ, KTV4_SESSION=state['session'])
        child = subprocess.Popen(['bash', str(APP / script)], cwd=RUN, env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = child
        state[name] = identify(child)
        save()

        def drain():
            # Bound memory even when a library emits a very long line.
            while True:
                chunk = child.stdout.readline(8192)
                if not chunk:
                    break
                rings[name].append(chunk.decode('utf-8', errors='replace').rstrip())
            child.stdout.close()
        threading.Thread(target=drain, daemon=True).start()

    def check_children():
        for name, child in children.items():
            code = child.poll()
            if code is not None:
                raise RuntimeError(f'{name} exited with code {code}')

    def ranks_ready():
        for rank in range(4):
            path = RUN / f'rank{rank}.json'
            if not path.exists():
                return False
            data = json.loads(path.read_text())
            if data['session'] != state['session']:
                return False
        return True

    try:
        clear_driver_logs()
        for path in RUN.glob('rank*.json'):
            path.unlink()
        for path in RUN.glob('result_*.json'):
            path.unlink()
        for path in RUN.glob('worker_error_rank*.json'):
            path.unlink()
        (RUN / 'command.json').unlink(missing_ok=True)
        save()
        launch('workers', 'launch_workers.sh')
        deadline = time.monotonic() + 2400
        while not stopped.is_set() and not ranks_ready():
            check_children()
            if time.monotonic() > deadline:
                raise TimeoutError('Four ranks did not load within 2400 seconds')
            stopped.wait(.25)
        if not stopped.is_set():
            state['phase'] = 'starting_api'
            save()
            launch('api', 'launch_api.sh')
            deadline = time.monotonic() + 600
            while not stopped.is_set() and not api_healthy():
                check_children()
                if time.monotonic() > deadline:
                    raise TimeoutError('API did not start within 600 seconds')
                stopped.wait(.5)
        if not stopped.is_set():
            state['phase'] = 'ready'
            state['ready_at'] = time.time()
            save()
            while not stopped.wait(1):
                check_children()
    except BaseException as exc:
        state['phase'] = 'failed'
        state['error'] = str(exc)
        state['last_output'] = {name: list(ring) for name, ring in rings.items()}
        save()
    finally:
        terminate(state.get('api'), 'vllm.entrypoints.openai.api_server')
        terminate(state.get('workers'), str(APP / 'worker.py'))
        clear_driver_logs()
        if state['phase'] != 'failed':
            state['phase'] = 'stopped'
        state['api'] = state['workers'] = state['supervisor'] = None
        save()
        for path in RUN.glob('rank*.json'):
            path.unlink(missing_ok=True)
        for path in RUN.glob('result_*.json'):
            path.unlink(missing_ok=True)
        (RUN / 'command.json').unlink(missing_ok=True)


def up():
    if not (CHECKPOINT / 'config.json').is_file() or not any(CHECKPOINT.glob('*.safetensors')):
        raise RuntimeError('Model config or safetensors weights missing: ' + str(CHECKPOINT))
    state = read_state()
    proc = recorded_process(state.get('supervisor'), str(APP / 'service.py'))
    if not proc:
        # An existing API/worker tree must be explicitly stopped, never adopted.
        if recorded_process(state.get('workers'), str(APP / 'worker.py')):
            raise RuntimeError('Stale managed workers exist; run stop.sh first')
        if recorded_process(state.get('api'), 'vllm.entrypoints.openai.api_server'):
            raise RuntimeError('Stale managed API exists; run stop.sh first')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', PORT)) == 0:
                raise RuntimeError(f'Port {PORT} is already in use by an unmanaged service')
        STATE.unlink(missing_ok=True)
        child = subprocess.Popen([str(PYTHON), '-B', '-u', str(APP / 'service.py'), '_supervise'],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        proc = psutil.Process(child.pid)
    started = time.monotonic()
    shown = -1
    while True:
        state = read_state()
        phase = state.get('phase')
        if phase == 'ready' and api_healthy():
            print('四卡模型和 API 已就绪。', flush=True)
            return
        if phase == 'failed':
            raise RuntimeError(json.dumps(state, ensure_ascii=False, indent=2))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError('Service supervisor exited; inspect run/state.json')
        elapsed = int(time.monotonic() - started)
        if elapsed // 10 != shown:
            shown = elapsed // 10
            ready_count = len(list(RUN.glob('rank*.json')))
            print(f'正在启动：{phase}，四卡就绪 {ready_count}/4，已等待 {elapsed} 秒。', flush=True)
        if elapsed > 3100:
            raise TimeoutError('Service startup exceeded 3100 seconds')
        time.sleep(.5)


def down():
    state = read_state()
    proc = recorded_process(state.get('supervisor'), str(APP / 'service.py'))
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except psutil.TimeoutExpired:
            # Validate each recorded child again before any forced cleanup.
            terminate(state.get('api'), 'vllm.entrypoints.openai.api_server')
            terminate(state.get('workers'), str(APP / 'worker.py'))
            proc.kill()
    else:
        terminate(state.get('api'), 'vllm.entrypoints.openai.api_server')
        terminate(state.get('workers'), str(APP / 'worker.py'))
    clear_driver_logs()
    atomic_json(STATE, {'phase': 'stopped', 'stopped_at': time.time()})
    print('本项目服务已停止。', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['up', 'down', 'status', '_supervise'])
    args = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    if args.action == '_supervise':
        supervisor()
        return
    if args.action == 'status':
        state = read_state()
        state['api_healthy'] = api_healthy()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    with (RUN / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        {'up': up, 'down': down}[args.action]()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
