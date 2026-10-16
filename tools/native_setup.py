"""Explicit installation/restart of the four existing containers, never new services."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, build_opener
import uuid
import os

from dabs_native.contracts import validate_capabilities, validate_observation, require_control_semantics
from dabs_ascend.common import read_json, write_json
from dabs_ascend.telemetry import metric_pressure, parse_metrics
from dabs_ascend.native_timing import require_timing


# Executed only if the live service lacks usable native capabilities.
INSPECT = '''
import json, sysconfig
from pathlib import Path
import dabs_native.install as installer
target = Path(sysconfig.get_paths()['purelib']) / 'dabs_native'
check = installer.check()
receipt = target / 'installation.json'
previous = json.loads(receipt.read_text()) if receipt.exists() else {}
result = dict(state=('update' if target.exists() else 'fresh') if check['compatible'] else 'incompatible',
              check=check, backup=previous.get('backup'))
print(json.dumps(result))
'''


class Installer:
    def __init__(self, workflow, ready_timeout=900, verify_restart=False):
        self.w = workflow
        self.timeout, self.verify_restart = ready_timeout, verify_restart
        self.python = os.environ.get('CONTROLLER_PYTHON', '/usr/local/python3.12.13/bin/python')
        self.opener = build_opener(ProxyHandler({}))
        self.backups = {}
        self.last_ready_error = None

    @staticmethod
    def container(replica):
        return 'wushibo-qwen-small-npu' + str(replica)

    def capture(self, args, input=None):
        result = subprocess.run(['sudo', 'docker', *args], input=input,
                                capture_output=True, text=True, encoding='utf-8', timeout=120)
        if result.returncode:
            raise RuntimeError('docker ' + args[0] + ' failed: ' + result.stderr[-2000:])
        return result.stdout

    def inspect(self, replica):
        container = self.container(replica)
        raw = json.loads(self.capture(['inspect', container]))[0]
        ports = raw.get('NetworkSettings', {}).get('Ports', {}).get('8000/tcp') or []
        if raw.get('Name') != '/' + container or not raw.get('State', {}).get('Running') or not any(
                p.get('HostPort') == str(8001 + replica) for p in ports):
            raise ValueError('现有容器未运行或端口不匹配: ' + container + '；不创建替代服务。')
        # Save only allowlisted identity, not Docker Env/Cmd credentials.
        identity = {'name': container, 'id': raw['Id'], 'image': raw['Image'], 'ports': ports}
        stage = '/tmp/dabs-setup-' + uuid.uuid4().hex
        self.capture(['exec', container, 'mkdir', '-p', stage])
        self.capture(['cp', str(self.w.root / 'experiments/ascend/dabs_native'), container + ':' + stage + '/dabs_native'])
        result = json.loads(self.capture(['exec', '-e', 'PYTHONPATH=' + stage, container,
                            self.python, '-c', INSPECT]))
        result['container'] = identity
        write_json(self.w.out / ('install-state-' + str(replica) + '.json'), result)
        return result

    def http(self, replica, route, payload=None):
        headers = {}
        if payload is not None:
            headers = {'Content-Type': 'application/json'}
        request = Request('http://127.0.0.1:' + str(8001 + replica) + route,
                          data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        with self.opener.open(request, timeout=10) as response:
            data = response.read(4_000_001)
            if len(data) > 4_000_000:
                raise ValueError('service response too large')
            return data.decode('utf-8')

    def idle(self, replica):
        pressure = metric_pressure(parse_metrics(self.http(replica, '/metrics')))
        write_json(self.w.out / ('idle-' + str(replica) + '-' + uuid.uuid4().hex[:8] + '.json'), pressure)
        if pressure['running'] != 0 or pressure['waiting'] != 0:
            raise ValueError('副本 ' + str(replica) + ' 尚有请求或空闲状态未知，停止安装/重启: ' + str(pressure))

    def ready(self, replica):
        try:
            self.http(replica, '/health')
            row = json.loads(self.http(replica, '/dabs/v1/control', {'op': 'capabilities'}))
            validate_capabilities(row)
            require_control_semantics(row)
            require_timing(row)
            snapshot = json.loads(self.http(replica, '/dabs/v1/control', {'op': 'snapshot'}))
            validate_observation(snapshot)
            self.last_ready_error = None
            return row
        except (URLError, OSError, ValueError, KeyError) as exc:
            self.last_ready_error = str(exc)
            return None

    def wait_ready(self, replica):
        deadline = time.monotonic() + self.timeout
        while True:
            row = self.ready(replica)
            if row:
                write_json(self.w.out / ('loaded-' + str(replica) + '-' + uuid.uuid4().hex[:8] + '.json'), row)
                print('副本 ' + str(replica) + ' 原生引擎已就绪。', flush=True)
                return
            if time.monotonic() >= deadline:
                raise ValueError('副本 ' + str(replica) + ' 在 ' + str(self.timeout) + ' 秒内未提供所需原生接口: ' + str(self.last_ready_error))
            print('等待副本 ' + str(replica) + ' 加载模型和原生引擎……', flush=True)
            time.sleep(5)

    def restart(self, replica):
        # Shared controller locks exclude our experiments; refuse observed external work.
        self.idle(replica)
        container = self.container(replica)
        command = ['sudo', 'docker', 'restart', '--time', '120', container]
        logfile = self.w.out / ('restart-' + str(replica) + '-' + uuid.uuid4().hex[:8] + '.txt')
        step = {'mode': 'restart-' + str(replica), 'label': '重启现有服务 ' + container,
                'status': 'running', 'command': command, 'log': self.w.relative(logfile)}
        self.w.record['steps'].append(step)
        self.w.save()
        print(step['label'], flush=True)
        code = self.w.launch(command, logfile)
        step.update(exit_code=code, status='passed' if code == 0 else 'failed')
        self.w.save()
        if code:
            raise ValueError('重启失败: ' + container)
        self.wait_ready(replica)

    def rollback_commands(self):
        commands = []
        for replica, backup in self.backups.items():
            commands += ['bash run_dabs_r234.sh native-rollback --replica ' + str(replica) + ' --backup ' + shlex.quote(backup),
                         'sudo docker restart ' + self.container(replica)]
        return commands

    def save_backups(self):
        write_json(self.w.out / 'backups.json', self.backups)
        (self.w.out / 'ROLLBACK.txt').write_text('仅在需要撤销安装时执行；按副本选择对应两行：\n' +
            '\n'.join(self.rollback_commands()) + '\n', encoding='utf-8')
        self.w.record['rollback_file'] = self.w.relative(self.w.out / 'ROLLBACK.txt')
        self.w.save()

    def run(self):
        print('先连接现有服务；仅对缺少可用原生接口的副本安装补丁并逐个重启。', flush=True)
        self.save_backups()
        pending = []
        selected = [self.w.replica] if getattr(self.w, 'replica', None) is not None else list(range(4))
        for replica in selected:
            if self.ready(replica):
                print('副本 ' + str(replica) + ' 原生接口已可用，跳过安装和重启。', flush=True)
                continue
            print('副本 ' + str(replica) + ' 需要加载适配: ' + str(self.last_ready_error), flush=True)
            state = self.inspect(replica)
            pending.append((replica, state))
            if state.get('backup'):
                self.backups[replica] = state['backup']
        self.save_backups()
        errors = {r: s.get('check', {}).get('errors', []) for r, s in pending if s['state'] == 'incompatible'}
        if errors:
            raise ValueError('缺少安装所需接口，尚未修改或重启服务: ' + json.dumps(errors, ensure_ascii=False))
        for replica, _ in pending:
            self.idle(replica)
        for replica, state in pending:
            self.idle(replica)
            try:
                directory = self.w.step('native-install', '安装/更新副本 ' + str(replica), ['--replica', str(replica)])
            except Exception:
                try:
                    after = self.inspect(replica)
                    if after.get('backup'):
                        self.backups[replica] = after['backup']
                        self.save_backups()
                except Exception:
                    pass
                raise
            result = read_json(directory / 'result.json')['replicas'][0]['result']
            self.backups[replica] = result['backup']
            self.save_backups()
            self.restart(replica)
        if len(selected) == 1:
            self.w.step('native-check', '核验所选服务原生能力', ['--replica', str(selected[0])])
        else:
            self.w.step('native-check-four', '核验四服务的真实原生能力')
        if self.verify_restart:
            self.w.step('native-restart-arm', '保存副本 0 重启前身份')
            self.restart(0)
            self.w.step('native-restart-check', '验证副本 0 拒绝旧引擎身份动作')
        self.w.record['next_commands'] = ['bash run_native.sh smoke']
