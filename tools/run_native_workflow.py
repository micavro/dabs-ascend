#!/usr/bin/env python3
"""One command for native tests, one for the frozen 96-condition experiment."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
sys.path.insert(0, str(ROOT / 'tools'))
from dabs_ascend.common import read_json, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


class Stopped(Exception):
    def __init__(self, reason, commands=()):
        super().__init__(reason)
        self.commands = list(commands)


class Workflow:
    def __init__(self, root, mode, invalidate_cache=False, verify_restart=False, ready_timeout=900, replica=None, input_path=None):
        self.root, self.mode = Path(root).resolve(), mode
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = self.root / 'outputs' / ('native_' + mode + '_' + stamp)
        self.out.mkdir(parents=True)
        self.invalidate_cache = invalidate_cache
        self.verify_restart, self.ready_timeout = verify_restart, ready_timeout
        self.replica, self.input_path = replica, input_path
        self.record = {'mode': mode, 'status': 'running', 'started_utc': utc(),
                       'output': self.relative(self.out), 'steps': [], 'next_commands': [],
                       'explicit_cache_invalidation': invalidate_cache, 'explicit_restart_test': verify_restart}

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def host_path(self, value):
        """Only accept evidence under this checkout's outputs, including /project paths."""
        raw = str(value)
        if raw.startswith('/project/'):
            raw = raw[len('/project/'):]
        path = (self.root / raw).resolve()
        if not path.is_relative_to(self.root / 'outputs'):
            raise ValueError('Evidence must be under this project outputs: ' + raw)
        return path

    def container_path(self, path):
        return '/project/' + self.relative(path)

    def save(self):
        write_json(self.out / 'run.json', self.record)
        lines = ['DABS 原生' + {'test': '测试', 'experiment': '实验', 'install': '安装',
                 'smoke': '计时关联短诊断', 'calibrate': '单副本完整标定', 'audit': '旧事件离线审计',
                 'explore': '带预测偏差的四条件试跑（非正式验收）'}[self.mode],
                 '状态: ' + self.record['status'], '结果目录: ' + str(self.out)]
        for step in self.record['steps']:
            lines.append(step['label'] + ': ' + step['status'] + '；日志 ' + step['log'])
        if self.record.get('optional_tests_not_run'):
            lines.append('未运行的专项（不计通过）: ' + ', '.join(self.record['optional_tests_not_run']))
        if self.record.get('reason'):
            lines.append('停止原因: ' + self.record['reason'])
        for field in ('acceptance', 'experiment_output', 'rollback_file'):
            if field in self.record:
                lines.append(field + ': ' + self.record[field])
        if self.record['next_commands']:
            lines += ['下一步（按顺序，涉及重启时等待服务重新就绪）:', *self.record['next_commands']]
        lines += ['结果保留为普通文本/JSON/CSV；只需复制本摘要和失败步骤日志，不打包、不上传。']
        (self.out / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def launch(self, command, logfile):
        # Stream output and retain failures, including Docker errors before a worker starts.
        env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        with logfile.open('w', encoding='utf-8', buffering=1) as log:
            log.write('$ ' + shlex.join(command) + '\n')
            with subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                  errors='replace', env=env, bufsize=1) as child:
                try:
                    for line in child.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    return child.wait()
                except KeyboardInterrupt:
                    # Terminal Ctrl-C is delivered to the whole foreground group;
                    # wait for the controller's own cleanup, never signal a model service.
                    child.wait()
                    raise

    def step(self, mode, label, extra=(), native=True):
        number = len(self.record['steps']) + 1
        directory = self.out / (str(number).zfill(2) + '-' + mode)
        logfile = self.out / (str(number).zfill(2) + '-' + mode + '.txt')
        command = ['bash', str(self.root / 'run_dabs_r234.sh'), mode, *extra]
        if native:
            command += ['--out', self.relative(directory)]
        step = {'mode': mode, 'label': label, 'status': 'running', 'command': command,
                'log': self.relative(logfile), 'output': self.relative(directory), 'started_utc': utc()}
        self.record['steps'].append(step)
        self.save()
        print('\n[' + str(number) + '] ' + label, flush=True)
        code = self.launch(command, logfile)
        step.update(status='passed' if code == 0 else 'failed', exit_code=code, finished_utc=utc())
        self.save()
        if code:
            detail = ''
            if (directory / 'result.json').exists():
                result = read_json(directory / 'result.json')
                detail = json.dumps(result.get('missing') or result.get('failures') or result.get('checks'), ensure_ascii=False)
            commands = []
            if mode == 'native-accept':
                if 'native-restart-check' in detail:
                    commands += ['bash run_dabs_r234.sh native-restart-arm',
                                 'sudo docker restart wushibo-qwen-small-npu0',
                                 'bash run_dabs_r234.sh native-restart-check']
                if 'native-invalidate' in detail:
                    commands += ['bash run_native.sh test --invalidate-cache']
                elif commands:
                    commands += ['bash run_native.sh test']
            if mode == 'native-check-four':
                detail += ' 原生接口缺项见本步骤日志；安装命令会自动跳过已可用的副本。'
                commands = ['bash run_native.sh install', 'bash run_native.sh test']
            raise Stopped(label + '失败。' + detail + ' 日志: ' + self.relative(logfile), commands)
        return directory


    def install(self):
        from native_setup import Installer
        installer = Installer(self, self.ready_timeout, self.verify_restart)
        try:
            installer.run()
        except Exception as exc:
            raise Stopped(str(exc) + ' 已完成的副本及备份保留；回滚命令见本目录 ROLLBACK.txt。') from exc

    def audit(self):
        self.step('native-timing-audit', '离线统计已有原始事件（不发推理）', ['--input', str(self.input_path)])

    def explore(self):
        source = self.input_path
        if source is None:
            roots = [self.root, *sorted(self.root.parent.glob('dabs-ascend-main-*'))]
            candidates = set()
            for root in roots:
                for pattern in ('native_calibrate_*/01-native-calibrate-one', 'native_test_*/04-native-calibrate'):
                    for path in (root / 'outputs').glob(pattern):
                        if (path / 'native_profile.json').is_file() and (path / 'calibration_plan.json').is_file():
                            plan = read_json(path / 'calibration_plan.json')
                            if plan.get('diagnostic_only') is False:
                                candidates.add(path.resolve())
            if not candidates:
                raise Stopped('没有找到已有完整标定；用 bash run_native.sh explore --input 原标定步骤目录 指定（不要指定 smoke）。')
            source = max(candidates, key=lambda p: (p / 'native_profile.json').stat().st_mtime_ns)
        source = Path(source).resolve()
        staged = self.out / 'calibration_source'
        staged.mkdir()
        # Plain local copies make old-directory evidence available to the CPU container.
        for name in ('native_profile.json', 'calibration_plan.json', 'calibration_samples.json',
                     'capabilities.json', 'native_events.jsonl'):
            shutil.copyfile(source / name, staged / name)
        self.record.update(calibration_source=str(source), exploratory=True, formal_acceptance=False)
        self.save()
        paths = {'native-calibrate': self.container_path(staged)}
        self.step('native-check-four', '检查现有四服务接口（不安装、不重启）')
        for mode in ('native-probe-one', 'native-probe-four'):
            paths[mode] = self.container_path(self.step(mode, '确认实际原生机制：' + mode))
        evidence = self.out / 'exploratory_evidence.json'
        write_json(evidence, {'paths': paths, 'exploratory': True, 'formal_acceptance': False})
        result_dir = self.out / 'r234_exploratory'
        self.record['experiment_output'] = self.relative(result_dir)
        self.step('benchmark', '忽略已记录预测误差，试跑原计划前四条件 RR/SLL/Ayo/Ours',
                  ['--execution-scope', 'native', '--native-evidence', self.container_path(evidence),
                   '--exploratory-calibration', '--max-conditions', '4',
                   '--out', self.container_path(result_dir)], native=False)
        result = read_json(result_dir / 'status.json')
        summaries = read_json(result_dir / 'index.json')
        if (result.get('status') != 'paused' or result.get('completed_conditions') != 4 or
                len(summaries) != 4 or not all(s.get('functional_pass') is True and s.get('exploratory') is True for s in summaries)):
            raise Stopped('四条件试跑未完整结束；保留实际失败，不计正式验收通过。')
        self.record['completed_conditions'] = 4

    def smoke(self):
        self.step('native-calibration-smoke', '单副本短诊断（不能代替完整标定）', ['--replica', str(self.replica or 0)])
        self.record['next_commands'] = ['bash run_native.sh calibrate --replica ' + str(self.replica or 0)]

    def calibrate(self):
        self.step('native-calibrate-one', '单副本完整标定与留出验证', ['--replica', str(self.replica or 0)])
        self.record['next_commands'] = ['bash run_native.sh install', 'bash run_native.sh test --invalidate-cache']

    def test(self):
        # Do not silently install hooks or restart services as part of ordinary tests.
        self.step('native-check-four', '检查四个服务的原生能力（不发推理请求）')
        paths = {}
        for mode, label in [('native-probe-one', '单副本机制测试'),
                            ('native-probe-four', '四副本机制测试'),
                            ('native-calibrate', '独立标定与留出验证')]:
            paths[mode] = self.container_path(self.step(mode, label))
        if self.invalidate_cache:
            paths['native-invalidate'] = self.container_path(
                self.step('native-invalidate', '明确指定的缓存失效测试（清空副本 0 前缀缓存）'))
        evidence = self.out / 'evidence.json'
        write_json(evidence, {'paths': paths})
        accepted = self.step('native-accept', '核验原始证据、当前服务身份和标定',
                             ['--evidence', self.relative(evidence)]) / 'acceptance.json'
        result = read_json(accepted)
        if result.get('pass') is not True or result.get('live_verified') is not True or result.get('missing') != []:
            raise Stopped('验收未提供 pass=true、live_verified=true、missing=[]，不放行实验。')
        self.record['acceptance'] = self.relative(accepted)
        self.record['optional_tests_not_run'] = result.get('not_run', [])

    def experiment(self):
        latest = self.root / 'outputs/native-entry-last-test.json'
        if not latest.exists():
            raise Stopped('尚未完成一键原生测试。', ['bash run_native.sh test'])
        tested = self.host_path(read_json(latest)['output'])
        record = read_json(tested / 'run.json')
        if record.get('mode') != 'test' or record.get('status') != 'passed' or not record.get('acceptance'):
            raise Stopped('最近一次测试未通过；先查看 ' + str(tested / 'SUMMARY.txt'), ['bash run_native.sh test'])
        evidence = self.host_path(record['acceptance'])
        if not evidence.is_file():
            raise Stopped('测试原始验收记录缺失。', ['bash run_native.sh test'])
        # This pointer is convenience only: R234 recomputes raw evidence and
        # verifies live engine identities and resources before canaries or formal traffic.
        result_dir = self.out / 'r234_native'
        self.record['acceptance'] = self.relative(evidence)
        self.record['experiment_output'] = self.relative(result_dir)
        self.step('benchmark', '重新核验原生证据，然后运行全部 96 条件',
                  ['--execution-scope', 'native', '--native-evidence', self.container_path(evidence),
                   '--out', self.container_path(result_dir)], native=False)
        result = read_json(result_dir / 'status.json')
        if result.get('status') != 'complete' or result.get('completed_conditions') != 96 or result.get('functional_test_pass') is not True:
            raise Stopped('实验没有完整通过全部 96 条件；保留实际结果，不标记完成。')

    def run(self):
        # A new failed/interrupted test cannot leave an older PASS as the default.
        write_json(self.root / ('outputs/native-entry-last-' + self.mode + '.json'),
                   {'output': self.relative(self.out)})
        self.save()
        print('本次结果: ' + str(self.out), flush=True)
        try:
            getattr(self, self.mode)()
            self.record['status'] = 'passed'
            if self.mode == 'test':
                self.record['next_commands'] = ['bash run_native.sh experiment']
            code = 0
        except Stopped as exc:
            self.record.update(status='blocked', reason=str(exc), next_commands=exc.commands)
            code = 2
        except BaseException as exc:
            self.record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', reason=str(exc))
            (self.out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            code = 130 if isinstance(exc, KeyboardInterrupt) else 2
        for step in self.record['steps']:
            if step['status'] == 'running':
                step.update(status=self.record['status'], finished_utc=utc())
        self.record['finished_utc'] = utc()
        self.save()
        print('\n' + (self.out / 'SUMMARY.txt').read_text(encoding='utf-8'), flush=True)
        return code


def main():
    parser = argparse.ArgumentParser(description='现场一键入口：install 安装并重启现有服务；test 测试验收；experiment 跑全部 96 条件。')
    parser.add_argument('mode', choices=('install', 'smoke', 'calibrate', 'audit', 'explore', 'test', 'experiment'))
    parser.add_argument('--replica', type=int, choices=range(4), help='install/smoke/calibrate: select one existing service')
    parser.add_argument('--input', type=Path, help='audit: existing calibration directory (read-only)')
    parser.add_argument('--invalidate-cache', action='store_true', help='仅 test：明确执行副本 0 缓存清空/失效测试；默认不执行')
    parser.add_argument('--verify-restart', action='store_true', help='仅 install：明确额外重启副本 0 并验证旧身份失效')
    parser.add_argument('--ready-timeout', type=int, default=900, help='install 每个服务加载等待秒数，默认 900')
    args = parser.parse_args()
    if args.replica is not None and args.mode not in ('install', 'smoke', 'calibrate'):
        parser.error('--replica 仅用于 install/smoke/calibrate')
    if (args.mode == 'audit' and not args.input) or (args.input and args.mode not in ('audit', 'explore')):
        parser.error('audit 必须提供 --input；explore 可用 --input 指定已有标定目录')
    if args.verify_restart and args.replica not in (None, 0):
        parser.error('--verify-restart 仅支持副本 0')
    if args.invalidate_cache and args.mode != 'test':
        parser.error('--invalidate-cache 仅用于 test')
    if args.verify_restart and args.mode != 'install':
        parser.error('--verify-restart 仅用于 install')
    if args.ready_timeout < 1:
        parser.error('--ready-timeout 必须为正数')
    if sys.platform != 'linux':
        parser.error('请在昇腾 Linux 宿主机运行；开发机 CPU 检查仍用 python tools/run_ascend.py test')
    from dabs_ascend.runtime import device_locks
    ROOT.joinpath('outputs').mkdir(exist_ok=True)
    # Separate entry lock excludes two wrappers; the test also shares the
    # existing controller device locks. Benchmark obtains those itself.
    import fcntl
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((ROOT / 'outputs/.native-entry.lock').open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.mode in ('test', 'install', 'smoke', 'calibrate'):
                stack.enter_context(device_locks(range(4), ROOT / 'outputs/.existing_controller_locks'))
            return Workflow(ROOT, args.mode, args.invalidate_cache, args.verify_restart, args.ready_timeout,
                            args.replica, args.input).run()
    except BlockingIOError:
        print('已有测试或实验使用这些服务；请等待当前任务结束。', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
