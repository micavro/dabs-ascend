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
from dabs_ascend.copy_log import emit, folder


def utc():
    return datetime.now(timezone.utc).isoformat()


class Stopped(Exception):
    def __init__(self, reason, commands=()):
        super().__init__(reason)
        self.commands = list(commands)


class Workflow:
    def __init__(self, root, mode, invalidate_cache=False, verify_restart=False, ready_timeout=900, replica=None, input_path=None, max_conditions=None):
        self.root, self.mode = Path(root).resolve(), mode
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = self.root / 'outputs' / ('native_' + mode + '_' + stamp)
        self.out.mkdir(parents=True)
        self.invalidate_cache = invalidate_cache
        self.verify_restart, self.ready_timeout = verify_restart, ready_timeout
        self.replica, self.input_path = replica, input_path
        self.max_conditions = (8 if mode == 'verify' else 4) if max_conditions is None else max_conditions
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
        lines = ['DABS 原生' + {'test': '测试', 'verify': '规划验证（原计划前若干条件）', 'experiment': '实验', 'install': '安装',
                 'smoke': '计时关联短诊断', 'calibrate': '单副本完整标定', 'audit': '旧事件离线审计',
                 'refit': '旧数据离线重拟合（诊断，不作新验收）',
                 'explore': '带预测偏差的实验（非正式验收）'}[self.mode],
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
        lines += ['复制用摘要（运行中持续追加）: ' + str(folder(self.out) / 'COPY.txt'),
                  '原始证据保留；普通文本，不打包、不上传。']
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
        emit(self.out, 'STEP', n=number, mode=mode, command=shlex.join(command))
        print('\n[' + str(number) + '] ' + label, flush=True)
        code = self.launch(command, logfile)
        step.update(status='passed' if code == 0 else 'failed', exit_code=code, finished_utc=utc())
        emit(self.out, 'STEP_END', n=number, mode=mode, exit=code)
        if (directory / 'result.json').exists():
            result = read_json(directory / 'result.json')
            emit(self.out, 'CHECK', mode=mode, **{k: result[k] for k in
                ('pass', 'live_verified', 'inference_executed', 'checks', 'missing', 'failures') if k in result})
        if code and logfile.exists():
            # Retain the useful tail once, without copying huge JSON/trace dumps.
            tail = logfile.read_text(encoding='utf-8').splitlines()[-8:]
            emit(self.out, 'ERROR_TAIL', step=mode, lines=[line[:500] for line in tail])
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

    def refit(self):
        self.step('native-refit', '从原始事件比较旧模型与启动项模型（不发推理）', ['--input', str(self.input_path)])
        self.record.update(diagnostic_only=True, formal_acceptance=False, fresh_holdout_required=True)
        self.record['next_commands'] = ['bash run_native.sh calibrate']

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
        self.record.update(calibration_source=str(source), exploratory=True, formal_acceptance=False,
                           requested_conditions=self.max_conditions)
        self.save()
        paths = {'native-calibrate': self.container_path(staged)}
        self.step('native-check-four', '检查现有四服务接口（不安装、不重启）')
        for mode in ('native-probe-one', 'native-probe-four'):
            paths[mode] = self.container_path(self.step(mode, '确认实际原生机制：' + mode))
        evidence = self.out / 'exploratory_evidence.json'
        write_json(evidence, {'paths': paths, 'exploratory': True, 'formal_acceptance': False})
        result_dir = self.out / 'r234_exploratory'
        self.record['experiment_output'] = self.relative(result_dir)
        self.step('benchmark', '保留预测偏差，运行原计划前 ' + str(self.max_conditions) + ' 个条件',
                  ['--execution-scope', 'native', '--native-evidence', self.container_path(evidence),
                   '--exploratory-calibration', '--max-conditions', str(self.max_conditions),
                   '--out', self.container_path(result_dir)], native=False)
        result = read_json(result_dir / 'status.json')
        summaries = read_json(result_dir / 'index.json')
        expected_status = 'complete' if self.max_conditions == 96 else 'paused'
        if (result.get('status') != expected_status or result.get('completed_conditions') != self.max_conditions or
                len(summaries) != self.max_conditions or not all(s.get('functional_pass') is True and s.get('exploratory') is True for s in summaries)):
            raise Stopped('所选条件没有完整结束；保留实际失败，不计正式验收通过。')
        self.record['completed_conditions'] = self.max_conditions

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

    def import_test(self):
        """Copy plain evidence from an old checkout; no PASS flag grants acceptance."""
        tested = Path(self.input_path).resolve()
        if tested.parent.name != 'outputs':
            raise Stopped('--input 必须指向旧项目 outputs/native_test_* 测试结果目录。')
        source_root = tested.parent.parent
        record = read_json(tested / 'run.json')
        if record.get('mode') != 'test' or record.get('status') != 'passed':
            raise Stopped('指定测试没有完成；保留原始失败，不放行实验。')

        def source_path(value):
            raw = str(value)
            if raw.startswith('/project/'):
                raw = raw[len('/project/'):]
            path = (source_root / raw).resolve()
            if not path.is_relative_to(source_root / 'outputs'):
                raise Stopped('证据路径不在原项目 outputs 下: ' + raw)
            return path

        original = source_path(record['acceptance'])
        accepted = read_json(original)
        paths = {}
        for mode, value in accepted['paths'].items():
            if mode not in ('native-probe-one', 'native-probe-four', 'native-calibrate',
                            'native-invalidate', 'native-restart-check'):
                raise Stopped('未知原始证据类型: ' + mode)
            source = source_path(value)
            target = self.out / 'imported_test' / mode
            target.mkdir(parents=True)
            for path in source.iterdir():
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.csv', '.md'):
                    shutil.copyfile(path, target / path.name)
            paths[mode] = self.container_path(target)
        shutil.copyfile(original, self.out / 'original_acceptance.json')
        evidence = self.out / 'imported_evidence.json'
        write_json(evidence, {'paths': paths, 'source_test': str(tested),
                              'requires_raw_and_live_revalidation': True})
        self.record['imported_test_source'] = str(tested)
        return evidence

    def test_evidence(self):
        latest = self.root / 'outputs/native-entry-last-test.json'
        if self.input_path:
            evidence = self.import_test()
        elif not latest.exists():
            raise Stopped('尚未完成一键原生测试。', ['bash run_native.sh test'])
        else:
            tested = self.host_path(read_json(latest)['output'])
            record = read_json(tested / 'run.json')
            if record.get('mode') != 'test' or record.get('status') != 'passed' or not record.get('acceptance'):
                raise Stopped('最近一次测试未通过；先查看 ' + str(tested / 'SUMMARY.txt'), ['bash run_native.sh test'])
            evidence = self.host_path(record['acceptance'])
        if not evidence.is_file():
            raise Stopped('测试原始验收记录缺失。', ['bash run_native.sh test'])
        return evidence

    def verify(self):
        evidence = self.test_evidence()
        result_dir = self.out / 'r234_native'
        self.record.update(acceptance=self.relative(evidence), experiment_output=self.relative(result_dir))
        self.step('benchmark', '复用原始证据并运行前 ' + str(self.max_conditions) + ' 条件，核对 Ours 完整评分',
                  ['--execution-scope', 'native', '--native-evidence', self.container_path(evidence),
                   '--max-conditions', str(self.max_conditions), '--out', self.container_path(result_dir)], native=False)
        from dabs_ascend.planning_audit import audit
        coverage = audit(result_dir, self.max_conditions)
        write_json(self.out / 'planning_audit.json', coverage)
        emit(self.out, 'PLANNING_AUDIT', **coverage)
        if not coverage['pass']:
            raise Stopped('规划验证未通过：' + '; '.join(coverage['missing']) + '。复制 copy_logs/COPY.txt；原结果保留。')
        self.record['next_commands'] = ['bash run_native.sh experiment']

    def experiment(self):
        verified = self.root / 'outputs/native-entry-last-verify.json'
        if not self.input_path and verified.exists():
            prior = self.host_path(read_json(verified)['output'])
            record = read_json(prior / 'run.json')
            if record.get('mode') != 'verify' or record.get('status') != 'passed':
                raise Stopped('最近一次规划验证未通过，不能退回旧测试 PASS。复制 ' + str(prior / 'copy_logs/COPY.txt'))
            result_dir = self.host_path(record['experiment_output'])
            evidence = self.host_path(record['acceptance'])
            from dabs_ascend.planning_audit import audit
            count = len(read_json(result_dir / 'index.json'))
            coverage = audit(result_dir, count)
            if not coverage['pass']:
                raise Stopped('续跑前重新核对完整评分失败：' + str(coverage['missing']))
            self.record.update(acceptance=self.relative(evidence), experiment_output=self.relative(result_dir),
                               resumed_from=self.relative(prior), copy_log_owner=str(prior))
            self.step('benchmark', '复核原始证据，从已完成条件继续到 96 条',
                      ['--execution-scope', 'native', '--native-evidence', self.container_path(evidence),
                       '--resume', self.container_path(result_dir)], native=False)
        else:
            evidence = self.test_evidence()
            result_dir = self.out / 'r234_native'
            self.record['acceptance'] = self.relative(evidence)
            self.record['experiment_output'] = self.relative(result_dir)
            self.step('benchmark', '重新核验原生证据，然后运行全部 96 条件',
                      ['--execution-scope', 'native', '--native-evidence', self.container_path(evidence),
                       '--out', self.container_path(result_dir)], native=False)
        # This pointer is convenience only: R234 recomputes raw evidence and
        # verifies live engine identities and resources before canaries or formal traffic.
        result = read_json(result_dir / 'status.json')
        if result.get('status') != 'complete' or result.get('completed_conditions') != 96 or result.get('functional_test_pass') is not True:
            raise Stopped('实验没有完整通过全部 96 条件；保留实际结果，不标记完成。')

    def run(self):
        # A new failed/interrupted test cannot leave an older PASS as the default.
        write_json(self.root / ('outputs/native-entry-last-' + self.mode + '.json'),
                   {'output': self.relative(self.out)})
        self.save()
        emit(self.out, 'START', mode=self.mode, output=self.relative(self.out))
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
        emit(self.out, 'END', status=self.record['status'], reason=self.record.get('reason'),
             experiment=self.record.get('experiment_output'), next=self.record['next_commands'])
        print('\n' + (self.out / 'SUMMARY.txt').read_text(encoding='utf-8'), flush=True)
        return code


def main():
    parser = argparse.ArgumentParser(description='现场入口：install 安装；test 完整验收；verify 前8条规划验证；experiment 全部96条/续跑；logs 打印复制摘要。')
    parser.add_argument('mode', choices=('install', 'smoke', 'calibrate', 'audit', 'refit', 'explore', 'test', 'verify', 'experiment', 'logs'))
    parser.add_argument('--replica', type=int, choices=range(4), help='install/smoke/calibrate: select one existing service')
    parser.add_argument('--input', type=Path, help='audit/refit/explore: calibration directory; verify/experiment: prior native_test directory (read-only)')
    parser.add_argument('--invalidate-cache', action='store_true', help='仅 test：明确执行副本 0 缓存清空/失效测试；默认不执行')
    parser.add_argument('--verify-restart', action='store_true', help='仅 install：明确额外重启副本 0 并验证旧身份失效')
    parser.add_argument('--ready-timeout', type=int, default=900, help='install 每个服务加载等待秒数，默认 900')
    parser.add_argument('--max-conditions', type=int, help='explore: 1..96 default 4; verify: 4..92, multiple of 4, default 8')
    args = parser.parse_args()
    if args.mode == 'logs':
        pointer = ROOT / 'outputs/native-entry-last-copy.json'
        if not pointer.exists():
            print('没有新版本生成的复制摘要；旧结果保持原样。', file=sys.stderr)
            return 2
        path = (ROOT / read_json(pointer)['file']).resolve()
        if not path.is_relative_to(ROOT / 'outputs'):
            parser.error('复制摘要路径不在 outputs 下')
        print(path.read_text(encoding='utf-8'), end='')
        return 0
    if args.max_conditions is not None:
        if args.mode == 'verify':
            if not 4 <= args.max_conditions <= 92 or args.max_conditions % 4:
                parser.error('verify --max-conditions 为 4..92 的 4 的倍数（完整方法组）')
        elif args.mode != 'explore' or not 1 <= args.max_conditions <= 96:
            parser.error('--max-conditions 用于 explore 或 verify')
    if args.replica is not None and args.mode not in ('install', 'smoke', 'calibrate'):
        parser.error('--replica 仅用于 install/smoke/calibrate')
    if (args.mode in ('audit', 'refit') and not args.input) or (args.input and args.mode not in ('audit', 'refit', 'explore', 'verify', 'experiment')):
        parser.error('audit/refit 必须提供 --input；explore 可指定已有标定目录；experiment 可指定旧 native_test 结果目录')
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
                            args.replica, args.input, args.max_conditions).run()
    except BlockingIOError:
        print('已有测试或实验使用这些服务；请等待当前任务结束。', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
