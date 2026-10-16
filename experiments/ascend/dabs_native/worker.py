"""NPU forward-stream events; asynchronous return is never a completion time."""
import os
import time
import uuid
from .contracts import DeviceMeasurements, require_members


def install_worker_class(cls):
    require_members(cls, 'NPUWorker', methods=('execute_model',))
    original = cls.execute_model
    def execute(self, scheduler_output):
        if not hasattr(self, '_dabs_measurements'):
            require_members(self, 'NPUWorker', fields=('model_runner',))
            require_members(self.model_runner, 'model_runner', methods=('_model_forward',))
            import torch
            require_members(torch, 'torch', fields=('npu',))
            require_members(torch.npu, 'torch.npu', methods=('Event', 'current_stream', 'is_available'))
            self._dabs_measurements = DeviceMeasurements()
            forward = self.model_runner._model_forward
            def measured(*args, **kwargs):
                import torch
                batch = self._dabs_current_batch
                if not batch.dabs_batch_id:
                    raise RuntimeError('missing explicit async batch ID')
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                stream = torch.npu.current_stream()
                host = time.monotonic_ns()
                start.record(stream)
                output = forward(*args, **kwargs)
                end.record(stream)
                host_ms = (time.monotonic_ns() - host) / 1e6
                self._dabs_measurements.add(batch.dabs_batch_id, start, end, host_ms, batch.dabs_batch_meta)
                # query() only; no synchronize() in the measured path.
                self._dabs_measurements.ready()
                if len(self._dabs_measurements.rows) > 200000:
                    raise RuntimeError('NPU timing record capacity exhausted')
                return output
            self.model_runner._model_forward = measured
        self._dabs_current_batch = scheduler_output
        return original(self, scheduler_output)
    def info(self):
        import torch
        require_members(self, 'NPUWorker', fields=('model_runner',))
        require_members(self.model_runner, 'model_runner', methods=('_model_forward',))
        require_members(torch, 'torch', fields=('npu',))
        require_members(torch.npu, 'torch.npu', methods=('Event', 'current_stream', 'is_available'))
        if not hasattr(self, '_dabs_worker_id'):
            self._dabs_worker_id = uuid.uuid4().hex
        return {'pid': os.getpid(), 'worker_instance': self._dabs_worker_id,
                'class': type(self).__module__ + '.' + type(self).__name__,
                'npu_available': bool(torch.npu.is_available()),
                'timing': 'two torch.npu.Event records per nonempty model forward; nonblocking query; no forced synchronization'}
    def read(self, after_step=0):
        if not hasattr(self, '_dabs_measurements'):
            return []
        return [r for r in self._dabs_measurements.ready() if r['batch']['step'] > after_step][:128]
    def ack(self, step):
        if hasattr(self, '_dabs_measurements'):
            self._dabs_measurements.rows[:] = [r for r in self._dabs_measurements.rows if r['batch']['step'] > step]
        return True
    cls.execute_model, cls.dabs_worker_info, cls.dabs_timing_read = execute, info, read
    cls.dabs_timing_ack = ack
