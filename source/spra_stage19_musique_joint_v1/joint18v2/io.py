"""Bounded asynchronous JSON persistence. Full records are never silently dropped."""
from __future__ import annotations
import json,os,queue,threading,time,uuid
from pathlib import Path
from .common import atomic

class JsonWriter:
    def __init__(self, maxsize: int = 50000):
        self.q=queue.Queue(maxsize=maxsize);self.error=None;self.closed=False;self.high_water=0
        self.thread=threading.Thread(target=self._run,daemon=True,name='joint18v2-json-writer');self.thread.start()
    def _run(self):
        handles={}
        try:
            while True:
                item=self.q.get()
                try:
                    if item is None:break
                    mode,path,obj=item;path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
                    if mode=='replace':atomic(path,obj)
                    else:
                        if path not in handles:handles[path]=path.open('a',encoding='utf-8',buffering=1)
                        handles[path].write(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                finally:self.q.task_done()
        except BaseException as ex:self.error=repr(ex)
        finally:
            for h in handles.values():h.close()
    def check(self):
        if self.error:raise RuntimeError('asynchronous writer failed: '+self.error)
    def submit(self,path,obj,replace=False):
        self.check()
        if self.closed:raise RuntimeError('writer closed')
        # Objects submitted by the caller must be immutable snapshots, not live nodes.
        try:self.q.put_nowait(('replace' if replace else 'append',str(path),obj))
        except queue.Full:raise RuntimeError('audit queue overflow; condition cannot be valid')
        self.high_water=max(self.high_water,self.q.qsize())
    def drain(self,timeout=10.):
        end=time.monotonic()+timeout
        while self.q.unfinished_tasks:
            self.check()
            if time.monotonic()>end:raise RuntimeError('audit queue did not drain')
            time.sleep(.002)
        self.check()
    def close(self):
        if self.closed:return
        self.drain();self.closed=True;self.q.put(None);self.thread.join(5);self.check()
