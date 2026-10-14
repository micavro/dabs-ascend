"""Explicit process-local hook installation without changing vLLM's process topology.

The CLI uses the native v0.6.3 parser/run_server, not runpy as __main__. The native
multiprocessing entry is wrapped by an importable target; spawn does not inherit
parent-process monkey patches. This module is inert on import.
"""
from __future__ import annotations
import importlib, inspect, json, os, sys, time, traceback, uuid
from pathlib import Path
from .common import atomic
REVISION='stage18v2-boot-metrics-fix-1'

def marker(phase, **details):
    row={'revision':REVISION,'phase':phase,'pid':os.getpid(),
         'pod':os.environ.get('POD_NAME','unknown'),
         'launch_id':os.environ.get('SPRA_J18V2_LAUNCH_ID'),
         'mono_ns':time.monotonic_ns(),**details}
    print('JOINT18V2_STARTUP='+json.dumps(row,sort_keys=True),file=sys.stderr,flush=True)
    try:
        path=Path(os.environ.get('SPRA_E1_ROOT','/spra-e1'))/'joint18v2'
        # Diagnostic only: never fabricate the Scheduler-owned .status.json.
        atomic(path/(str(row['pod'])+'.boot.'+str(row['pid'])+'.json'),row)
    except OSError as exc:
        print('JOINT18V2_BOOT_RECORD_ERROR='+repr(exc),file=sys.stderr,flush=True)

def mp_engine_with_hook(*args, **kwargs):
    """Pickleable spawn target: install the hook in the actual engine process."""
    try:
        marker('engine_child_enter')
        hook=importlib.import_module('joint18v2_hook')
        hook.install()
        marker('engine_child_hook_installed')
        native=importlib.import_module('vllm.engine.multiprocessing.engine').run_mp_engine
        return native(*args,**kwargs)
    except BaseException as exc:
        marker('engine_child_failed',error=repr(exc),traceback=traceback.format_exc())
        raise

def precheck_bootstrap():
    """Inspect only. No engine, scheduler, model or child process is started."""
    api=importlib.import_module('vllm.entrypoints.openai.api_server')
    native=importlib.import_module('vllm.engine.multiprocessing.engine').run_mp_engine
    for name in ('make_arg_parser','validate_parsed_serve_args','run_server'):
        if not callable(getattr(api,name,None)):
            raise RuntimeError('native API startup interface missing: '+name)
    if getattr(api,'run_mp_engine',None) is not native:
        raise RuntimeError('API server multiprocessing target already replaced')
    return {'revision':REVISION,'native_mp_target_module':native.__module__,
            'native_mp_target_signature':str(inspect.signature(native)),
            'guarded_launcher':True,'child_hook_install_explicit':True,
            'frontend_mode_not_forced':True,'engine_started':False,'model_requests':0}

def run_server():
    """Match the native v0.6.3 CLI while guarding import and spawn boundaries."""
    os.environ['SPRA_J18V2_LAUNCH_ID']=uuid.uuid4().hex
    try:
        marker('launcher_enter')
        api=importlib.import_module('vllm.entrypoints.openai.api_server')
        precheck_bootstrap()
        parser=api.FlexibleArgumentParser(description='vLLM OpenAI API with Stage18-v2 experiment hook')
        parser=api.make_arg_parser(parser)
        args=parser.parse_args()
        api.validate_parsed_serve_args(args)
        hook=importlib.import_module('joint18v2_hook')
        hook.install()
        # The direct-engine path uses the hook installed above. The spawned path
        # uses this named module target, which installs its own process-local hook.
        api.run_mp_engine=mp_engine_with_hook
        marker('api_server_enter',disable_frontend_multiprocessing=bool(getattr(args,'disable_frontend_multiprocessing',False)),
               hook_file=str(Path(hook.__file__).resolve()))
        api.uvloop.run(api.run_server(args))
        marker('api_server_returned')
    except BaseException as exc:
        marker('launcher_failed',error=repr(exc),traceback=traceback.format_exc())
        raise
