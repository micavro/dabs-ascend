"""Safe CLI entry. A spawned engine imports this file without starting a second server."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    from multiprocessing import freeze_support
    freeze_support()
    from joint18v2.bootstrap import run_server
    run_server()

if __name__ == '__main__':
    main()
