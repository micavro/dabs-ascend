"""Bounded raw RPC buffering during a decision; durable flush before trace ACK.

No evidence is discarded. The event loop owns this buffer. Disk/serialization
cost is paid outside the snapshot-to-commit interval, not by each replica reply.
"""
import json
import os
from pathlib import Path


class EvidenceBuffer:
    def __init__(self, limit=128):
        self.rows = []
        self.limit = limit

    def append(self, path, row):
        if len(self.rows) >= self.limit:
            raise RuntimeError('native evidence buffer full; stop without dropping records')
        self.rows.append((Path(path), row))

    def flush(self, durable=False):
        # Keep rows on any failure. The run fails rather than acknowledging loss.
        paths = {}
        for path, row in self.rows:
            paths.setdefault(path, []).append(row)
        for path, rows in paths.items():
            with path.open('a', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()
                if durable:
                    os.fsync(stream.fileno())
        self.rows.clear()
