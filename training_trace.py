"""Opt-in crash breadcrumbs; no third-party dependencies beyond the trainer."""
import faulthandler
import json
import os
import sys
import time
from pathlib import Path


def enable_fault_handler():
    # Called before torch is imported, also in spawned data-loader workers.
    if os.environ.get('TRAIN_TRACE', '0') == '1':
        faulthandler.enable(file=sys.stderr, all_threads=True)


def process_resources(pid):
    result = {'pid': pid}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSize', 'Threads'):
                result[key] = value.strip()
        result['fds'] = len(list(Path(f'/proc/{pid}/fd').iterdir()))
    except OSError:
        pass  # /proc is Linux-only; workers can exit during collection.
    return result


class TrainingTrace:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.enabled = os.environ.get('TRAIN_TRACE', '0') == '1'
        self.sync = os.environ.get('TRACE_CUDA_SYNC', '0') == '1'
        self.every = max(1, int(os.environ.get('TRACE_EVERY', '100')))

    def mark(self, phase, epoch=None, batch=None, resources=False):
        if not self.enabled:
            return
        entry = dict(time=time.time(), pid=os.getpid(), phase=phase,
                     epoch=epoch, batch=batch)
        # Write before synchronization so an async CUDA failure has a breadcrumb.
        print('[train-trace] ' + json.dumps(entry), file=sys.stderr, flush=True)
        if self.sync and self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        if resources or (phase == 'batch_end' and batch % self.every == 0):
            stats = {'main': process_resources(os.getpid())}
            try:
                children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
                stats['children'] = [process_resources(int(pid)) for pid in children]
                shm = os.statvfs('/dev/shm')
                stats['shm_free_bytes'] = shm.f_bavail * shm.f_frsize
            except OSError:
                pass
            if self.device.type == 'cuda':
                stats['cuda_allocated'] = self.torch.cuda.memory_allocated(self.device)
                stats['cuda_reserved'] = self.torch.cuda.memory_reserved(self.device)
                stats['cuda_peak_allocated'] = self.torch.cuda.max_memory_allocated(self.device)
            print('[train-resources] ' + json.dumps(stats), file=sys.stderr, flush=True)
