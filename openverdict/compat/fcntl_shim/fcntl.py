"""Windows stand-in for the POSIX fcntl module, used only on the PYTHONPATH of
the JevBench harness subprocess (openverdict/eval/jevbench_run.py).

JevBench's budget ledger calls fcntl.flock to serialize concurrent writers.
openverdict runs one harness process at a time, so the lock has nothing to
guard and flock is a no-op here.
"""
LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(fd, operation):
    return None
