import time
from datetime import timedelta


class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        if self.name:
            print(f"[{self.name}] Elapsed: {format_duration(self.elapsed)}")

    @property
    def elapsed_seconds(self):
        if self.end_time is not None:
            return self.elapsed
        if self.start_time is not None:
            return time.perf_counter() - self.start_time
        return 0.0


def format_duration(seconds):
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimate_remaining(current, total, elapsed):
    if current == 0:
        return float("inf")
    rate = elapsed / current
    remaining = (total - current) * rate
    return remaining
