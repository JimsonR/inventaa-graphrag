import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def track_time(task_name: str, custom_logger: logging.Logger = None):
    """
    Lightweight context manager to track the execution time of a block of code.
    Usage:
        with track_time("My Task"):
            do_something_heavy()
    """
    start_time = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start_time
    target_logger = custom_logger or logger
    target_logger.info(f"[Timer] {task_name} took {elapsed:.3f} seconds")
