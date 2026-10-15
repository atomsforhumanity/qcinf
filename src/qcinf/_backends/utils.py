"""Shared utilities for the cheminformatics backends."""

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

# one global lock per process
_STDERR_LOCK = threading.Lock()


@contextmanager
def mute_c_stderr() -> Iterator[None]:
    """Redirect the C-level `stderr` (fd 2) to /dev/null within the context.

    This silences C / C++ libraries like RDKit that write directly with
    `fprintf(stderr, …)` or `std::cerr`.

    Acquires a global lock (to make it thread-safe), redirects fd 2 to /dev/null,
    runs the body, then restores stderr and releases the lock.

    Be aware that if another thread NOT using this context manager writes to
    stderr while this lock is held, it will be lost (written to /dev/null).
    In practice this is rare, just be aware of it.

    Yields:
        None; C-level stderr output is discarded while the context is active.
    """
    with _STDERR_LOCK:
        # Duplicate the original fd so we can restore later
        orig_fd = os.dup(2)

        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 2)  #  ← fd 2 now points to /dev/null
            yield
        finally:
            os.dup2(orig_fd, 2)  #  Restore real stderr
            os.close(orig_fd)
