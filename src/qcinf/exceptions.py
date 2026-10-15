"""Exceptions raised by qcinf."""


class NumbaUnavailableError(RuntimeError):
    """Exception raised when numba is unavailable but required for a fast algorithm."""

    pass
