# errors.py
class TransientError(Exception):
    """
    Explicit marker for retryable operational failures (I/O, network, API flaps).

    This exception MUST be raised intentionally at retryable boundaries.
    """
    pass
