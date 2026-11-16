"""Exception type shared by encoder and decoder."""


class BrotliError(Exception):
    """Raised on invalid compressed data or invalid parameters (``brotli_pure.error``)."""
