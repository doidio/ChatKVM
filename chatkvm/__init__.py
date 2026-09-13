from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("chatkvm")
except _metadata.PackageNotFoundError:
    __version__ = "0.1.0"
