from importlib import metadata as _metadata

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    # Source tree / build hook / CI checkout
    __version__ = "0.0.0+local"

from .algorithms.connectivity import *  # noqa: F401, F403
from .algorithms.conversion import *  # noqa: F401, F403
from .algorithms.geometry import *  # noqa: F401, F403
from .algorithms.snap import *  # noqa: F401, F403
