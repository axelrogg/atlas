from importlib.metadata import version

from atlas._atlas import Simulator
from atlas.core.model.elements import ElementKind

# The version lives in pyproject.toml only; commitizen bumps it there.
__version__ = version("atlas")
