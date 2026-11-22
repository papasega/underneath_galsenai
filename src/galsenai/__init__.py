"""galsenai: a small, modern decoder-only transformer built for on-device use."""

from galsenai.config import ModelConfig
from galsenai.generate import generate
from galsenai.model import GPT

__all__ = ["GPT", "ModelConfig", "generate"]
__version__ = "0.1.0"
