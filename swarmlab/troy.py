"""The single import point for Troy Moment (vendor/troy-moment, MIT).

Troy is used as a library only; nothing under vendor/ is edited.
"""
import sys
from pathlib import Path

TROY_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "troy-moment"
for _p in (TROY_ROOT / "src", TROY_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pp.models.api_backends import OpenAIBackend  # noqa: E402
from pp.models.base import (ActionParseError, BaseModelBackend,  # noqa: E402
                            GenerationResult, parse_action)
from pp.schemas import Action, ActionType, Usage  # noqa: E402

__all__ = ["TROY_ROOT", "Action", "ActionParseError", "ActionType",
           "BaseModelBackend", "GenerationResult", "OpenAIBackend", "Usage",
           "parse_action"]
