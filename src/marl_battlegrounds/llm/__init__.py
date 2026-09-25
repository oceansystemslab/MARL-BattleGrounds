"""Format permitted game views and validate named LLM actions on the host.

These small helpers do not start a server, load weights or change Core. Default
JSON decoding and native legality checks are separate so custom formats can
share the same action boundary.
"""

from marl_battlegrounds.llm.actions import (
    IllegalActionError,
    ReplyFormatError,
    legal_action_names,
    parse_action_reply,
    validate_actor_action,
)
from marl_battlegrounds.llm.text import TEXT_VERSION, format_actor_view

__all__ = [
    "TEXT_VERSION",
    "IllegalActionError",
    "ReplyFormatError",
    "format_actor_view",
    "legal_action_names",
    "parse_action_reply",
    "validate_actor_action",
]
