"""Build LLM Systems from permitted views, checked replies and optional history.

make_system returns an ordinary host System. Client owns bounded HTTP work;
explicitly supplied clients remain caller-owned. Formatting and native legality
helpers also work alone. Nothing starts a server, loads weights or changes Core.
"""

from marl_battlegrounds.llm.actions import (
    IllegalActionError,
    InvalidActionError,
    MaskedActionError,
    ReplyFormatError,
    legal_action_names,
    parse_action_reply,
    validate_actor_action,
)
from marl_battlegrounds.llm.client import (
    Client,
    ContextLimitError,
    ProviderConfigurationError,
    RequestCancelledError,
    TransportError,
)
from marl_battlegrounds.llm.recording import call_summary, read_calls
from marl_battlegrounds.llm.system import HistoryEntry, make_system
from marl_battlegrounds.llm.text import TEXT_VERSION, format_actor_view

__all__ = [
    "TEXT_VERSION",
    "Client",
    "ContextLimitError",
    "HistoryEntry",
    "IllegalActionError",
    "InvalidActionError",
    "MaskedActionError",
    "ProviderConfigurationError",
    "ReplyFormatError",
    "RequestCancelledError",
    "TransportError",
    "call_summary",
    "format_actor_view",
    "legal_action_names",
    "make_system",
    "parse_action_reply",
    "read_calls",
    "validate_actor_action",
]
