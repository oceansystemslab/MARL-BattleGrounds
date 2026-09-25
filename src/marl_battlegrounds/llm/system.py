"""Build an ordinary host System from bounded model calls and actor-local history.

Only permitted actor inputs reach prompts, parsers and token counters. History
is immutable prospective memory: callers adopt it only with a successful game
step. Shared runners own game identifiers, durable records and managed lifetime.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ALIVE,
    CONTEXT_FEATURE_CURRENT_TIMESTEP,
    CONTEXT_FEATURE_MAP_WIDTH,
    ActionMask,
)
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.evaluation.recording_identity import (
    _callable_evidence,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.llm.actions import (
    COMBAT_NAMES,
    MOVE_NAMES,
    MaskedActionError,
    ReplyFormatError,
    legal_action_names,
    parse_action_reply,
    validate_actor_action,
)
from marl_battlegrounds.llm.client import Client, ContextLimitError, TransportError
from marl_battlegrounds.llm.recording import (
    CallEvidence,
    RecordingMode,
    actor_channel,
    attach,
)
from marl_battlegrounds.llm.text import TEXT_VERSION, format_actor_view
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import (
    ActorInput,
    mirror_move,
    mirror_team_view,
    obstacle_mirror_partners,
    team_on_right,
)


@dataclass(frozen=True, eq=False)
class HistoryEntry:
    """One previous living-actor turn in the System's chosen coordinate frame.

    actor and masks are this actor's permitted pre-step inputs. action is the
    submitted choice, not proof of a hit or successful ability. All array bytes
    are read-only and independent of caller-owned buffers. timestep comes from
    the public context. Entries are retained only by adopted System memory;
    death preserves them and a new game's normal memory reset clears them.
    """

    actor: ActorInput
    masks: ActionMask
    action: ActorAction
    timestep: int


type History = tuple[HistoryEntry, ...]
type PromptBuilder = Callable[[ActorInput, ActionMask, History], str]
type ReplyParser = Callable[[str, ActorInput, ActionMask, History], ActorAction]
type TokenCounter = Callable[[Mapping[str, object]], tuple[int, int]]
type _Memory = list[tuple[History, ...]]


def _snapshot[T](value: T) -> T:
    """Make a numerical tree immutable without exposing a writable base buffer."""

    def freeze(leaf: Array) -> Array:
        """Copy one leaf into bytes-backed host storage, preserving dtype/shape."""
        array = np.asarray(leaf)
        return cast(
            Array,
            np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape),
        )

    return jax.tree.map(freeze, value)


def _slice[T](value: T, lane: int, actor: int) -> T:
    """Take exactly one actor's permitted numerical rows without other actors."""

    def take(leaf: Array) -> Array:
        """Take one lane and actor while retaining all feature axes."""
        return leaf[lane, actor]

    return jax.tree.map(take, value)


def _history_text(entry: HistoryEntry, frame: Literal["left", "world"]) -> str:
    """Render one earlier view once, even if fitting later removes older turns."""
    action = entry.action
    return "\n".join(
        [
            f"Earlier turn {entry.timestep}:",
            format_actor_view(
                entry.actor,
                entry.masks,
                frame=frame,
                include_static=False,
                reply_instruction=None,
            ),
            f"Submitted: move={MOVE_NAMES[int(action.move)]} "
            f"combat={COMBAT_NAMES[int(action.select_target)][int(action.use_ultimate)]}",
        ]
    )


def _prompt(current: str, earlier: tuple[str, ...]) -> str:
    """Join already rendered views without changing current facts or their order."""
    if not earlier:
        return current
    return "\n".join(
        [
            current,
            "Earlier turns follow. Their menus are historical; "
            "use only the current menu above.",
            *earlier,
            "End of history. Choose for the current view and its current legal menu.",
        ]
    )


@jax.jit
def _left_view(
    actors: ActorInput, masks: ActionMask
) -> tuple[ActorInput, ActionMask, Array]:
    """Reflect a whole permitted batch with one obstacle comparison per game."""
    flag = team_on_right(actors)
    tables = actors.observation.map_obstacle_features
    widths = actors.observation.context_features[..., CONTEXT_FEATURE_MAP_WIDTH]
    partners = obstacle_mirror_partners(tables[:, 0], widths[:, 0])
    partners = jnp.broadcast_to(partners[:, None], tables.shape[:-1])
    view, mask = mirror_team_view(actors, masks, flag, obstacle_partners=partners)
    return view, mask, flag


_world_moves = jax.jit(mirror_move)


def _host_device() -> object:
    """Prefer CPU for existing reflection helpers after the host input boundary."""
    try:
        return cast(object, jax.devices("cpu")[0])
    except RuntimeError:
        return cast(object, jax.devices()[0])


class _Resources:
    """Own one lazily used Client across nested runner scopes, never a server."""

    def __init__(
        self, client: Client, *, owned: bool, identity_known: bool, identity_json: str
    ) -> None:
        """Retain configured transport and identity without opening any resource."""
        self._client = client
        self._owned = owned
        self._identity_known = identity_known
        self._identity_json = identity_json
        self._leases = 0
        self._closed = False
        self._failed = False
        self._closing = False
        self._lock = threading.RLock()

    @property
    def client(self) -> Client:
        """Return this run's client, refusing unmanaged or failed owned use."""
        with self._lock:
            if self._owned and not self._leases:
                raise RuntimeError(
                    "Use a runner or System.resource_scope(False), "
                    "or supply a caller-owned LLM Client"
                )
            if self._failed:
                raise RuntimeError("LLM cleanup failed; the old client remains fenced")
            return self._client

    def scope(self, recording: bool) -> AbstractContextManager[None]:
        """Check saved custom identity now; acquire only on context entry."""
        if recording and not self._identity_known:
            raise ValueError(
                "Saved custom LLM methods need custom_version and identifiable "
                "Python hooks; declare answer-relevant custom_settings too"
            )
        return self._lease()

    @contextmanager
    def _lease(self) -> Generator[None]:
        """Share nested use and close an owned client only after its last user."""
        with self._lock:
            if self._failed or self._closing:
                raise RuntimeError(
                    "LLM cleanup is unfinished; the old client remains fenced"
                )
            if self._owned and self._closed:
                self._client = Client(
                    self._client.server_url,
                    concurrency=self._client.concurrency,
                    timeout=self._client.timeout,
                    max_response_bytes=self._client.max_response_bytes,
                )
                self._closed = False
            self._leases += 1
        body_error: BaseException | None = None
        try:
            attach(self._identity_json)
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            with self._lock:
                self._leases -= 1
                close = self._owned and not self._leases
                if close:
                    self._closing = True
            if close:
                try:
                    self._client.close()
                except BaseException as error:
                    with self._lock:
                        self._failed = True
                    if body_error is None:
                        raise
                    body_error.add_note(
                        f"LLM cleanup also failed: {type(error).__name__}: {error}"
                    )
                finally:
                    with self._lock:
                        self._closed = not self._failed
                        self._closing = False


@dataclass(frozen=True, eq=False)
class _Settings:
    """Keep fixed LLM method configuration outside actor memory and JAX trees."""

    model: str
    resources: _Resources
    borrowed: bool
    history_turns: int
    frame: Literal["left", "world"]
    failure_policy: Literal["stop", "fallback"]
    prompt_builder: PromptBuilder | None
    reply_parser: ReplyParser | None
    token_counter: TokenCounter | Literal["vllm"] | None
    context_limit: int | None
    generation_json: str
    identity_json: str
    custom_identity_declared: bool
    records: RecordingMode
    method_id: str

    @property
    def client(self) -> Client:
        """Resolve a scoped managed transport or the supplied caller-owned client."""
        return self.resources.client

    def generation(self) -> dict[str, Any]:
        """Return fresh request fields, keeping caller dictionaries detached."""
        return json.loads(self.generation_json)


def _init(settings: _Settings, inputs: SystemInput, keys: Array) -> _Memory:
    """Create empty lane-local histories without opening a connection or acting."""
    del settings, keys
    return [tuple(() for _ in range(5)) for _ in range(len(inputs.valid))]


def _count(
    settings: _Settings, request: dict[str, Any], evidence: CallEvidence | None = None
) -> tuple[int, int] | None:
    """Count the exact local chat wrapper or call a declared trusted counter."""
    counter = settings.token_counter
    if counter is None:
        return None
    if counter == "vllm":
        if evidence is not None:
            evidence.count("tokenizer_calls")
        result = settings.client.request(
            "/tokenize",
            {
                "model": settings.model,
                "messages": request["messages"],
                "add_generation_prompt": True,
                "continue_final_message": False,
                "add_special_tokens": False,
                "chat_template_kwargs": request.get("chat_template_kwargs", {}),
            },
        )
        count, limit = result.get("count"), result.get("max_model_len")
    else:
        try:
            count, limit = counter(request)
        except (TransportError, ReplyFormatError, MaskedActionError) as exc:
            raise RuntimeError("Custom token counter failed") from exc
    if type(count) is not int or count < 0 or type(limit) is not int or limit < 1:
        raise ValueError(
            "Token counter must return a nonnegative count and positive context limit"
        )
    if settings.context_limit is not None:
        limit = min(limit, settings.context_limit)
    return count, limit


def _request(
    settings: _Settings,
    actor: ActorInput,
    masks: ActionMask,
    history: History,
    evidence: CallEvidence | None = None,
) -> ActorAction:
    """Fit whole history entries, generate once and validate the exact reply."""
    included = history
    current = ""
    earlier: tuple[str, ...] = ()
    if settings.prompt_builder is None:
        current = format_actor_view(actor, masks, frame=settings.frame)
        earlier = tuple(_history_text(entry, settings.frame) for entry in history)
    while True:
        if settings.prompt_builder is None:
            prompt = _prompt(current, earlier[len(history) - len(included) :])
        else:
            try:
                prompt = settings.prompt_builder(actor, masks, included)
            except (TransportError, ReplyFormatError, MaskedActionError) as exc:
                raise RuntimeError("Custom prompt builder failed") from exc
        if not isinstance(cast(object, prompt), str):
            raise TypeError("Prompt builder must return text")
        request = settings.generation()
        request.update(
            model=settings.model, messages=[{"role": "user", "content": prompt}]
        )
        counted = _count(settings, request, evidence)
        if counted is None or counted[0] + request["max_tokens"] <= counted[1]:
            break
        if not included:
            raise ContextLimitError(
                "Current view and reserved reply exceed the model context; "
                "nothing was shortened"
            )
        included = included[1:]
    if evidence is not None:
        evidence.update(history_entries=len(included))
        evidence.request(request)
    response = settings.client.request("chat/completions", request)
    if evidence is not None:
        evidence.response(response)
    usage = response.get("usage", {})
    if counted is not None and (
        not isinstance(usage, dict) or usage.get("prompt_tokens") != counted[0]
    ):
        raise ValueError(
            "Generation token count differs from the exact request counter"
        )
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(cast(list[object], choices)) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ReplyFormatError("Model must return exactly one choice")
    choice = cast(dict[str, Any], choices[0])
    if choice.get("finish_reason") != "stop":
        raise ReplyFormatError("Model did not finish one complete reply")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ReplyFormatError("Model must return a text reply")
    reply = cast(str, message["content"])
    if settings.reply_parser is None:
        action = parse_action_reply(reply, masks)
    else:
        try:
            action = settings.reply_parser(reply, actor, masks, included)
        except TransportError as exc:
            raise RuntimeError("Custom reply parser failed") from exc
    return validate_actor_action(action, masks)


def _decide(
    settings: _Settings,
    actor: ActorInput,
    masks: ActionMask,
    history: History,
    evidence: CallEvidence | None = None,
) -> tuple[ActorAction, History]:
    """Choose one actor action; immutable history is prospective until adopted."""
    menu = legal_action_names(masks)
    source = "model"
    if len(menu["move"]) == len(menu["combat"]) == 1:
        source = "forced"
        action = parse_action_reply(
            json.dumps({name: choices[0] for name, choices in menu.items()}), masks
        )
    else:
        try:
            action = _request(settings, actor, masks, history, evidence)
        except (ReplyFormatError, MaskedActionError, TransportError) as error:
            if evidence is not None:
                evidence.failure(
                    "transport_failures"
                    if isinstance(error, TransportError)
                    else "reply_failures",
                    error,
                )
            if settings.failure_policy == "stop":
                raise
            source = "fallback"
            action = validate_actor_action(
                ActorAction(*(cast(Array, np.int32(0)) for _ in range(3))), masks
            )
    if evidence is not None:
        evidence.action(action, source, frame=settings.frame)
    if not settings.history_turns:
        return action, history
    # Custom hooks already received independent, bytes-backed immutable inputs.
    immutable_inputs = (
        settings.prompt_builder is not None or settings.reply_parser is not None
    )
    entry = HistoryEntry(
        actor if immutable_inputs else _snapshot(actor),
        masks if immutable_inputs else _snapshot(masks),
        _snapshot(action),
        int(actor.observation.context_features[CONTEXT_FEATURE_CURRENT_TIMESTEP]),
    )
    return action, (*history, entry)[-settings.history_turns :]


def _apply(
    settings: _Settings, memory: _Memory, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Return a checked full-team action and new memory without mutating old state."""
    del keys
    if len(memory) != len(inputs.valid) or any(len(lane) != 5 for lane in memory):
        raise ValueError("LLM memory must contain five actor histories per game")
    device = _host_device()
    with jax.default_device(device):
        if settings.frame == "left":
            actors, masks, flag = jax.device_get(
                _left_view(inputs.actors, inputs.action_mask)
            )
        else:
            actors, masks = inputs.actors, inputs.action_mask
            flag = np.zeros(inputs.active_mask.shape, dtype=np.bool_)
    active = np.asarray(inputs.valid)[:, None] & np.asarray(inputs.active_mask)
    living = np.asarray(actors.observation.self_features)[..., AGENT_FEATURE_ALIVE] > 0
    selected = np.argwhere(active & living)
    evidence = {
        (int(lane), int(slot)): actor_channel(
            int(lane), int(slot), settings.records, settings.method_id
        )
        for lane, slot in selected
    }
    actions = [np.zeros(inputs.active_mask.shape, np.int32) for _ in range(3)]
    histories = list(memory)
    changed: dict[int, list[History]] = {}

    def decide(
        index: np.ndarray[Any, np.dtype[np.int64]],
    ) -> tuple[int, int, ActorAction, History]:
        """Expose only one actor's row to ordinary custom functions and transport."""
        lane, slot = (int(value) for value in index)
        actor, mask = _slice(actors, lane, slot), _slice(masks, lane, slot)
        if settings.prompt_builder is not None or settings.reply_parser is not None:
            actor, mask = _snapshot(actor), _snapshot(mask)
        action, history = _decide(
            settings, actor, mask, memory[lane][slot], evidence[lane, slot]
        )
        return lane, slot, action, history

    for lane, slot, action, history in settings.client.map(
        decide, selected, ordered=False
    ):
        for target, value in zip(actions, action, strict=True):
            target[lane, slot] = int(value)
        if settings.history_turns:
            if lane not in changed:
                changed[lane] = list(memory[lane])
            changed[lane][slot] = history
    if settings.frame == "left":
        with jax.default_device(device):
            actions[0] = np.asarray(
                jax.device_get(_world_moves(jnp.asarray(actions[0]), jnp.asarray(flag)))
            )
    for lane, slot in np.argwhere(active):
        checked = validate_actor_action(
            ActorAction(*(cast(Array, value[lane, slot]) for value in actions)),
            _slice(inputs.action_mask, int(lane), int(slot)),
        )
        channel = evidence.get((int(lane), int(slot)))
        if channel is not None:
            channel.update(world_action=[int(value) for value in checked])
    for lane, values in changed.items():
        histories[lane] = tuple(values)
    return SystemOutput(
        ActorAction(*(cast(Array, value) for value in actions)), histories
    )


def _validate_sampling(settings: Mapping[str, object], max_tokens: int) -> None:
    """Reject unresolved or invalid sampling values before any network request."""
    for name, value in settings.items():
        if name == "ignore_eos":
            valid = type(value) is bool
        elif name in ("seed", "top_k", "min_tokens"):
            valid = type(value) is int
            if valid:
                number = cast(int, value)
                valid = (
                    -(2**63) <= number < 2**63
                    if name == "seed"
                    else number == -1 or number >= 1
                    if name == "top_k"
                    else 0 <= number <= max_tokens
                )
        else:
            valid = type(value) in (int, float) and math.isfinite(cast(float, value))
            if valid:
                number = cast(float, value)
                valid = (
                    0 <= number <= 2
                    if name == "temperature"
                    else 0 < number <= 1
                    if name == "top_p"
                    else 0 <= number <= 1
                    if name == "min_p"
                    else number > 0
                    if name == "repetition_penalty"
                    else -2 <= number <= 2
                )
        if not valid:
            raise ValueError(f"sampling {name} must contain an explicit valid value")


def make_system(
    model: str,
    server_url: str = "http://127.0.0.1:8000/v1",
    *,
    client: Client | None = None,
    server_type: Literal["vllm", "chat"] = "vllm",
    name: str | None = None,
    model_revision: str | None = None,
    history_turns: int = 0,
    frame: Literal["left", "world"] = "left",
    failure_policy: Literal["stop", "fallback"] = "stop",
    records: RecordingMode = "light",
    prompt_builder: PromptBuilder | None = None,
    reply_parser: ReplyParser | None = None,
    token_counter: TokenCounter | Literal["vllm", "auto"] | None = "auto",
    context_limit: int | None = None,
    max_tokens: int = 64,
    sampling: Mapping[str, object] | None = None,
    response_format: Mapping[str, object] | Literal["default"] | None = "default",
    custom_version: str | None = None,
    custom_settings: Mapping[str, object] | None = None,
) -> System:
    """Describe an LLM team using the ordinary host System interface.

    Parameters
    ----------
    model : str
        Nonempty model name sent to the server. No weights are loaded here.
    server_url : str
        Chat API base URL, default local vLLM on port 8000. A supplied client owns
        its own URL and transport settings; server_url then has no effect.
    client : Client | None
        Optional caller-owned client. Otherwise create one lazy client with
        16 request slots, a 60-second timeout and no retry. Construction and init
        make no network calls. Supported runners acquire and close managed
        clients, including after failure. Direct callers must enter the System's
        resource_scope(False) or supply a context-managed Client. Later runs
        reopen owned clients; supplied clients are never closed by the runner.
    server_type : {"vllm", "chat"}
        Default vLLM sends its explicit sampling and thinking controls. Chat
        sends standard Chat Completions fields. Changing the token counter does
        not change these generation settings.
    name : str | None
        Optional display label, default the model name. It participates in saved
        System identity, but the label alone does not prove which method ran.
    model_revision : str | None
        Declared pinned model revision, default unknown. This is not proof of
        the weights loaded by an external server.
    history_turns : int
        Number of previous living turns retained per actor, default 0. Forced
        and fallback choices count. History survives death and clears on reset.
        Oldest whole entries may be excluded from a request to fit its context.
    frame : {"left", "world"}
        Default left frame uses existing reflection helpers. Custom functions
        receive this same frame. Movement returns to world coordinates once.
    failure_policy : {"stop", "fallback"}
        Default stop propagates reply and transport failures. Fallback submits
        checked Stay/no combat for those failures only. Configuration, context,
        custom-code bugs and invalid native return types still stop execution.
    prompt_builder : callable | None
        Optional pure function (actor, masks, history) -> str. Inputs contain
        only that actor's permitted view. May be called repeatedly during fitting.
    reply_parser : callable | None
        Optional (reply, actor, masks, history) -> ActorAction. History is exactly
        what the fitted request used. Raise ReplyFormatError for expected invalid
        replies. Native shapes, categories and original masks are always checked.
    token_counter : {"auto", "vllm", None} or callable
        Default auto selects vLLM /tokenize for a vLLM server and None for chat.
        The vLLM route counts the wrapped chat with thinking disabled.
        A custom pure counter receives the complete request and returns
        (input_tokens, context_limit). None requires history 0 and relies on the
        server's context check. Counted requests must agree with generation usage.
    context_limit : int | None
        Optional positive stricter context limit; requires a token counter.
    max_tokens : int
        Positive reserved reply allowance, default 64, also sent to generation.
    sampling : mapping | None
        Explicit overrides of temperature, top_p, seed, presence_penalty and
        frequency_penalty. Local vLLM also supports top_k, min_p,
        repetition_penalty, min_tokens and ignore_eos. Defaults are resolved
        explicitly and None values are refused. Server-side stop tokens and
        other serving defaults must also be pinned in the experiment recipe;
        request settings alone cannot override every server default.
    response_format : mapping, "default" or None
        Default fixed JSON schema with stable move/combat names. A custom parser
        disables that default. A supplied restriction must match its parser.
        None asks for unrestricted text; the same action checks still apply.
    records : {"none", "light", "full"}
        Saved model-call detail, default light: request hash, reply and checked
        actions. Full also keeps exact request JSON. None skips call files, prompt
        hashing and reply copies. Both teams still receive failure and cost totals.
        Unsaved runs retain only totals. Game completion remains owned by the
        ordinary writer; interrupted unfinished games restart with fresh history.
    custom_version : str | None
        Declared format version for reliable saved custom-method identity.
        Ordinary unsaved custom calls do not require one.
    custom_settings : mapping | None
        Frozen JSON settings used by custom functions, default empty. Undeclared
        mutable closure/global state cannot support reliable recorded resume.

    Returns
    -------
    System
        Ordinary host System with immutable actor-local memory. Its apply builds
        prospective memory; adopt it only after the corresponding successful step.

    Raises
    ------
    ValueError, TypeError
        Settings are invalid or history fitting lacks a trustworthy counter.
        Runtime custom-code errors propagate; failed joint calls keep old memory.
    """
    if not isinstance(cast(object, model), str) or not model.strip():
        raise ValueError("model must be a nonempty name")
    for label, value, lower in (
        ("history_turns", history_turns, 0),
        ("max_tokens", max_tokens, 1),
    ):
        if type(value) is not int or value < lower:
            raise ValueError(f"{label} must be an integer at least {lower}")
    if frame not in ("left", "world") or failure_policy not in ("stop", "fallback"):
        raise ValueError("Choose frame left/world and failure_policy stop/fallback")
    if records not in ("none", "light", "full"):
        raise ValueError("records must be none, light or full")
    if server_type not in ("vllm", "chat"):
        raise ValueError("server_type must be vllm or chat")
    if token_counter == "auto":
        token_counter = "vllm" if server_type == "vllm" else None
    if client is not None and not isinstance(cast(object, client), Client):
        raise TypeError("client must be an LLM Client")
    for label, value in (
        ("model_revision", model_revision),
        ("custom_version", custom_version),
    ):
        if value is not None and (
            not isinstance(cast(object, value), str) or not value.strip()
        ):
            raise ValueError(f"{label} must be nonempty text when supplied")
    for hook in (prompt_builder, reply_parser):
        if hook is not None and not callable(hook):
            raise TypeError("Custom prompt and reply functions must be callable")
    if (
        token_counter != "vllm"
        and token_counter is not None
        and not callable(token_counter)
    ):
        raise TypeError("token_counter must be vllm, a callable or None")
    if context_limit is not None and (
        type(context_limit) is not int or context_limit < 1
    ):
        raise ValueError("context_limit must be a positive integer")
    if token_counter is None and (history_turns or context_limit is not None):
        raise ValueError(
            "History and explicit context limits require a trustworthy token counter"
        )
    generation: dict[str, object] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    if server_type == "vllm":
        generation.update(
            top_k=-1, min_p=0.0, repetition_penalty=1.0, min_tokens=0, ignore_eos=False
        )
    if sampling is not None:
        if not set(sampling) <= generation.keys():
            raise ValueError(
                "sampling contains an unsupported or routing-related option"
            )
        generation.update(sampling)
    _validate_sampling(generation, max_tokens)
    generation.update(max_tokens=max_tokens, n=1, stream=False)
    if server_type == "vllm":
        generation["chat_template_kwargs"] = {"enable_thinking": False}
    if response_format == "default":
        if reply_parser is None:
            generation["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "marl_actor_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "move": {"type": "string", "enum": MOVE_NAMES},
                            "combat": {
                                "type": "string",
                                "enum": tuple(
                                    name for pair in COMBAT_NAMES for name in pair
                                ),
                            },
                        },
                        "required": ["move", "combat"],
                        "additionalProperties": False,
                    },
                },
            }
    elif response_format is not None:
        generation["response_format"] = dict(response_format)
    generation_json = json.dumps(generation, allow_nan=False, sort_keys=True)
    custom = (
        prompt_builder is not None
        or reply_parser is not None
        or callable(token_counter)
    )
    selected_client = client or Client(server_url)
    hooks = {
        "prompt_builder": _callable_evidence(prompt_builder),
        "reply_parser": _callable_evidence(reply_parser),
        "token_counter": _callable_evidence(token_counter)
        if callable(token_counter)
        else None,
    }
    identity = {
        "model": model,
        "server_type": server_type,
        "model_revision": model_revision,
        "text_version": TEXT_VERSION,
        "frame": frame,
        "history_turns": history_turns,
        "failure_policy": failure_policy,
        "generation": json.loads(generation_json),
        "context_limit": context_limit,
        "counter": "vllm"
        if token_counter == "vllm"
        else "custom"
        if callable(token_counter)
        else None,
        "custom_version": custom_version,
        "custom_settings": dict(custom_settings or {}),
        "custom_hooks": hooks,
        "transport": {
            "concurrency": selected_client.concurrency,
            "timeout": selected_client.timeout,
            "max_response_bytes": selected_client.max_response_bytes,
        },
    }
    identity_json = json.dumps(identity, allow_nan=False, sort_keys=True)
    identity_known = not custom or (
        bool(custom_version)
        and all(
            value is None or value["code_digest"] is not None
            for value in hooks.values()
        )
    )
    resources = _Resources(
        selected_client,
        owned=client is None,
        identity_known=identity_known,
        identity_json=identity_json,
    )
    settings = _Settings(
        model,
        resources,
        client is not None,
        history_turns,
        frame,
        failure_policy,
        prompt_builder,
        reply_parser,
        token_counter,
        context_limit,
        generation_json,
        identity_json,
        identity_known,
        records,
        canonical_digest_sha256(identity),
    )
    return System(
        name or model,
        _apply,
        variables=settings,
        init=_init,
        execution="host",
        resource_scope=resources.scope,
        components=(
            {
                "name": name or model,
                "version": 1,
                "parameters_digest": canonical_digest_sha256(identity),
            },
        ),
    )
