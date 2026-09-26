"""Evaluate an LLM with default JSON or ordinary custom prompt/parser functions.

Run ``python examples/llm.py --model MODEL --server-url URL`` with an independently
running vLLM server. Add ``--format words`` to use a two-word reply. This script
owns its explicitly supplied Client and closes it on success or failure. It never
starts/stops a server or downloads weights. Optional game output includes separate
model-call evidence. Both teams' failure and cost totals are printed with outcomes.
"""

import argparse
import json
import os
from pathlib import Path

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.llm.actions import COMBAT_NAMES, MOVE_NAMES
from marl_battlegrounds.llm.system import History
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput


def _view_and_history(
    actor: ActorInput,
    masks: ActionMask,
    history: History,
    *,
    reply_instruction: str,
) -> str:
    """Compose permitted views with clear dates and one current reply instruction.

    actor/masks describe the current left-frame decision. history contains only
    this actor's earlier living turns. Historical menus never govern the current
    reply. No own scheduler, memory store or native action rules are needed.
    """
    parts = ["You control the actor described by Self."]
    for entry in history:
        parts.extend(
            [
                f"Earlier turn {entry.timestep}; its menu is historical:",
                llm.format_actor_view(
                    entry.actor,
                    entry.masks,
                    include_static=False,
                    reply_instruction=None,
                ),
                f"Submitted: move={MOVE_NAMES[int(entry.action.move)]} "
                f"combat={COMBAT_NAMES[int(entry.action.select_target)][int(entry.action.use_ultimate)]}",
            ]
        )
    parts.extend(
        [
            "Current decision:",
            llm.format_actor_view(actor, masks, reply_instruction=reply_instruction),
        ]
    )
    return "\n".join(parts)


def prompt_with_note(actor: ActorInput, masks: ActionMask, history: History) -> str:
    """Change the presentation while retaining the default named JSON parser.

    Use this example with make_system's default left frame. This pure
    function may run again if context fitting removes an older history entry.
    """
    return _view_and_history(
        actor,
        masks,
        history,
        reply_instruction=(
            'Reply with exactly {"move":"<legal move>","combat":"<legal combat>"}.'
        ),
    )


def prompt_for_words(actor: ActorInput, masks: ActionMask, history: History) -> str:
    """Ask for two stable names using make_system's default left-frame inputs."""
    return _view_and_history(
        actor,
        masks,
        history,
        reply_instruction="Reply with two names on one line: <move> <combat>.",
    )


def parse_words(
    reply: str, actor: ActorInput, masks: ActionMask, history: History
) -> ActorAction:
    """Read two names and reuse the shared name mapping and legal-pair check.

    actor, masks and history describe this exact fitted request. This parser
    needs only masks. Bad text raises ReplyFormatError. make_system checks the
    returned native ActorAction again and owns world-frame conversion.
    """
    del actor, history
    words = reply.split()
    if len(words) != 2:
        raise llm.ReplyFormatError("Reply must contain one move and one combat name")
    return llm.parse_action_reply(
        json.dumps({"move": words[0], "combat": words[1]}), masks
    )


def _factory(*, custom: bool) -> marl_bgs.System:
    """Build a lazy CLI/DevClient System from the example's environment settings.

    MARL_LLM_MODEL is the required served model name. MARL_LLM_URL defaults to
    http://127.0.0.1:8000/v1. MARL_LLM_REVISION optionally records pinned model
    content; omission means unknown. MARL_LLM_HISTORY defaults to 0 earlier turns.
    custom selects the two-word tutorial functions. No request or server starts.
    """
    model = os.environ.get("MARL_LLM_MODEL")
    if not model:
        raise ValueError("Set MARL_LLM_MODEL to the model name served by your server")
    return llm.make_system(
        model,
        os.environ.get("MARL_LLM_URL", "http://127.0.0.1:8000/v1"),
        model_revision=os.environ.get("MARL_LLM_REVISION"),
        history_turns=int(os.environ.get("MARL_LLM_HISTORY", "0")),
        prompt_builder=prompt_for_words if custom else None,
        reply_parser=parse_words if custom else None,
        custom_version="tutorial-v1" if custom else None,
        custom_settings={"format": "words" if custom else "default"},
    )


def make_default_system() -> marl_bgs.System:
    """Return the default JSON System for a trusted module:function declaration.

    Set MARL_LLM_MODEL first. Optional MARL_LLM_URL, MARL_LLM_REVISION and
    MARL_LLM_HISTORY select the endpoint, declared content and earlier turns.
    This zero-argument factory starts no request. The runner owns its client.
    """
    return _factory(custom=False)


def make_custom_system() -> marl_bgs.System:
    """Return the two-word tutorial System with the same managed runner lifetime.

    Use the same environment settings as make_default_system. Only prompt and
    reply format change; history, action checks, recording and cleanup are shared.
    """
    return _factory(custom=True)


def main() -> None:
    """Parse example settings, evaluate through the public API and print outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--format", choices=("default", "note", "words"), default="default"
    )
    parser.add_argument("--history-turns", type=int, default=0)
    parser.add_argument("--records", choices=("none", "light", "full"), default="light")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--map-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    builder = (
        prompt_with_note
        if args.format == "note"
        else prompt_for_words
        if args.format == "words"
        else None
    )
    decoder = parse_words if args.format == "words" else None
    with llm.Client(args.server_url) as client:
        system = llm.make_system(
            args.model,
            client=client,
            history_turns=args.history_turns,
            records=args.records,
            prompt_builder=builder,
            reply_parser=decoder,
            custom_version="tutorial-v1" if builder else None,
            custom_settings={"format": args.format},
        )
        result = marl_bgs.evaluate(
            system,
            "random",
            num_episodes=args.games,
            maps=[args.map_id],
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
    episodes = result.table("episodes")
    print(
        {
            column: episodes[column]
            for column in (
                "episode_id",
                "outcome",
                "episode_length",
                "team_a_score",
                "team_b_score",
            )
        }
    )
    print("Model calls:", llm.call_summary(result))
    if result.paths:
        print("Game files:", result.paths)


if __name__ == "__main__":
    main()
