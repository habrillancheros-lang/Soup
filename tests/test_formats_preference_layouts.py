"""Preference row layouts: what the ``dpo`` and ``kto`` converters accept, emit and refuse.

#1849: every bundled DPO example writes ``prompt`` as a string and ``chosen`` /
``rejected`` as message lists that repeat the user turn. ``_convert_dpo`` passed
that through unchanged, and trl decides "conversational or text" by asking ONE
key of the row, taken from a set (``trl.data_utils.is_conversational``). A row
whose three fields are not the same kind therefore failed inside trl's
``DPOTrainer`` (dpo, ipo) and, unpaired by Soup, inside BCO, after the model
had loaded, with an error that changed with ``PYTHONHASHSEED``.

The contract pinned here is the one that makes the key trl asks irrelevant: a
converted row is either TEXT (``prompt``, ``chosen`` and ``rejected`` are all
strings) or CONVERSATIONAL (all three are non-empty lists of messages, the
prompt turns in ``prompt`` and nowhere else). The spellings are enumerated first
and tested one by one, as are the refusals, each by the message a user would
search for.

Nothing here imports torch or trl; the trainers are driven in
``tests/test_trainer_preference_layouts.py``.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from rich.console import Console

import soup_cli.data.loader as loader
from soup_cli.config.loader import load_config_from_string
from soup_cli.data.formats import (
    detect_format,
    format_to_messages,
    format_to_messages_with_reason,
    unify_preference_layout,
)
from soup_cli.data.validator import validate_and_stats

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples" / "data"
_FIXTURES = _ROOT / "src" / "soup_cli" / "data" / "_fixtures"
_TEMPLATES = _ROOT / "src" / "soup_cli" / "templates"

Q = "What is 2+2?"
SYS = {"role": "system", "content": "Be brief."}
USER = {"role": "user", "content": Q}
GOOD = {"role": "assistant", "content": "4"}
BAD = {"role": "assistant", "content": "Fish"}
USER2 = {"role": "user", "content": "And 3+3?"}
GOOD2 = {"role": "assistant", "content": "6"}
BAD2 = {"role": "assistant", "content": "Cat"}
ASKED = {"role": "user", "content": Q}  # what a string prompt becomes
SAID_GOOD = {"role": "assistant", "content": "4"}  # what a string answer becomes
SAID_BAD = {"role": "assistant", "content": "Fish"}
IMAGE_TURN = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}
TOOL_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"type": "function", "function": {"name": "add", "arguments": "{}"}}],
}
#: The legacy spelling of a tool call, and the turn an export leaves behind
#: when it writes a ``tool_calls`` list on every turn. Both loaded and trained
#: before this converter looked at messages, so both still convert.
FUNCTION_CALL = {
    "role": "assistant",
    "content": None,
    "function_call": {"name": "add", "arguments": "{}"},
}
NO_CALLS = {"role": "assistant", "content": None, "tool_calls": []}
TOOL_RESULT = {"role": "tool", "content": '{"sum": 4}'}
TOOL_RESULT_OBJECT = {"role": "tool", "content": {"sum": 4}}
_NO_TEXT = " (a turn without text needs tool_calls or function_call)"


def _layout(row: dict, fields: tuple[str, ...]) -> str:
    """``text`` or ``conversational``, asserting the row is exactly one of them.

    This is the property trl needs: whichever of the fields it inspects, the
    answer to "is this a conversation?" is the same.
    """
    values = [row[name] for name in fields]
    if all(isinstance(value, str) for value in values):
        return "text"
    for name, value in zip(fields, values):
        assert isinstance(value, list), f"{name} is a {type(value).__name__} beside a message list"
        assert value, f"{name} is an empty message list"
        for message in value:
            assert isinstance(message, dict), f"{name} holds a {type(message).__name__}"
            assert isinstance(message.get("role"), str) and message["role"], f"{name}: no role"
    return "conversational"


_DPO = ("prompt", "chosen", "rejected")
_KTO = ("prompt", "completion")


# --------------------------------------------------------------------------
# dpo: the spellings that convert
# --------------------------------------------------------------------------
_DPO_ACCEPTED = {
    # --- text: three strings, handed to trl exactly as written -----------------
    "text": (
        {"prompt": Q, "chosen": "4", "rejected": "Fish"},
        {"prompt": Q, "chosen": "4", "rejected": "Fish"},
    ),
    "text, empty strings": (
        {"prompt": "", "chosen": "", "rejected": ""},
        {"prompt": "", "chosen": "", "rejected": ""},
    ),
    "text, identical answers": (
        {"prompt": Q, "chosen": "4", "rejected": "4"},
        {"prompt": Q, "chosen": "4", "rejected": "4"},
    ),
    "text, extra keys are not carried": (
        {"prompt": Q, "chosen": "4", "rejected": "Fish", "prompt_id": "a1", "score_chosen": 9.0},
        {"prompt": Q, "chosen": "4", "rejected": "Fish"},
    ),
    # A top-level ``system`` key was never read for dpo rows; that is unchanged
    # (a system prompt belongs in the prompt message list).
    "text, a top-level system key is not read": (
        {"system": "Be brief.", "prompt": Q, "chosen": "4", "rejected": "Fish"},
        {"prompt": Q, "chosen": "4", "rejected": "Fish"},
    ),
    # --- the bundled layout (#1849) and its relatives --------------------------
    "bundled: string prompt, lists repeat the user turn": (
        {"prompt": Q, "chosen": [USER, GOOD], "rejected": [USER, BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "bundled, with a shared system turn": (
        {"prompt": Q, "chosen": [SYS, USER, GOOD], "rejected": [SYS, USER, BAD]},
        {"prompt": [SYS, USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "bundled, the prompt is the last turn of a shared history": (
        {
            "prompt": "And 3+3?",
            "chosen": [USER, GOOD, USER2, GOOD2],
            "rejected": [USER, GOOD, USER2, BAD2],
        },
        {"prompt": [USER, GOOD, USER2], "chosen": [GOOD2], "rejected": [BAD2]},
    ),
    # trl's own reading of this layout (``maybe_extract_prompt``): the prompt is
    # in the conversations, the string column beside them is not read. ORPO,
    # SimPO and BCO have always trained such a row this way.
    "bundled, the string says something the shared turns do not": (
        {"prompt": "Something else", "chosen": [USER, GOOD], "rejected": [USER, BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "bundled, UltraFeedback's extra columns": (
        {
            "prompt": Q,
            "prompt_id": "a1",
            "chosen": [USER, GOOD],
            "rejected": [USER, BAD],
            "messages": [USER, GOOD],
            "score_chosen": 9.0,
            "score_rejected": 2.0,
        },
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "string prompt, lists hold the answers only": (
        {"prompt": Q, "chosen": [GOOD], "rejected": [BAD]},
        {"prompt": [ASKED], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "string prompt, answers are single messages": (
        {"prompt": Q, "chosen": GOOD, "rejected": BAD},
        {"prompt": [ASKED], "chosen": [GOOD], "rejected": [BAD]},
    ),
    # Kept convertible on purpose: `soup data lint` reports identical pairs from
    # the converted rows, and a text row with chosen == rejected converts too.
    "bundled, identical conversations keep their last turn": (
        {"prompt": Q, "chosen": [USER, GOOD], "rejected": [USER, GOOD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [GOOD]},
    ),
    "identical single answers": (
        {"prompt": Q, "chosen": [GOOD], "rejected": [GOOD]},
        {"prompt": [ASKED], "chosen": [GOOD], "rejected": [GOOD]},
    ),
    # The rejected conversation ends on an answer the chosen one goes on from:
    # that answer is the rejected one, as trl splits it.
    "one conversation stops after an answer the other continues": (
        {"chosen": [USER, GOOD, USER2, GOOD2], "rejected": [USER, GOOD]},
        {"prompt": [USER], "chosen": [GOOD, USER2, GOOD2], "rejected": [GOOD]},
    ),
    # A shared turn is the same message, not a turn of the same role: here the
    # two answers part at the first assistant turn, although every later turn
    # has the role of its opposite number.
    "the answers part early and go on for several turns": (
        {
            "prompt": Q,
            "chosen": [USER, GOOD, USER2, GOOD2],
            "rejected": [USER, BAD, USER2, BAD2],
        },
        {"prompt": [USER], "chosen": [GOOD, USER2, GOOD2], "rejected": [BAD, USER2, BAD2]},
    ),
    # Role order is not checked. Two answers that open with different user
    # turns share nothing, so the string is the user turn and both are kept
    # whole: nothing is merged on the strength of a role.
    "the answers open with different user turns and share none": (
        {"prompt": Q, "chosen": [USER, GOOD], "rejected": [USER2, BAD]},
        {"prompt": [ASKED], "chosen": [USER, GOOD], "rejected": [USER2, BAD]},
    ),
    # --- trl's explicit layout --------------------------------------------------
    "explicit": (
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "explicit, lists repeat the prompt": (
        {"prompt": [SYS, USER], "chosen": [SYS, USER, GOOD], "rejected": [SYS, USER, BAD]},
        {"prompt": [SYS, USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    # Only a prompt that EVERY answer repeats and continues is taken out of
    # them. A row where one answer repeats it and the other does not is left as
    # written, as it always was (that answer still carries the prompt twice).
    "explicit, only one answer repeats the prompt: left as written": (
        {"prompt": [SYS, USER], "chosen": [SYS, USER, GOOD], "rejected": [BAD]},
        {"prompt": [SYS, USER], "chosen": [SYS, USER, GOOD], "rejected": [BAD]},
    ),
    "explicit, an answer that is only the prompt is left as written": (
        {"prompt": [USER], "chosen": [USER], "rejected": [USER, BAD]},
        {"prompt": [USER], "chosen": [USER], "rejected": [USER, BAD]},
    ),
    "explicit, an answer that begins with part of the prompt is left as written": (
        {"prompt": [SYS, USER], "chosen": [SYS, GOOD], "rejected": [SYS, USER2, BAD]},
        {"prompt": [SYS, USER], "chosen": [SYS, GOOD], "rejected": [SYS, USER2, BAD]},
    ),
    "explicit, the prompt is one message": (
        {"prompt": USER, "chosen": [GOOD], "rejected": [BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "explicit, string answers": (
        {"prompt": [SYS, USER], "chosen": "4", "rejected": "Fish"},
        {"prompt": [SYS, USER], "chosen": [SAID_GOOD], "rejected": [SAID_BAD]},
    ),
    "explicit, answers of several turns": (
        {"prompt": [USER], "chosen": [GOOD, USER2, GOOD2], "rejected": [BAD, USER2, BAD2]},
        {"prompt": [USER], "chosen": [GOOD, USER2, GOOD2], "rejected": [BAD, USER2, BAD2]},
    ),
    "explicit, multimodal content blocks": (
        {"prompt": [IMAGE_TURN], "chosen": [GOOD], "rejected": [BAD]},
        {"prompt": [IMAGE_TURN], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "explicit, an assistant tool call without content": (
        {"prompt": [USER], "chosen": [TOOL_CALL], "rejected": [BAD]},
        {"prompt": [USER], "chosen": [TOOL_CALL], "rejected": [BAD]},
    ),
    "explicit, a legacy function_call without content": (
        {"prompt": [USER], "chosen": [FUNCTION_CALL], "rejected": [BAD]},
        {"prompt": [USER], "chosen": [FUNCTION_CALL], "rejected": [BAD]},
    ),
    "explicit, an empty tool_calls list and no content": (
        {"prompt": [USER], "chosen": [NO_CALLS], "rejected": [BAD]},
        {"prompt": [USER], "chosen": [NO_CALLS], "rejected": [BAD]},
    ),
    "bundled, a tool call and its result are part of the shared turns": (
        {
            "prompt": Q,
            "chosen": [USER, TOOL_CALL, TOOL_RESULT, GOOD],
            "rejected": [USER, TOOL_CALL, TOOL_RESULT, BAD],
        },
        {"prompt": [USER, TOOL_CALL, TOOL_RESULT], "chosen": [GOOD], "rejected": [BAD]},
    ),
    # --- no prompt key ------------------------------------------------------------
    "no prompt: trl's implicit layout": (
        {"chosen": [USER, GOOD], "rejected": [USER, BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "an empty string prompt counts as none": (
        {"prompt": " ", "chosen": [USER, GOOD], "rejected": [USER, BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "an empty list prompt counts as none": (
        {"prompt": [], "chosen": [USER, GOOD], "rejected": [USER, BAD]},
        {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "no prompt: messages carry it, list answers": (
        {"messages": [SYS, USER], "chosen": [GOOD], "rejected": [BAD]},
        {"prompt": [SYS, USER], "chosen": [GOOD], "rejected": [BAD]},
    ),
    "no prompt: messages carry it, string answers": (
        {"messages": [SYS, USER], "chosen": "4", "rejected": "Fish"},
        {"prompt": [SYS, USER], "chosen": [SAID_GOOD], "rejected": [SAID_BAD]},
    ),
}


class TestDpoSpellingsThatConvert:
    @pytest.mark.parametrize("name", list(_DPO_ACCEPTED))
    def test_the_row_converts_to_exactly_this(self, name):
        row, expected = _DPO_ACCEPTED[name]
        converted, reason = format_to_messages_with_reason(row, "dpo")
        assert reason is None, reason
        assert converted == expected

    @pytest.mark.parametrize("name", list(_DPO_ACCEPTED))
    def test_the_three_fields_are_one_kind(self, name):
        row, expected = _DPO_ACCEPTED[name]
        converted = format_to_messages(row, "dpo")
        assert set(converted) == set(_DPO)
        assert _layout(converted, _DPO) == _layout(expected, _DPO)

    @pytest.mark.parametrize("name", list(_DPO_ACCEPTED))
    def test_the_source_row_is_left_as_it_was(self, name):
        row, _ = _DPO_ACCEPTED[name]
        before = json.dumps(row, sort_keys=True)
        format_to_messages(row, "dpo")
        assert json.dumps(row, sort_keys=True) == before

    @pytest.mark.parametrize("name", list(_DPO_ACCEPTED))
    def test_converting_twice_changes_nothing(self, name):
        """A converted row is itself a spelling the converter accepts."""
        row, _ = _DPO_ACCEPTED[name]
        once = format_to_messages(row, "dpo")
        assert format_to_messages(once, "dpo") == once

    def test_the_prompt_turns_are_in_the_prompt_and_nowhere_else(self):
        """The reported duplication: the user turn sat in ``chosen`` as well."""
        row, _ = _DPO_ACCEPTED["bundled: string prompt, lists repeat the user turn"]
        converted = format_to_messages(row, "dpo")
        assert USER in converted["prompt"]
        assert USER not in converted["chosen"] and USER not in converted["rejected"]


# --------------------------------------------------------------------------
# dpo: the spellings that are refused, each by its message
# --------------------------------------------------------------------------
_DPO_REFUSED = {
    "chosen is null": (
        {"prompt": Q, "chosen": None, "rejected": "Fish"},
        "dpo.chosen must not be null",
    ),
    "rejected is missing": (
        {"prompt": Q, "chosen": "4"},
        "dpo.rejected must not be null",
    ),
    "no prompt beside string answers": (
        {"chosen": "4", "rejected": "Fish"},
        "dpo.prompt must not be null",
    ),
    "null prompt beside string answers": (
        {"prompt": None, "chosen": "4", "rejected": "Fish"},
        "dpo.prompt must not be null",
    ),
    "no prompt and the lists share no leading turn": (
        {"chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt is missing and dpo.chosen and dpo.rejected share no leading turn",
    ),
    # not a user turn that says nothing
    "a blank string prompt and the lists share no leading turn": (
        {"prompt": "  ", "chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt is missing and dpo.chosen and dpo.rejected share no leading turn",
    ),
    "chosen is a list, rejected a string": (
        {"prompt": Q, "chosen": [USER, GOOD], "rejected": "Fish"},
        "dpo.chosen is a message list and dpo.rejected is a string: give both as "
        "strings or both as message lists",
    ),
    "chosen is a string, rejected a list": (
        {"prompt": Q, "chosen": "4", "rejected": [USER, BAD]},
        "dpo.chosen is a string and dpo.rejected is a message list: give both as "
        "strings or both as message lists",
    ),
    "rejected stops where the answer should start": (
        {"prompt": Q, "chosen": [USER, GOOD], "rejected": [USER]},
        "dpo.rejected has no turn after the 1 it shares with dpo.chosen",
    ),
    "chosen stops where the answer should start": (
        {"chosen": [SYS, USER], "rejected": [SYS, USER, BAD]},
        "dpo.chosen has no turn after the 2 it shares with dpo.rejected",
    ),
    "an empty answer list": (
        {"prompt": Q, "chosen": [], "rejected": [USER, BAD]},
        "dpo.chosen is an empty message list",
    ),
    "the prompt is a number": (
        {"prompt": 7, "chosen": "4", "rejected": "Fish"},
        "dpo.prompt must be a string, a message or a list of messages, got int",
    ),
    "an answer is a number": (
        {"prompt": Q, "chosen": 4, "rejected": "Fish"},
        "dpo.chosen must be a string, a message or a list of messages, got int",
    ),
    "an answer is a boolean": (
        {"prompt": Q, "chosen": "4", "rejected": False},
        "dpo.rejected must be a string, a message or a list of messages, got bool",
    ),
    "a list element is not a message": (
        {"prompt": Q, "chosen": [USER, "4"], "rejected": [USER, BAD]},
        "dpo.chosen[1] must be a message (an object with 'role' and 'content'), got str",
    ),
    "a message has no role": (
        {"prompt": Q, "chosen": [USER, {"content": "4"}], "rejected": [USER, BAD]},
        "dpo.chosen[1].role must be a non-empty string",
    ),
    "a message's role is an empty string": (
        {"prompt": [{"role": "", "content": Q}], "chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt[0].role must be a non-empty string",
    ),
    "a message's role is a number": (
        {"prompt": [{"role": 1, "content": Q}], "chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt[0].role must be a non-empty string",
    ),
    "a message's role is a list": (
        {"prompt": [USER], "chosen": [GOOD], "rejected": [{"role": ["assistant"], "content": "x"}]},
        "dpo.rejected[0].role must be a non-empty string",
    ),
    # A blank ``messages`` key is no prompt, like a blank ``prompt``: the row is
    # told it has none, not that the key it did not mean to use is malformed.
    "an empty messages list is no prompt": (
        {"messages": [], "chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt is missing and dpo.chosen and dpo.rejected share no leading turn",
    ),
    "a blank messages string is no prompt": (
        {"messages": "  ", "chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt is missing and dpo.chosen and dpo.rejected share no leading turn",
    ),
    "a blank messages key beside string answers is no prompt": (
        {"messages": [], "chosen": "4", "rejected": "Fish"},
        "dpo.prompt must not be null",
    ),
    "an answer is an object that is not a message": (
        {"prompt": Q, "chosen": {"text": "4"}, "rejected": BAD},
        "dpo.chosen[0].role must be a non-empty string",
    ),
    "a message's content is null": (
        {
            "prompt": Q,
            "chosen": [USER, {"role": "assistant", "content": None}],
            "rejected": [USER, BAD],
        },
        "dpo.chosen[1].content must be a string or a list of content blocks, got NoneType"
        + _NO_TEXT,
    ),
    "a message has no content": (
        {"prompt": [{"role": "user"}], "chosen": [GOOD], "rejected": [BAD]},
        "dpo.prompt[0].content must be a string or a list of content blocks, got NoneType"
        + _NO_TEXT,
    ),
    # A null ``tool_calls`` is no call (Arrow writes the key on every turn).
    "a message's content and its tool_calls are both null": (
        {
            "prompt": [USER],
            "chosen": [{"role": "assistant", "content": None, "tool_calls": None}],
            "rejected": [BAD],
        },
        "dpo.chosen[0].content must be a string or a list of content blocks, got NoneType"
        + _NO_TEXT,
    ),
    # Never loaded: beside string contents Arrow refuses it ("cannot mix struct
    # and non-struct, non-null values"), after the model has loaded.
    "a tool result that is a JSON object": (
        {
            "prompt": Q,
            "chosen": [USER, TOOL_CALL, TOOL_RESULT_OBJECT, GOOD],
            "rejected": [USER, TOOL_CALL, TOOL_RESULT_OBJECT, BAD],
        },
        "dpo.chosen[2].content must be a string or a list of content blocks, got dict "
        "(write a JSON tool result as a JSON string)",
    ),
    "a message's content is a number": (
        {
            "prompt": Q,
            "chosen": [USER, GOOD],
            "rejected": [USER, {"role": "assistant", "content": 4}],
        },
        "dpo.rejected[1].content must be a string or a list of content blocks, got int",
    ),
    "a message's content is a boolean": (
        {"prompt": [USER], "chosen": [GOOD], "rejected": [{"role": "assistant", "content": True}]},
        "dpo.rejected[0].content must be a string or a list of content blocks, got bool",
    ),
    "messages that are not a list, used as the prompt": (
        {"messages": "What is 2+2?", "chosen": [GOOD], "rejected": [BAD]},
        "dpo.messages must be a message or a list of messages, got str",
    ),
}


class TestDpoSpellingsThatAreRefused:
    @pytest.mark.parametrize("name", list(_DPO_REFUSED))
    def test_the_row_is_dropped_with_this_reason(self, name):
        row, message = _DPO_REFUSED[name]
        converted, reason = format_to_messages_with_reason(row, "dpo")
        assert converted is None
        assert message in reason

    @pytest.mark.parametrize("name", list(_DPO_REFUSED))
    def test_the_plain_entry_point_drops_it_too(self, name):
        row, _ = _DPO_REFUSED[name]
        assert format_to_messages(row, "dpo") is None

    def test_the_two_conversations_that_are_one_question_have_no_answer(self):
        """Identical conversations convert when they end on an answer (above);
        when they end on the question there is nothing to prefer."""
        converted, reason = format_to_messages_with_reason(
            {"prompt": Q, "chosen": [USER], "rejected": [USER]}, "dpo"
        )
        assert converted is None
        assert "dpo.chosen has no turn after the 1 it shares with dpo.rejected" in reason


# --------------------------------------------------------------------------
# kto: its own converter, the same contract over prompt + completion
# --------------------------------------------------------------------------
_KTO_ACCEPTED = {
    "text": (
        {"prompt": Q, "completion": "4", "label": True},
        {"prompt": Q, "completion": "4", "label": True},
    ),
    "text, an undesirable completion": (
        {"prompt": Q, "completion": "Fish", "label": False},
        {"prompt": Q, "completion": "Fish", "label": False},
    ),
    "explicit": (
        {"prompt": [USER], "completion": [GOOD], "label": True},
        {"prompt": [USER], "completion": [GOOD], "label": True},
    ),
    "explicit, the completion repeats the prompt": (
        {"prompt": [SYS, USER], "completion": [SYS, USER, BAD], "label": False},
        {"prompt": [SYS, USER], "completion": [BAD], "label": False},
    ),
    "string prompt, the completion repeats it as the user turn": (
        {"prompt": Q, "completion": [USER, GOOD], "label": "true"},
        {"prompt": [USER], "completion": [GOOD], "label": True},
    ),
    "string prompt, the completion leads with system and user turns": (
        {"prompt": Q, "completion": [SYS, USER, BAD], "label": 0},
        {"prompt": [SYS, USER], "completion": [BAD], "label": False},
    ),
    "string prompt, the leading turns say something else and are the prompt": (
        {"prompt": "Something else", "completion": [USER, GOOD], "label": True},
        {"prompt": [USER], "completion": [GOOD], "label": True},
    ),
    "string prompt, the completion is the answer only": (
        {"prompt": Q, "completion": [GOOD], "label": 1},
        {"prompt": [ASKED], "completion": [GOOD], "label": True},
    ),
    "string prompt, the completion is one message": (
        {"prompt": Q, "completion": GOOD, "label": True},
        {"prompt": [ASKED], "completion": [GOOD], "label": True},
    ),
    "explicit, a completion that is only the prompt is left as written": (
        {"prompt": [USER], "completion": [USER], "label": False},
        {"prompt": [USER], "completion": [USER], "label": False},
    ),
    # Longer than the prompt is not enough: its leading turns have to BE the prompt.
    "explicit, a completion that opens with another turn is left as written": (
        {"prompt": [USER], "completion": [USER2, GOOD], "label": True},
        {"prompt": [USER], "completion": [USER2, GOOD], "label": True},
    ),
    "list prompt, string completion": (
        {"prompt": [USER], "completion": "4", "label": True},
        {"prompt": [USER], "completion": [SAID_GOOD], "label": True},
    ),
    "one-message prompt, string completion": (
        {"prompt": USER, "completion": "Fish", "label": -1},
        {"prompt": [USER], "completion": [SAID_BAD], "label": False},
    ),
}

_KTO_REFUSED = {
    "no prompt": (
        {"completion": "4", "label": True},
        "kto.prompt must not be null",
    ),
    "null completion": (
        {"prompt": Q, "completion": None, "label": True},
        "kto.completion must not be null",
    ),
    "the completion stops where the answer should start": (
        {"prompt": Q, "completion": [SYS, USER], "label": True},
        "kto.completion has no turn after the 2 that lead it",
    ),
    "an empty completion list": (
        {"prompt": Q, "completion": [], "label": True},
        "kto.completion is an empty message list",
    ),
    "an empty prompt list": (
        {"prompt": [], "completion": [GOOD], "label": True},
        "kto.prompt is an empty message list",
    ),
    "the prompt is a number": (
        {"prompt": 7, "completion": "4", "label": True},
        "kto.prompt must be a string, a message or a list of messages, got int",
    ),
    "a message's content is null": (
        {"prompt": [USER], "completion": [{"role": "assistant", "content": None}], "label": True},
        "kto.completion[0].content must be a string or a list of content blocks, got NoneType",
    ),
    "an unparseable label": (
        {"prompt": Q, "completion": "4", "label": "maybe"},
        "KTO label must be true/false",
    ),
}


class TestKtoSpellings:
    @pytest.mark.parametrize("name", list(_KTO_ACCEPTED))
    def test_the_row_converts_to_exactly_this(self, name):
        row, expected = _KTO_ACCEPTED[name]
        converted, reason = format_to_messages_with_reason(row, "kto")
        assert reason is None, reason
        assert converted == expected
        assert converted["label"] is expected["label"]
        _layout(converted, _KTO)

    @pytest.mark.parametrize("name", list(_KTO_ACCEPTED))
    def test_converting_twice_changes_nothing(self, name):
        row, _ = _KTO_ACCEPTED[name]
        once = format_to_messages(row, "kto")
        assert format_to_messages(once, "kto") == once

    @pytest.mark.parametrize("name", list(_KTO_REFUSED))
    def test_the_row_is_dropped_with_this_reason(self, name):
        row, message = _KTO_REFUSED[name]
        converted, reason = format_to_messages_with_reason(row, "kto")
        assert converted is None
        assert message in reason


# --------------------------------------------------------------------------
# one layout per dataset: a file that mixes text and conversational rows
# --------------------------------------------------------------------------
_TEXT_ROW = {"prompt": Q, "chosen": "4", "rejected": "Fish"}
_CHAT_ROW = {"prompt": [USER2], "chosen": [GOOD2], "rejected": [BAD2]}
_TEXT_ROW_AS_CHAT = {"prompt": [ASKED], "chosen": [SAID_GOOD], "rejected": [SAID_BAD]}


class TestOneLayoutPerDataset:
    """``Dataset.from_list`` needs one type per column, and trl reads the layout
    off the first row only. A text row beside conversational ones is written as
    one user turn and one assistant turn, the same way a chatml row beside
    tool-calling ones loads as tool-calling."""

    def test_text_rows_alone_are_returned_untouched(self):
        rows = [dict(_TEXT_ROW), dict(_TEXT_ROW)]
        unified, promoted = unify_preference_layout(rows)
        assert unified is rows and promoted == 0

    def test_conversational_rows_alone_are_returned_untouched(self):
        rows = [dict(_CHAT_ROW), dict(_CHAT_ROW)]
        unified, promoted = unify_preference_layout(rows)
        assert unified is rows and promoted == 0

    def test_rows_of_another_format_are_returned_untouched(self):
        rows = [{"messages": [USER, GOOD]}, {"text": "plain"}, {"prompt": "only a prompt"}]
        unified, promoted = unify_preference_layout(rows)
        assert unified is rows and promoted == 0

    def test_no_rows(self):
        assert unify_preference_layout([]) == ([], 0)

    @pytest.mark.parametrize("order", ["text first", "conversation first"])
    def test_a_text_row_beside_a_conversation_becomes_one(self, order):
        rows = [_TEXT_ROW, _CHAT_ROW] if order == "text first" else [_CHAT_ROW, _TEXT_ROW]
        expected = (
            [_TEXT_ROW_AS_CHAT, _CHAT_ROW]
            if order == "text first"
            else [_CHAT_ROW, _TEXT_ROW_AS_CHAT]
        )
        unified, promoted = unify_preference_layout(rows)
        assert unified == expected
        assert promoted == 1
        assert {_layout(row, _DPO) for row in unified} == {"conversational"}

    def test_the_rows_passed_in_are_not_changed(self):
        rows = [dict(_TEXT_ROW), dict(_CHAT_ROW)]
        unify_preference_layout(rows)
        assert rows == [_TEXT_ROW, _CHAT_ROW]

    def test_kto_rows_too_and_the_label_stays(self):
        rows = [
            {"prompt": Q, "completion": "4", "label": True},
            {"prompt": [USER2], "completion": [BAD2], "label": False},
        ]
        unified, promoted = unify_preference_layout(rows)
        assert promoted == 1
        assert unified[0] == {"prompt": [ASKED], "completion": [SAID_GOOD], "label": True}
        assert unified[1] == rows[1]

    def test_other_columns_of_a_promoted_row_are_kept(self):
        """``soup train`` keeps the source columns for some tasks (GRPO rewards)."""
        rows = [{**_TEXT_ROW, "answer": "4"}, dict(_CHAT_ROW)]
        unified, _ = unify_preference_layout(rows)
        assert unified[0] == {**_TEXT_ROW_AS_CHAT, "answer": "4"}

    def test_a_row_that_has_answers_and_no_prompt_key_is_passed_over(self):
        """Not a converted row (the converters always write ``prompt``): source
        columns kept for another task can look like this."""
        rows = [{"chosen": [GOOD], "rejected": [BAD]}, dict(_CHAT_ROW), dict(_TEXT_ROW)]
        unified, promoted = unify_preference_layout(rows)
        assert promoted == 1
        assert unified == [rows[0], _CHAT_ROW, _TEXT_ROW_AS_CHAT]

    def test_a_row_whose_answer_is_not_a_string_is_left_alone(self):
        """Only a row that IS text is rewritten; anything else is not ours to wrap."""
        odd = {"prompt": Q, "chosen": 5, "rejected": "Fish"}
        rows = [odd, dict(_CHAT_ROW)]
        unified, promoted = unify_preference_layout(rows)
        assert unified is rows and promoted == 0
        assert rows[0] == {"prompt": Q, "chosen": 5, "rejected": "Fish"}


# --------------------------------------------------------------------------
# through the loader: what `soup train` and `soup data validate` say
# --------------------------------------------------------------------------
@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(loader, "console", Console(file=buf, width=400, color_system=None))
    return buf


def _load(tmp_path, monkeypatch, rows, *, task="dpo", fmt="dpo", extra=""):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    cfg = load_config_from_string(
        f"base: x\ntask: {task}\ndata:\n  train: d.jsonl\n  format: {fmt}\n  val_split: 0\n{extra}"
    )
    return loader.load_dataset(
        loader.data_config_for_task(cfg.data, cfg.task),
        preserve_source_columns=loader.task_preserves_source_columns(cfg.task),
    )


class TestThroughTheLoader:
    def test_a_refused_row_is_named_by_index_field_and_problem(self, tmp_path, monkeypatch, out):
        bundled = {"prompt": Q, "chosen": [USER, GOOD], "rejected": [USER, BAD]}
        mixed_pair = {"prompt": Q, "chosen": [USER, GOOD], "rejected": "Fish"}
        ds = _load(tmp_path, monkeypatch, [bundled, bundled, mixed_pair, bundled])

        assert len(ds["train"]) == 3
        text = " ".join(out.getvalue().split())
        assert "Warning: 1 of 4 rows dropped: they do not convert as 'dpo'" in text
        assert (
            "First: row 2: dpo.chosen is a message list and dpo.rejected is a string: "
            "give both as strings or both as message lists"
        ) in text
        assert "soup data validate d.jsonl --format dpo" in text

    def test_soup_data_validate_gives_the_same_reason(self):
        bundled = {"prompt": Q, "chosen": [USER, GOOD], "rejected": [USER, BAD]}
        bad = {"prompt": Q, "chosen": [USER, {"role": "assistant", "content": None}],
               "rejected": [USER, BAD]}
        stats = validate_and_stats([bundled, bad], "dpo")
        assert stats["valid_rows"] == 1
        assert (
            "row 1: dpo.chosen[1].content must be a string or a list of content blocks, "
            "got NoneType" + _NO_TEXT
        ) in stats["issues"]

    def test_a_mixed_file_loads_as_conversations_and_says_so(self, tmp_path, monkeypatch, out):
        ds = _load(tmp_path, monkeypatch, [_TEXT_ROW, _CHAT_ROW, _TEXT_ROW])

        assert ds["train"] == [_TEXT_ROW_AS_CHAT, _CHAT_ROW, _TEXT_ROW_AS_CHAT]
        text = " ".join(out.getvalue().split())
        assert "2 of 3 train rows are plain-text preference rows" in text
        assert "one user turn" in text

    def test_a_file_of_one_layout_prints_nothing_about_it(self, tmp_path, monkeypatch, out):
        ds = _load(tmp_path, monkeypatch, [_TEXT_ROW, _TEXT_ROW])
        assert ds["train"] == [_TEXT_ROW, _TEXT_ROW]
        assert "plain-text preference row" not in out.getvalue()
        assert "dropped" not in out.getvalue()

    def test_each_split_is_unified_on_its_own(self, tmp_path, monkeypatch, out):
        """A text validation set beside a conversational train set is fine for
        trl (it prepares the two separately), so it is left as written."""
        cfg = load_config_from_string(
            "base: x\ntask: dpo\ndata:\n  train: d.jsonl\n  format: dpo\n  val_split: 0\n"
        )
        result = loader._finalize(
            [dict(_CHAT_ROW), dict(_TEXT_ROW)], cfg.data, val=[dict(_TEXT_ROW)]
        )
        assert result["train"] == [_CHAT_ROW, _TEXT_ROW_AS_CHAT]
        assert result["val"] == [_TEXT_ROW]
        text = " ".join(out.getvalue().split())
        assert "Note: 1 of 2 train rows is a plain-text preference row in a dataset" in text
        assert "It was rewritten as one user turn" in text
        assert " val " not in text

    def test_the_note_counts_in_the_plural(self, out):
        cfg = load_config_from_string(
            "base: x\ntask: dpo\ndata:\n  train: d.jsonl\n  format: dpo\n  val_split: 0\n"
        )
        loader._finalize([dict(_CHAT_ROW), dict(_TEXT_ROW), dict(_TEXT_ROW)], cfg.data)
        text = " ".join(out.getvalue().split())
        assert "Note: 2 of 3 train rows are plain-text preference rows in a dataset" in text
        assert "Each was rewritten as one user turn" in text

    def test_two_interleaved_files_in_different_layouts_are_unified(
        self, tmp_path, monkeypatch, out
    ):
        """Each file has one layout and the dataset has two, which is why the
        layouts are unified where the splits are final and not per file."""
        monkeypatch.chdir(tmp_path)
        for name, row in (("text.jsonl", _TEXT_ROW), ("chat.jsonl", _CHAT_ROW)):
            (tmp_path / name).write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
        cfg = load_config_from_string(
            "base: x\ntask: dpo\ndata:\n  train: [text.jsonl, chat.jsonl]\n"
            "  interleave: concat\n  format: dpo\n  val_split: 0\n"
        )
        ds = loader.load_dataset(cfg.data)
        assert ds["train"] == [_TEXT_ROW_AS_CHAT] * 2 + [_CHAT_ROW] * 2
        assert "2 of 4 train rows are plain-text preference rows" in out.getvalue()

    def test_a_split_taken_from_a_mixed_file_trains(self, tmp_path, monkeypatch, out):
        rows = [_TEXT_ROW, _CHAT_ROW] * 10
        monkeypatch.chdir(tmp_path)
        (tmp_path / "d.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        cfg = load_config_from_string(
            "base: x\ntask: dpo\ndata:\n  train: d.jsonl\n  format: dpo\n  val_split: 0.2\n"
        )
        ds = loader.load_dataset(cfg.data)
        assert len(ds["train"]) + len(ds["val"]) == 20
        for split in ("train", "val"):
            assert len({_layout(row, _DPO) for row in ds[split]}) == 1, split


# --------------------------------------------------------------------------
# every bundled file, and every row a template shows
# --------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


_BUNDLED = sorted(_EXAMPLES.glob("*.jsonl")) + sorted(_FIXTURES.glob("*.jsonl"))
_BUNDLED_IDS = [f"{path.parent.name}/{path.name}" for path in _BUNDLED]

#: task -> the lines a config of that task needs beyond base / task / data.
_DPO_FORMAT_TASKS = {
    "dpo": "",
    "ipo": "",
    "orpo": "",
    "simpo": "",
    "bco": "",
    "preference": "training:\n  preference_loss: dpo\n",
    "reward_model": "",
}


def _is_preference_file(path: Path) -> bool:
    """By the keys of its rows, not by ``detect_format``, which is under test."""
    return all({"chosen", "rejected"} <= set(row) for row in _jsonl(path))


_PREFERENCE_FILES = [path for path in _BUNDLED if _is_preference_file(path)]
_PREFERENCE_IDS = [f"{path.parent.name}/{path.name}" for path in _PREFERENCE_FILES]


class TestBundledData:
    def test_there_are_bundled_preference_files_to_check(self):
        """A glob that matches nothing would make every test below vacuous."""
        assert _PREFERENCE_IDS == [
            "data/chat_preferences.jsonl",
            "data/dpo_sample.jsonl",
            "_fixtures/chat_preferences.jsonl",
            "_fixtures/dpo_sample.jsonl",
        ]

    @pytest.mark.parametrize("path", _BUNDLED, ids=_BUNDLED_IDS)
    def test_every_bundled_file_detects_and_converts_whole(self, path):
        rows = _jsonl(path)
        fmt = detect_format(rows)
        for index, row in enumerate(rows):
            converted, reason = format_to_messages_with_reason(row, fmt)
            assert converted is not None, f"{path.name} row {index}: {reason}"

    @pytest.mark.parametrize("path", _PREFERENCE_FILES, ids=_PREFERENCE_IDS)
    def test_a_bundled_preference_file_gives_the_layout_trl_accepts(self, path):
        rows = _jsonl(path)
        assert detect_format(rows) == "dpo"
        for index, row in enumerate(rows):
            converted = format_to_messages(row, "dpo")
            assert _layout(converted, _DPO) == "conversational", f"row {index}"
            # the prompt is the row's own question, once, and the answers follow it
            assert converted["prompt"] == [{"role": "user", "content": row["prompt"]}]
            assert converted["chosen"] == row["chosen"][1:]
            assert converted["rejected"] == row["rejected"][1:]
            assert [turn["role"] for turn in converted["chosen"]] == ["assistant"]
            assert [turn["role"] for turn in converted["rejected"]] == ["assistant"]

    @pytest.mark.parametrize("task", list(_DPO_FORMAT_TASKS))
    @pytest.mark.parametrize("name", ["dpo_sample.jsonl", "chat_preferences.jsonl"])
    def test_every_task_that_reads_dpo_rows_gets_them_whole(
        self, tmp_path, monkeypatch, out, task, name
    ):
        rows = _jsonl(_EXAMPLES / name)
        ds = _load(tmp_path, monkeypatch, rows, task=task, extra=_DPO_FORMAT_TASKS[task])
        assert len(ds["train"]) == len(rows)
        assert {_layout(row, _DPO) for row in ds["train"]} == {"conversational"}
        assert "dropped" not in out.getvalue()

    @pytest.mark.parametrize("name", ["dpo_sample.jsonl", "chat_preferences.jsonl"])
    def test_auto_detection_reaches_the_same_rows(self, tmp_path, monkeypatch, out, name):
        rows = _jsonl(_EXAMPLES / name)
        explicit = _load(tmp_path, monkeypatch, rows)
        detected = _load(tmp_path, monkeypatch, rows, fmt="auto")
        assert detected == explicit


def _template_rows() -> list[tuple[str, str, dict]]:
    """``(template, task, row)`` for every JSON row a preference template shows.

    The templates point at a file the user supplies (``./data/preference_train.jsonl``),
    so the rows in their header comments are the only data they reference.
    """
    found = []
    for path in sorted(_TEMPLATES.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        task = yaml.safe_load(text).get("task")
        for line in text.splitlines():
            match = re.match(r"^#\s*(\{.*\})\s*$", line)
            if not match:
                continue
            try:
                row = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue  # one line of an example written over several
            if {"chosen", "rejected"} <= set(row) or {"completion", "label"} <= set(row):
                found.append((path.name, task, row))
    return found


class TestTheRowsTheTemplatesShow:
    def test_the_templates_show_preference_rows(self):
        shown = {name for name, _, _ in _template_rows()}
        assert {"ipo.yaml", "orpo.yaml", "simpo.yaml", "bco.yaml", "kto.yaml"} <= shown

    @pytest.mark.parametrize(
        "template, task, row", _template_rows(), ids=lambda value: str(value)[:40]
    )
    def test_each_shown_row_converts_as_text(self, template, task, row):
        fmt = detect_format([row])
        assert fmt == ("kto" if task == "kto" else "dpo")
        converted, reason = format_to_messages_with_reason(row, fmt)
        assert reason is None, reason
        assert _layout(converted, _KTO if fmt == "kto" else _DPO) == "text"


# --------------------------------------------------------------------------
# determinism: the conversion does not depend on the hash seed
# --------------------------------------------------------------------------
_CONVERT_ALL = """
import json, sys
from soup_cli.data.formats import format_to_messages_with_reason, unify_preference_layout

out = {}
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    converted = [format_to_messages_with_reason(row, "dpo") for row in rows]
    unified, promoted = unify_preference_layout([row for row, _ in converted if row])
    out[path] = [converted, unified, promoted]
sys.stdout.write(json.dumps(out))
"""


def _convert_in_a_fresh_interpreter(paths: list[Path], hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(_ROOT / "src")}
    done = subprocess.run(
        [sys.executable, "-c", _CONVERT_ALL, *map(str, paths)],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


class TestTheConversionIsTheSameUnderAnyHashSeed:
    def test_two_hash_seeds_give_byte_identical_rows(self, tmp_path):
        """trl's own decision varied with the seed (#1849); the rows it is given
        must not, key order included (the output is compared as bytes)."""
        spellings = tmp_path / "spellings.jsonl"
        spellings.write_text(
            "".join(json.dumps(row) + "\n" for row, _ in _DPO_ACCEPTED.values())
            + "".join(json.dumps(row) + "\n" for row, _ in _DPO_REFUSED.values()),
            encoding="utf-8",
        )
        paths = [*_PREFERENCE_FILES, spellings]
        first = _convert_in_a_fresh_interpreter(paths, "0")
        second = _convert_in_a_fresh_interpreter(paths, "1")
        assert first == second
        payload = json.loads(first)
        assert len(payload) == 5
        converted, unified, promoted = payload[str(spellings)]
        assert len(converted) == len(_DPO_ACCEPTED) + len(_DPO_REFUSED)
        assert [row for row, _ in converted[: len(_DPO_ACCEPTED)]] == [
            expected for _, expected in _DPO_ACCEPTED.values()
        ]
        assert all(row is None for row, _ in converted[len(_DPO_ACCEPTED):])
        assert promoted > 0, "the spellings file mixes text and conversational rows"
