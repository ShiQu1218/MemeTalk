from __future__ import annotations

import json
from typing import Any


def extract_json_object(payload: str, array_field: str | None = None) -> dict[str, Any]:
    text = _strip_code_fence(payload)
    if not text:
        return {}

    direct_error: Exception | None = None
    try:
        return _coerce_json_value(json.loads(text), array_field=array_field)
    except (json.JSONDecodeError, ValueError) as exc:
        direct_error = exc

    for candidate in _iter_json_fragments(text):
        try:
            return _coerce_json_value(json.loads(candidate), array_field=array_field)
        except (json.JSONDecodeError, ValueError):
            continue

    raise ValueError("Provider response does not contain a JSON object.") from direct_error


def _strip_code_fence(payload: str) -> str:
    text = payload.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _coerce_json_value(value: Any, array_field: str | None = None) -> dict[str, Any]:
    current = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        nested = current.strip()
        if not nested:
            break
        current = json.loads(nested)

    if isinstance(current, dict):
        return current
    if isinstance(current, list) and array_field is not None:
        return {array_field: current}
    raise ValueError("Provider response does not contain a JSON object.")


def _iter_json_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for start in [index for index, char in enumerate(text) if char in "{["]:
        fragment = _extract_balanced_fragment(text, start)
        if fragment is not None:
            fragments.append(fragment)
    return fragments


def _extract_balanced_fragment(text: str, start: int) -> str | None:
    closing = {"{": "}", "[": "]"}
    opener = text[start]
    if opener not in closing:
        return None
    stack: list[str] = [closing[opener]]
    in_string = False
    escaped = False

    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in closing:
            stack.append(closing[char])
            continue
        if char not in "}]":
            continue
        if not stack or char != stack[-1]:
            return None
        stack.pop()
        if not stack:
            return text[start : index + 1]

    return None
