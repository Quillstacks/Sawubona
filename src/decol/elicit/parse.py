"""Parse a model's response into a canonical scale category.

The primary elicitation path uses Ollama structured output (a JSON object whose ``answer``
is constrained to the scale's labels), so most parsing is trivial. This module also
handles free-text (open-ended cross-check, or a model that ignored the schema): canonical
labels, an English + multilingual yes/no/agree lexicon, numeric answers, and refusal
detection. Below-scale / off-topic output is reported as ``garble`` rather than guessed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from ..config import Scale, load_answer_lexicon


class Status(str, Enum):
    OK = "ok"
    REFUSAL = "refusal"
    GARBLE = "garble"
    EMPTY = "empty"


@dataclass(frozen=True)
class ParseResult:
    category: str | None
    status: Status
    raw: str

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


_REFUSAL_PATTERNS = [
    r"\bas an ai\b", r"\bas a language model\b", r"\bi('m| am) (just )?an ai\b",
    r"\bi (can('|no)t|cannot|am unable to|am not able to)\b.*\b(answer|provide|share|give|express)\b",
    r"\bi (do|don'?t) ?not have (personal )?(opinions|views|beliefs|feelings)\b",
    r"\bi (don'?t|do not) hold (political )?(opinions|views)\b",
    r"\bi prefer not to\b", r"\bit('s| is) not appropriate\b",
    r"\bi remain neutral\b.*\bcannot\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# Strip surrounding quotes/markdown/punctuation but keep internal spaces and hyphens.
_CLEAN_RE = re.compile(r"[^\w\s\-']", re.UNICODE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _CLEAN_RE.sub(" ", text.lower())).strip()


def _maybe_json_answer(text: str) -> str:
    """If the model emitted a JSON object (structured output), pull ``answer``."""
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return text
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return text
    if isinstance(obj, dict):
        for key in ("answer", "response", "choice", "label"):
            if key in obj and isinstance(obj[key], (str, int)):
                return str(obj[key])
    return text


def _lexicon_for(lang_code: str | None) -> list[tuple[str, str]]:
    """Surface→canonical-token pairs, longest surface first (multiword before single)."""
    lex = load_answer_lexicon()
    pairs: dict[str, str] = {}
    pairs.update({str(k).lower(): str(v) for k, v in (lex.get("canonical") or {}).items()})
    if lang_code:
        by_lang = (lex.get("by_language") or {}).get(lang_code) or {}
        pairs.update({str(k).lower(): str(v) for k, v in by_lang.items()})
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def parse_answer(text: str, scale: Scale, lang_code: str | None = None) -> ParseResult:
    """Map ``text`` onto one of ``scale.ordered``."""
    raw = text if text is not None else ""
    if not raw.strip():
        return ParseResult(None, Status.EMPTY, raw)

    payload = _maybe_json_answer(raw)
    norm = _normalize(payload)
    if not norm:
        return ParseResult(None, Status.EMPTY, raw)

    valid = set(scale.ordered)
    # canonical labels can appear with underscores or spaces
    label_surface = {cat.replace("_", " "): cat for cat in scale.ordered}

    # 1. exact canonical label (whole answer or a clear token within it)
    if norm in valid:
        return ParseResult(norm, Status.OK, raw)
    if norm in label_surface:
        return ParseResult(label_surface[norm], Status.OK, raw)

    # 2. refusal (check before fuzzy matching so "i cannot agree" isn't read as agree)
    if _REFUSAL_RE.search(payload):
        return ParseResult(None, Status.REFUSAL, raw)

    # 3. numeric answer on an ordered scale (1-based, as Afrobarometer codes)
    m = re.fullmatch(r"(\d+)", norm)
    if m and scale.is_ordinal:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(scale.ordered):
            return ParseResult(scale.ordered[idx], Status.OK, raw)

    # 4. lexicon surface words -> canonical token, if that token is in this scale
    for surface, token in _lexicon_for(lang_code):
        if token not in valid:
            continue
        if re.search(rf"(?<![\w-]){re.escape(surface)}(?![\w-])", norm):
            return ParseResult(token, Status.OK, raw)

    # 5. canonical label appearing as a phrase inside a longer sentence
    for surface, cat in sorted(label_surface.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<![\w-]){re.escape(surface)}(?![\w-])", norm):
            return ParseResult(cat, Status.OK, raw)

    return ParseResult(None, Status.GARBLE, raw)
