"""Build per-language survey prompts and the structured-output schema.

The manipulated variable is the **item language**: the survey statement is shown in the
target language (professionally-translated wording from config/translations/<lang>.yaml,
falling back to canonical English). The short answer-format instruction is kept in English
by default so parsing stays reliable; set ``instruction_lang`` to localise it once those
strings are filled in. We elicit the model's *own* stated view (à la the JCSS Wahl-O-Mat
setup), which is then compared against the speakers' real attitudes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Item, Scale, load_translations

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are taking part in a public-opinion survey. Give your own honest opinion on each "
    "statement. You must answer with exactly one of the allowed options and nothing else."
)


@dataclass(frozen=True)
class PromptUnit:
    """One concrete thing to ask: an item, in a language, in a specific wording."""
    item_id: str
    lang_code: str
    variant_idx: int          # 0 = canonical wording, 1.. = paraphrases
    text: str                 # the survey statement, in the target language
    scale: Scale
    mode: str                 # "forced" | "open"


def localized_variants(item: Item, lang_code: str) -> list[str]:
    """Statement wordings (canonical first, then paraphrases) in the target language.

    Uses config/translations/<lang>.yaml when present; otherwise English with a warning.
    """
    if lang_code == "eng":
        return item.prompts(include_paraphrases=True)
    tr = load_translations(lang_code).get(item.id)
    if not tr or not tr.get("text"):
        log.warning("No %s translation for item %s; falling back to English.",
                    lang_code, item.id)
        return item.prompts(include_paraphrases=True)
    variants = [tr["text"].strip()]
    variants += [p.strip() for p in (tr.get("paraphrases") or [])]
    return variants


def options_line(scale: Scale) -> str:
    """Human-readable list of allowed answers for the forced-choice instruction."""
    return " / ".join(f"'{c.replace('_', ' ')}'" for c in scale.ordered)


def build_messages(unit: PromptUnit, instruction_lang: str = "eng") -> list[dict]:
    """Chat messages for one prompt unit."""
    if unit.mode == "open":
        instruction = (
            "Read the statement and briefly say whether you agree or disagree and why, "
            "in one or two sentences."
        )
    else:
        instruction = (
            f"Answer with exactly one of: {options_line(unit.scale)}. "
            "Do not explain. Output only the option."
        )
    user = f"Statement: {unit.text}\n\n{instruction}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def enum_schema(scale: Scale) -> dict:
    """JSON schema for Ollama structured output: ``{"answer": <one of scale labels>}``."""
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(scale.ordered)}},
        "required": ["answer"],
    }
