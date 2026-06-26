from decol import config
from decol.elicit import prompts, runner


def test_enum_schema_matches_scale():
    scales, _ = config.load_items()
    schema = prompts.enum_schema(scales["approve_disapprove"])
    assert schema["properties"]["answer"]["enum"] == scales["approve_disapprove"].ordered
    assert schema["required"] == ["answer"]


def test_localized_variants_english_fallback():
    _, items = config.load_items()
    item = items[0]
    # a language with no translation file falls back to English wording
    eng = prompts.localized_variants(item, "eng")
    fallback = prompts.localized_variants(item, "zul")
    assert eng[0] == item.english.strip()
    assert fallback[0] == item.english.strip()   # warned + fell back


def test_build_messages_forced_lists_options():
    _, items = config.load_items()
    item = items[0]
    unit = prompts.PromptUnit(item.id, "eng", 0, item.english, item.scale, "forced")
    msgs = prompts.build_messages(unit)
    assert msgs[0]["role"] == "system"
    assert "Statement:" in msgs[1]["content"]
    assert item.scale.ordered[0].replace("_", " ") in msgs[1]["content"]


def test_seed_is_deterministic():
    a = runner._seed("qwen2.5:0.5b", "zul", "media_freedom", "forced", 0, 3)
    b = runner._seed("qwen2.5:0.5b", "zul", "media_freedom", "forced", 0, 3)
    c = runner._seed("qwen2.5:0.5b", "zul", "media_freedom", "forced", 0, 4)
    assert a == b and a != c
    assert 0 <= a < 2 ** 31


def test_units_for_language_counts():
    _, items = config.load_items()
    units = list(runner._units_for_language("eng", include_paraphrases=False, k_open=0))
    # one forced unit per item when paraphrases off and no open mode
    assert len(units) == len(items)
    assert all(u.mode == "forced" for u in units)
