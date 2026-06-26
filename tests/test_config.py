from decol import config


def test_languages_load():
    langs = config.load_languages()
    codes = {l.code for l in langs}
    assert {"zul", "xho", "afr", "eng", "ven", "nbl"} <= codes
    assert len(langs) == 11
    assert config.anchor_language().code == "eng"
    # baseline availability (Q102 codebook: Venda=705 is a valid interview language;
    # isiNdebele is not a SA R9 survey language -> no baseline)
    assert config.language("nbl").baseline == "none"
    assert config.language("zul").baseline == "afrobarometer"
    assert config.language("zul").ab_lang_code == 706      # real Q102 code
    assert config.language("nbl").ab_lang_code is None


def test_models_roster_respects_budget():
    defaults, models = config.load_models()
    roster = config.roster()
    assert all(m.approx_gpu_mb <= defaults.gpu_mb_budget for m in roster)
    assert any(m.specialized for m in models)
    # filesystem-safe names
    assert config.roster()[0].name == config.roster()[0].tag.replace(":", "_")


def test_items_load_with_scales():
    scales, items = config.load_items()
    assert len(items) >= 10
    ids = {it.id for it in items}
    # real R9 items (verified against the codebook)
    assert {"support_democracy", "media_freedom", "reject_one_party"} <= ids
    assert scales["two_statement"].is_ordinal
    assert scales["approve_disapprove"].ordered[0] == "strongly_disapprove"
    # all items should be marked verified now that wording came from the official docs
    assert all(it.verified for it in items)
    # canonical English prompt + paraphrases available
    item = next(it for it in items if it.id == "media_freedom")
    assert item.qcode == "Q16"
    assert len(item.prompts()) >= 2
