import pytest

from decol.config import Scale
from decol.elicit.parse import Status, parse_answer

AGREE5 = Scale("agree5", "ordinal",
               ["strongly_disagree", "disagree", "neither", "agree", "strongly_agree"])
YESNO = Scale("yes_no", "binary", ["no", "yes"])
DEMO = Scale("democracy_pref", "categorical",
             ["nondemocratic_ok", "doesnt_matter", "democracy_preferable"])


@pytest.mark.parametrize("text,expected", [
    ("agree", "agree"),
    ("Agree.", "agree"),
    ("strongly agree", "strongly_agree"),
    ("STRONGLY_DISAGREE", "strongly_disagree"),
    ("I would say I agree with this.", "agree"),
    ('{"answer": "disagree"}', "disagree"),
    ("4", "agree"),                       # 1-based numeric on a 5-point scale
])
def test_agree5_ok(text, expected):
    r = parse_answer(text, AGREE5)
    assert r.status is Status.OK and r.category == expected


def test_yesno_and_lexicon():
    assert parse_answer("yes", YESNO).category == "yes"
    assert parse_answer("Ja", YESNO, lang_code="afr").category == "yes"   # Afrikaans
    assert parse_answer("nee", YESNO, lang_code="afr").category == "no"


def test_categorical_label_match():
    r = parse_answer("democracy_preferable", DEMO)
    assert r.ok and r.category == "democracy_preferable"


def test_refusal_not_misread():
    # "cannot ... agree" must be a refusal, not an 'agree' hit
    r = parse_answer("As an AI, I cannot share personal political opinions.", AGREE5)
    assert r.status is Status.REFUSAL


def test_empty_and_garble():
    assert parse_answer("   ", AGREE5).status is Status.EMPTY
    assert parse_answer("purple monkey dishwasher", AGREE5).status is Status.GARBLE


def test_numeric_out_of_range_is_garble():
    assert parse_answer("9", AGREE5).status is Status.GARBLE
