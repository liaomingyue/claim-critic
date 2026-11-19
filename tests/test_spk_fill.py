import random
from types import SimpleNamespace as NS

import pytest

spacy = pytest.importorskip("spacy")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from ragtruth.spk_fill import Src, do_swap, do_tag, speakers

SOURCE = ("The storm hit the coastal town early on Monday and flooded dozens of homes near the harbour. "
          "Mayor Anna Keller said the town would open two shelters for displaced families. "
          "Power was restored to most of the town by Tuesday evening after crews repaired the damaged substation. "
          "\"We have never seen water rise this fast,\" said Tom Reyes, a fisherman who lost his boat.")
A = NS(min_overlap=3, window=4)


def test_speakers_need_reported_content():
    assert [sp.text for _, sp in speakers(nlp("Mayor Anna Keller said the town would open two shelters."))] == ["Mayor Anna Keller"]
    assert speakers(nlp("The accident claimed the life of the driver.")) == []


def test_swap_puts_another_named_speaker_in():
    S = Src(nlp(SOURCE))
    s = do_swap(S, nlp("Mayor Anna Keller said the town would open two shelters for displaced families."), random.Random(0), A)
    assert s and s["form"] == "swap" and "Reyes" in s["sentence"] and "Keller" not in s["sentence"] and s["sentence"].endswith("open two shelters for displaced families.")


def test_tag_only_on_narration():
    S = Src(nlp(SOURCE))
    s = do_tag(S, nlp("Power was restored to most of the town by Tuesday evening after crews repaired the substation."), random.Random(0), A)
    assert s and s["form"] == "tag" and ("Keller" in s["sentence"] or "Reyes" in s["sentence"]) and "Power was restored" in s["sentence"] or "power was restored" in s["sentence"]
    assert do_tag(S, nlp("The town would open two shelters for displaced families after the storm."), random.Random(0), A) is None
