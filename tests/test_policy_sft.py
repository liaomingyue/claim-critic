"""ragtruth.policy_sft：S1 改写按句子替换回原回答（没改的原样保留）、改写结果校验、S2 取每个 prompt 奖励最高的一条。"""
import json
import types

from ragtruth import policy_sft as P


def test_splice_replaces_only_fixed_sentences():
    t = "The cafe opens at nine. It closes at six.\nIt serves tea."
    assert P.splice(t, []) == t
    assert P.splice(t, [{"idx": 2, "new": "It closes at five."}]) == "The cafe opens at nine. It closes at five.\nIt serves tea."
    assert P.splice(t, [{"idx": 1, "new": ""}]) == "It closes at six.\nIt serves tea."           # 删句不留空白
    assert P.splice(t, [{"idx": 3, "new": ""}]) == "The cafe opens at nine. It closes at six."


def test_valid_fix():
    assert P.valid_fix({"fixes": [{"idx": 2, "new": " x. "}], "rewrite": None}, 3) == {"fixes": [{"idx": 2, "new": "x."}], "rewrite": None}
    assert P.valid_fix({"fixes": [{"idx": 4, "new": "x"}]}, 3) is None          # 句号越界
    assert P.valid_fix({"fixes": "no"}, 3) is None
    assert P.valid_fix({"fixes": [], "rewrite": " Full answer. "}, 1)["rewrite"] == "Full answer."


class Tok:
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        return "|".join(m["content"] for m in msgs) + "|A:"


def test_s2_build_picks_best_and_drops_nonpositive(tmp_path, monkeypatch):
    rows = [{"id": "p1", "task_type": "QA", "source": "S1"}, {"id": "p2", "task_type": "Summary", "source": "S2"}]
    monkeypatch.setattr(P, "rows_of", lambda path=None: rows)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tok())
    samples = [{"id": "p1", "samples": [{"text": "a", "reward": 0.2, "why": "ok"}, {"text": "b", "reward": 0.9, "why": "ok"}]},
               {"id": "p2", "samples": [{"text": "c", "reward": 0.0, "why": "length"}]}]
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in samples), encoding="utf-8")
    P.cmd_s2_build(types.SimpleNamespace(out=str(tmp_path), tokenizer="x"))
    out = [json.loads(l) for l in open(tmp_path / "sft_train.jsonl", encoding="utf-8")]
    assert len(out) == 1 and out[0]["completion"] == "b" + P.EOS and out[0]["prompt"].startswith(P.__dict__["rows_of"]()[0]["source"][:0])
    assert "S1" in out[0]["prompt"]
    m = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert m["kept"] == 1 and m["by"]["no_positive"] == 1
