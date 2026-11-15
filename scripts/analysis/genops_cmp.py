import json, sys, collections, numpy as np
from ragtruth.error_ops import OPS
from train.gen_reward import edit_sim
def load(tag):
    P = {r["id"]: r for r in map(json.loads, open(f"outputs/rt/core/samples_ops_{tag}_rm4.jsonl"))}
    A = []
    for l in open(f"outputs/rt/core/samples_ops_{tag}.jsonl", encoding="utf-8"):
        x = json.loads(l); x["ok"] = x["why"] == "ok"
        if x["ok"]: x.update(P[x["id"]]); x["sim"] = edit_sim(x["original"], x["sentence"])
        x["g3"] = x["ok"] and x["p_err"] >= .5 and x["p_nat"] >= .5 and x["p_op"] >= .5
        for k in ("source", "context", "sentences_orig"): x.pop(k, None)
        A.append(x)
    return A
pc = lambda xs, f: 100 * np.mean([bool(f(x)) for x in xs]) if xs else float("nan")
D = {t: load(t) for t in sys.argv[1:]}
print("%-13s %5s %4s | %4s %4s %4s | %5s | sim  | critic(of passing): >=.9  .5-.9  .1-.5  <.1 | pass&critic<.5 (of all)  pass&.1-.5" % ("model", "n", "ok", "real", "nat", "op", "all3"))
for t, A in D.items():
    S = [x for x in A if x["ok"]]; g = [x for x in A if x["g3"]]
    print("%-13s %5d %3.0f%% | %3.0f%% %3.0f%% %3.0f%% | %4.1f%% | %.2f |              %3.0f%%  %3.0f%%  %3.0f%%  %3.0f%% |      %4.1f%%            %4.1f%%" % (t, len(A), pc(A, lambda x: x["ok"]), pc(S, lambda x: x["p_err"] >= .5), pc(S, lambda x: x["p_nat"] >= .5), pc(S, lambda x: x["p_op"] >= .5), pc(A, lambda x: x["g3"]), np.median([x["sim"] for x in S]),
          pc(g, lambda x: x["p_edit"] >= .9), pc(g, lambda x: .5 <= x["p_edit"] < .9), pc(g, lambda x: .1 <= x["p_edit"] < .5), pc(g, lambda x: x["p_edit"] < .1), 100 * sum(x["p_edit"] < .5 for x in g) / len(A), 100 * sum(.1 <= x["p_edit"] < .5 for x in g) / len(A)))
print("\nper op all3%:  " + "  ".join("%-12s" % t[:12] for t in D))
for op in OPS:
    print("%-18s " % op + "  ".join("%-12.1f" % pc([x for x in A if x["op"] == op], lambda x: x["g3"]) for A in D.values()))
print("\nper layer all3%:")
for l in ("value", "binding", "qualifier", "between"):
    print("%-18s " % l + "  ".join("%-12.1f" % pc([x for x in A if OPS[x["op"]]["layer"] == l], lambda x: x["g3"]) for A in D.values()))
print("\ndiscriminator d1 (P real-looking), of ok samples:")
for t, A in D.items():
    S = [x for x in A if x["ok"] and x.get("p_disc") is not None]; g = [x for x in S if x["g3"]]
    print("%-13s mean %.3f  >=0.5 %4.1f%% | among passing: mean %.3f >=0.5 %4.1f%% | pass & disc>=0.5 (of all) %4.1f%% | pass & disc>=0.5 & critic<0.5 (of all) %4.1f%%" % (t, np.mean([x["p_disc"] for x in S]), pc(S, lambda x: x["p_disc"] >= .5), np.mean([x["p_disc"] for x in g]), pc(g, lambda x: x["p_disc"] >= .5), 100 * sum(x["p_disc"] >= .5 for x in g) / len(A), 100 * sum(x["p_disc"] >= .5 and x["p_edit"] < .5 for x in g) / len(A)))
