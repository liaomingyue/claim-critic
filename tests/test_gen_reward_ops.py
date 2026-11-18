"""按占比给的稀有奖励（OpsRewarder.share_post）：只用 group_post，不连打分模型。"""
from train.gen_reward_ops import OpsRewarder


def sample(node: str, op: str = "swap_term", c_op: str | None = None, reward: float = 1.0) -> dict:
    return {"why": "ok", "gate": True, "type": op, "c_op": c_op or op, "c_node": node, "original": "a b c", "sentence": "a b d", "reward": reward}


def test_group_share_rewards_the_rare_node_and_not_the_wrong_operation():
    r = OpsRewarder(None, {}, cls=2, share_w=(1.0, 0.0))
    g = [sample("replace/content"), sample("replace/content"), sample("replace/content"), sample("replace/name"), sample("add/content", c_op="add_detail")]
    r.group_post(g)
    assert [round(x["grp_div"], 2) for x in g] == [0.25, 0.25, 0.25, 0.75, 0.0]   # 只在做对操作的 4 条里算：1 − 3/4、1 − 1/4；做成别的操作的不参与
    assert abs(sum(x["grp_div"] for x in g[:4]) / 4 - (1 - 0.75 ** 2 - 0.25 ** 2)) < 1e-9   # 平均值 = 这一组节点分布的多样性
    assert g[3]["reward"] > g[0]["reward"] > g[4]["reward"] == 1.0
    one = [sample("replace/content")]
    r.group_post(one)
    assert one[0]["grp_div"] == 0.0 and one[0]["reward"] == 1.0


def test_global_share_starts_from_the_base_distribution_and_accumulates():
    base = {"swap_term": {"share": {"replace/content": 0.8, "replace/name": 0.2}}}
    r = OpsRewarder(None, {}, cls=2, node_base=base, share_w=(0.0, 1.0), share_n0=100.0)
    g = [sample("replace/content"), sample("replace/name"), sample("move/structure")]
    r.group_post(g)
    assert [round(x["all_div"], 2) for x in g] == [0.2, 0.8, 1.0]
    assert r.share_cnt["swap_term"] == {"replace/content": 81.0, "replace/name": 21.0, "move/structure": 1.0}
    for _ in range(100):   # 一直产出同一个节点，它的占比上去，奖励下来
        g = [sample("move/structure")]
        r.group_post(g)
    assert g[0]["all_div"] < 0.55
    r2 = OpsRewarder(None, {}, cls=2, node_base=base, share_w=(0.0, 1.0))
    r2.load_state_dict(r.state_dict())
    assert r2.share_cnt == r.share_cnt


def test_share_terms_off_by_default():
    r = OpsRewarder(None, {}, cls=2)
    g = [sample("replace/content"), sample("replace/name")]
    r.group_post(g)
    assert all("share_mult" not in x and x["reward"] == 1.0 for x in g)


def test_global_share_only_covers_operations_without_a_real_target():
    tg = {"swap_term": {"n": 100, "node": {"replace/content": 1.0}, "word": {}}, "add_hedge": {"n": 3, "node": {}, "word": {}}}
    base = {op: {"share": {"replace/content": 0.5, "weaken/modal": 0.5}} for op in ("swap_term", "add_hedge")}
    r = OpsRewarder(None, {}, cls=2, node_base=base, real_target=tg, share_w=(0.0, 1.0))
    g = [sample("replace/content")]
    r.group_post(g)
    assert g[0]["all_div"] == 0.0 and "swap_term" not in r.share_cnt
    g = [sample("weaken/modal", op="add_hedge")]
    r.group_post(g)
    assert round(g[0]["all_div"], 2) == 0.5
