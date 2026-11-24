"""τ²-bench retail 的多轮 rollout：策略（vLLM，按 token 采样）× 用户模拟器（τ² 官方 UserSimulator，后端指向我们的 vLLM）× 环境（τ² 官方工具和数据库）。

一条轨迹：
  1. 固定开场白（τ² 的 "Hi! How can I help you today?"，不算 loss）；
  2. 用户模拟器按任务设定说话；说出 ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE### 就结束；
  3. 策略一轮一轮地生成：有工具调用就在环境里执行、把结果接回去、策略接着生成；纯文字回复就交给用户模拟器；
  4. 结束后用 τ² 官方的数据库比对算结果奖励（retail 的 reward_basis 是 DB + NL_ASSERTION，训练只用 DB，NL 断言要 LLM 裁判，留给评测）。
非正常结束（策略轮数或上下文超限、模拟器出错）结果奖励记 0，和 τ² 一致。
"""
from __future__ import annotations

import json
import time
import uuid

import cs_agent  # noqa: F401  （3.10 下的 tomllib 兼容）
from cs_agent.qwen_fmt import Conv, parse_turn

GREETING = "Hi! How can I help you today?"


def agent_system_prompt(env) -> str:
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    return SYSTEM_PROMPT.format(domain_policy=env.policy, agent_instruction=AGENT_INSTRUCTION)


def load_tasks(split: str = "train") -> list:
    from tau2.domains.retail.environment import get_tasks
    return get_tasks(split)


def db_reward(task, msgs: list) -> float:
    from tau2.domains.retail.environment import get_environment
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    return float(EnvironmentEvaluator.calculate_reward(environment_constructor=get_environment, task=task, full_trajectory=msgs).reward)


def communicate_reward(task, msgs: list) -> float | None:
    """必须告知用户的信息是否都说了（子串匹配，不调 LLM）。retail 的 reward_basis 不含这一项，只做记录。"""
    if not (task.evaluation_criteria and task.evaluation_criteria.communicate_info):
        return None
    from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    return float(CommunicateEvaluator.calculate_reward(task=task, full_trajectory=msgs, scoring=CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)).reward)


def run_episode(tok, task, sample, user_llm: str, user_args: dict, max_steps: int = 60, max_ctx: int = 32000,
                gen_max_tokens: int = 4096, user_retries: int = 3, max_errors: int = 10, deadline: float = 0.0) -> dict:
    """跑一条轨迹。sample(prompt_ids, max_tokens) -> {"ids", "old_lp", "text", "finish"}（一次采一条）。"""
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.domains.retail.environment import get_environment
    from tau2.user.user_simulator import UserSimulator

    env = get_environment()
    tools = [t.openai_schema for t in env.get_tools()]
    conv = Conv(tok, agent_system_prompt(env), tools)
    user = UserSimulator(llm=user_llm, instructions=str(task.user_scenario), llm_args=dict(user_args))
    ust = user.get_init_state()
    greet = AssistantMessage(role="assistant", content=GREETING)
    conv.add_assistant_text(GREETING)
    msgs, last, term, steps, t0 = [greet], greet, None, 0, time.time()
    n_user = n_err = 0
    while term is None:
        um = None
        for k in range(user_retries):
            try:
                um, ust = user.generate_next_message(last, ust)
                break
            except Exception as e:  # noqa: BLE001  模拟器服务偶尔超时，重试
                err = repr(e)[:300]
                time.sleep(2 * (k + 1))
        if um is None:
            term = f"user_error: {err}"
            break
        if um.content and "</think>" in um.content:   # 模拟器万一还带着思考块：只留思考之后的发言
            um.content = um.content.split("</think>")[-1].strip()
        msgs.append(um)
        n_user += 1
        if UserSimulator.is_stop(um):
            term = "user_stop"
            break
        conv.add_user(um.content or "")
        while True:   # 策略连续调用工具，直到给用户发一条文字
            if steps >= max_steps:
                term = "max_steps"
                break
            if deadline and time.time() > deadline:   # 墙钟超时（GRPO 采样：一条拖太久的对话不能卡住整批），按没办完算
                term = "timeout"
                break
            prompt = conv.gen_prompt()
            if len(prompt) + gen_max_tokens > max_ctx:
                term = "max_ctx"
                break
            g = sample(prompt, gen_max_tokens)
            a, b = conv.add_policy(g["ids"], g["old_lp"], g["finish"])
            p = parse_turn(g["text"], tools)
            steps += 1
            turn = {"a": a, "b": b, "text": g["text"], "finish": g["finish"], **p, "msg_idx": len(msgs)}
            conv.turns.append(turn)
            if g["finish"] == "length":   # 一轮没写完：内容不完整，不执行，记为非正常结束
                term = "gen_length"
                break
            if p["calls"]:
                tcs = [ToolCall(id=f"c{steps}_{j}", name=c["name"], arguments=c["arguments"], requestor="assistant") for j, c in enumerate(p["calls"])]
                msgs.append(AssistantMessage(role="assistant", content=p["content"] or None, tool_calls=tcs))
                res = [env.get_response(tc) for tc in tcs]
                msgs += res
                turn["results"] = [r.content for r in res]
                turn["tool_error"] = [bool(r.error) for r in res]
                conv.add_tool_results([r.content for r in res])
                n_err += sum(turn["tool_error"])
                if n_err >= max_errors:   # 同 τ² 的 max_errors：工具报错太多次就结束
                    term = "too_many_errors"
                    break
                continue
            # 纯文字（或者工具调用格式坏了：τ² 里同样会当成一条消息发给用户）
            text = p["content"] if not p["error"] else g["text"].split("<|im_end|>")[0].strip()
            last = AssistantMessage(role="assistant", content=text or "...")
            msgs.append(last)
            break
    ok = term == "user_stop"
    rdb = db_reward(task, msgs) if ok else 0.0
    return {"task_id": task.id, "term": term, "reward": rdb, "db": rdb, "communicate": communicate_reward(task, msgs) if ok else None,
            "prompt_ids": conv.prefix_ids, "ids": conv.ids, "lmask": conv.lmask, "old_lp": conv.old_lp, "turns": conv.turns,
            "n_steps": steps, "n_user": n_user, "n_tool_errors": n_err, "n_tokens": conv.n_tokens(), "n_policy_tokens": sum(conv.lmask),
            "secs": round(time.time() - t0, 1), "messages": [m.model_dump(mode="json", exclude_none=True) for m in msgs],
            "episode_id": uuid.uuid4().hex[:12]}


def vllm_sampler(server, model: str, temperature: float):
    """train.gen_grpo.Server 包一层：一次采一条，<|im_end|> 处停（Qwen3.5 的生成配置里它就是结束符）。"""
    def f(prompt_ids: list[int], max_tokens: int) -> dict:
        return server.sample(model, prompt_ids, 1, max_tokens, temperature)[0]
    return f


def dump(ep: dict) -> str:
    return json.dumps(ep, ensure_ascii=False)
