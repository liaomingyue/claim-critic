"""可视化应用（Gradio）。只做布局和事件绑定，逻辑都在 app/stats.py。

用法：python -m app.main [--config configs/data.yaml] [--eval_dir outputs/eval] [--port 7860]
"""
from __future__ import annotations

import argparse

from app import stats
from common.config import load_config
from common.schema import ERROR_TYPES

EXAMPLE_TEXT = ("In adults with type 2 diabetes, metformin reduced HbA1c by 1.1% over 24 weeks [ref-1]. "
                "Metformin therefore prevents diabetes in all patients [ref-1].")
EXAMPLE_REFS = "ref-1: In adults aged 18 to 65 with type 2 diabetes, metformin reduced HbA1c by 1.1% over 24 weeks."
EXAMPLE_OUTPUT = ('<think>原文只覆盖已确诊的成年患者。</think>\n<verdict>hallucination</verdict>\n'
                  '<evidence>ref-1: "In adults aged 18 to 65 with type 2 diabetes"</evidence>\n'
                  '<type>missing_condition</type>\n<reason>原文限定为成年糖尿病患者，claim 推广到所有患者。</reason>')


def _df(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def build(cfg: dict, eval_dir: str, max_batch: int = 3):
    import gradio as gr

    files = stats.sample_files(cfg, max_batch)

    # ---------- Tab 1 事件 ----------
    def run_judge(text, refs_text, base_url, model):
        from common.llm import LLM, LLMConfig
        from infer.judge import judge_review

        refs = stats.parse_refs(refs_text)
        llm = LLM(LLMConfig(model=model, base_url=base_url, temperature=0.0, max_tokens=1024))
        results = judge_review(llm, text, refs)
        return _df(stats.judge_table(results)), stats.render_claims_html(results)

    def run_reward(output, label, type_, refs_text):
        return _df([stats.reward_breakdown(output, label, type_, stats.parse_refs(refs_text))])

    # ---------- Tab 2 事件 ----------
    def refresh_data():
        hard_rows = []
        for b in range(1, max_batch + 1):
            hard_rows += stats.load_kept(cfg["mine_hard"]["out"].format(batch=b)) or []
        meta = stats.hard_meta_distribution(hard_rows)
        dist = stats.hard_distribution(cfg, max_batch)
        for d in dist:
            d["batch/round"] = f"b{d['batch']}/r{d['round']}"
        audit = stats.audit_summary(cfg, max_batch)
        audit_html = _audit_html(audit)
        return (_df(stats.pipeline_counts(cfg, max_batch)),
                _df(dist) if dist else _df([{"batch/round": stats.NOT_READY, "type": "", "n": 0}]),
                _df(meta["detect_rate"]), _df(meta["rewrites"]), audit_html)

    def browse(name, type_, round_):
        rows = stats.load_kept(files[name])
        if rows is None:
            return _df([{"提示": f"{files[name]} {stats.NOT_READY}"}])
        rnd = None if round_ in ("", "全部", None) else int(round_)
        return _df(stats.filter_samples(rows, None if type_ == "全部" else type_, rnd))

    def refresh_eval(baseline):
        results = stats.scan_metrics(eval_dir)
        models = [r["model"] for r in results]
        if not results:
            return _df([{"提示": f"{eval_dir}/*/metrics.json {stats.NOT_READY}"}]), gr.update(choices=[])
        table = stats.metrics_table(results, baseline if baseline in models else None)
        return _df(table), gr.update(choices=models, value=baseline if baseline in models else None)

    with gr.Blocks(title="claim-critic") as demo:
        gr.Markdown("# claim-critic 可视化")
        with gr.Tab("在线判定 Demo"):
            with gr.Row():
                text = gr.Textbox(label="综述正文（句末带 [ref-N]）", lines=8, value=EXAMPLE_TEXT)
                refs_text = gr.Textbox(label="参考文献（JSON，或每行 ref-1: 文本）", lines=8, value=EXAMPLE_REFS)
            with gr.Row():
                base_url = gr.Textbox(label="critic base_url", value="http://localhost:8000/v1")
                model = gr.Textbox(label="model", value="critic-3")
                go = gr.Button("逐条判定", variant="primary")
            table = gr.Dataframe(label="判定结果", wrap=True)
            claims_html = gr.HTML()
            go.click(run_judge, [text, refs_text, base_url, model], [table, claims_html])

            with gr.Accordion("Reward 计算器（不需要模型）", open=False):
                with gr.Row():
                    out_box = gr.Textbox(label="模型输出", lines=8, value=EXAMPLE_OUTPUT)
                    rw_refs = gr.Textbox(label="参考文献", lines=8, value=EXAMPLE_REFS)
                with gr.Row():
                    gt_label = gr.Radio(["hallucination", "faithful"], value="hallucination", label="真值 label")
                    gt_type = gr.Dropdown(list(ERROR_TYPES), value=ERROR_TYPES[0], label="真值 type")
                    calc = gr.Button("计算")
                rw_table = gr.Dataframe(label="得分（总分 = 1.0×label + 0.3×format + 0.3×loc）")
                calc.click(run_reward, [out_box, gt_label, gt_type, rw_refs], rw_table)

        with gr.Tab("数据与评测看板"):
            refresh = gr.Button("刷新")
            counts = gr.Dataframe(label="各阶段样本量")
            dist_plot = gr.BarPlot(x="batch/round", y="n", color="type", label="难样本按批次/轮次/类型分布")
            with gr.Row():
                rate_plot = gr.BarPlot(x="bucket", y="n", label="critic 检出率分布（难样本）")
                rew_plot = gr.BarPlot(x="rewrites", y="n", label="重写次数分布")
            audit_box = gr.HTML(label="人工抽检")
            refresh.click(refresh_data, None, [counts, dist_plot, rate_plot, rew_plot, audit_box])

            gr.Markdown("### 样本浏览")
            with gr.Row():
                fsel = gr.Dropdown(list(files), value="positives", label="文件")
                tsel = gr.Dropdown(["全部", *ERROR_TYPES, "none"], value="全部", label="type")
                rsel = gr.Dropdown(["全部", "0", "1", "2", "3"], value="全部", label="round")
                bshow = gr.Button("显示")
            samples = gr.Dataframe(wrap=True)
            bshow.click(browse, [fsel, tsel, rsel], samples)

            gr.Markdown("### 评测对比")
            with gr.Row():
                base_sel = gr.Dropdown([], label="基线（算 F1 绝对差 / 相对提升）")
                beval = gr.Button("刷新评测")
            eval_table = gr.Dataframe(wrap=True)
            beval.click(refresh_eval, base_sel, [eval_table, base_sel])
            base_sel.change(refresh_eval, base_sel, [eval_table, base_sel])
    return demo


def _audit_html(rows: list[dict]) -> str:
    if not rows:
        return f"<p>抽检文件{stats.NOT_READY}</p>"
    head = "<tr><th>文件</th><th>指标</th><th>已检/总数</th><th>比例</th><th>阈值</th></tr>"
    body = []
    for r in rows:
        ratio = "—" if r["ratio"] is None else f"{r['ratio']:.1%}"
        thr = "—" if r["threshold"] is None else f"{r['threshold']:.0%}"
        style = " style='color:#c0392b;font-weight:600'" if r["over"] else ""
        body.append(f"<tr{style}><td>{r['file']}</td><td>{r['metric']}</td>"
                    f"<td>{r['checked']}/{r['n']}</td><td>{ratio}</td><td>{thr}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--eval_dir", default="outputs/eval")
    ap.add_argument("--max_batch", type=int, default=3)
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(load_config(args.config), args.eval_dir, args.max_batch).launch(server_port=args.port)


if __name__ == "__main__":
    main()
