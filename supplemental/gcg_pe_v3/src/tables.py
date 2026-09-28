"""Render every LaTeX table from the derived (consistent) data.
Writes <run_dir>/tables/*.tex using the same labels as the paper, with captions
that state the metric (record-level vs per-field) so tables can't be misread."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from . import analysis, evaluate
from .utils import ResultsStore, ensure_dir

DISPLAY = {"gpt2": "GPT-2-124M", "gpt2-medium": "GPT-2-355M",
           "pythia-1.4b": "Pythia-1.4B", "pythia-2.8b": "Pythia-2.8B",
           "llama2-7b": "Llama-2-7B", "gpt2-xl": "GPT-2-XL"}
FIELD_DISPLAY = {"name": "Name", "email": "Email", "phone": "Phone",
                 "address": "Address", "ssn": "SSN", "credit_card": "Credit Card",
                 "dob": "DOB"}
ORDER = ["gpt2", "gpt2-medium", "pythia-1.4b", "pythia-2.8b", "llama2-7b"]


def _disp(name: str) -> str:
    return DISPLAY.get(name, name)


def _w(run_dir: str, fname: str, body: str) -> None:
    path = os.path.join(ensure_dir(os.path.join(run_dir, "tables")), fname)
    with open(path, "w") as f:
        f.write(body)
    print(f"[tables] wrote {path}")


def t_main(store, run_dir, record_fields) -> None:
    res = evaluate.main_results(store, record_fields)
    rows = ""
    for m in [x for x in ORDER if x in res] + [x for x in res if x not in ORDER]:
        r = res[m]
        rows += (f"{_disp(m)} & {r['baseline_mean']:.1f} $\\pm$ "
                 f"{r['baseline_std']:.1f} & {r['opt_mean']:.1f} $\\pm$ "
                 f"{r['opt_std']:.1f} & {r['ratio']:.2f}$\\times$ \\\\\n")
    rf = ", ".join(record_fields)
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
        "Model & Baseline & Optimized & Ratio \\\\\n\\midrule\n" + rows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Extraction success rates (\\%) across model scales. Success is "
        "\\emph{record-level}: a target counts as extracted only when the full PII "
        f"record ({rf}) is recovered together. Mean $\\pm$ std across seeds. "
        "Per-field rates (Tables~\\ref{tab:frequency} and~\\ref{tab:field_level}) "
        "are higher because each field is scored independently.}\n"
        "\\label{tab:main_results}\n\\end{table}\n"
    )
    _w(run_dir, "tab_main_results.tex", body)


def t_frequency(store, run_dir, model) -> None:
    rows = ""
    for r in evaluate.frequency_table(store, model):
        lab = f"{r['frequency']} mention" + ("s" if r["frequency"] != 1 else "")
        rows += (f"{lab} & {r['n']} & {r['baseline']:.1f}\\% & "
                 f"{r['optimized']:.1f}\\% & +{r['delta']:.1f}\\% \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n"
        "Frequency & N & Baseline & Optimized & $\\Delta$ \\\\\n\\midrule\n" + rows +
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Per-field extraction success (\\%) by training frequency "
        f"({_disp(model)}). Each PII field attempt is scored independently, so "
        "rates are higher than the record-level metric in "
        "Table~\\ref{tab:main_results}.}\n\\label{tab:frequency}\n\\end{table}\n"
    )
    _w(run_dir, "tab_frequency.tex", body)


def t_field(store, run_dir, model, fields) -> None:
    rows = ""
    for r in evaluate.field_table(store, model, fields):
        rows += (f"{FIELD_DISPLAY.get(r['field'], r['field'])} & "
                 f"{r['baseline']:.1f}\\% & {r['optimized']:.1f}\\% \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lcc}\n\\toprule\n"
        "Field Type & Baseline & Optimized \\\\\n\\midrule\n" + rows +
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Per-field extraction success by PII field type "
        f"({_disp(model)}). Each field is evaluated independently as a substring "
        "match against model output.}\n\\label{tab:field_level}\n\\end{table}\n"
    )
    _w(run_dir, "tab_field_level.tex", body)


def t_transfer(store, run_dir, record_fields, pairs) -> None:
    rows = ""
    for r in evaluate.transfer_table(store, record_fields, pairs):
        rows += (f"{_disp(r['source'])} $\\rightarrow$ {_disp(r['target'])} & "
                 f"{r['direct']:.1f}\\% & {r['transfer']:.1f}\\% & "
                 f"{r['retention']:.1f}\\% \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
        "Source $\\rightarrow$ Target & Direct & Transfer & Retention \\\\\n"
        "\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Cross-model transferability (record-level metric). "
        "Retention = transfer success / direct success.}\n"
        "\\label{tab:transfer}\n\\end{table}\n"
    )
    _w(run_dir, "tab_transfer.tex", body)


def t_validation(store, run_dir) -> None:
    rows_data = evaluate.validation_table(store)
    if not rows_data:
        return
    rows = ""
    for r in rows_data:
        sep = "\\midrule\n" if r["category"] == "Overall" else ""
        rows += (sep + f"{r['category']} & {r['n']} & {r['baseline']:.1f}\\% & "
                 f"{r['optimized']:.1f}\\% & "
                 f"+{r['optimized'] - r['baseline']:.1f}\\% \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n"
        "Category & N & Baseline & Optimized & $\\Delta$ \\\\\n\\midrule\n" + rows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Validation on known-memorized sequences (GPT-2-XL). "
        "``Overall'' is computed from raw counts.}\n"
        "\\label{tab:validation}\n\\end{table}\n"
    )
    _w(run_dir, "tab_validation.tex", body)


def t_category(store, run_dir, model) -> None:
    rows = ""
    for r in evaluate.category_table(store, model):
        rows += (f"{r['category']} & {r['baseline']:.1f}\\% & "
                 f"{r['optimized']:.1f}\\% & +{r['delta']:.1f}\\% \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
        "Category & Baseline & Optimized & $\\Delta$ \\\\\n\\midrule\n" + rows +
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Extraction by field category ({_disp(model)}; per-field "
        "metric). Categories group the structured PII fields of our synthetic "
        "corpus.}\n\\label{tab:category}\n\\end{table}\n"
    )
    _w(run_dir, "tab_category.tex", body)


def t_convergence(store, run_dir, model) -> None:
    data = evaluate.convergence_table(store, model)
    if not data:
        return
    head = " & ".join(str(d["iteration"]) for d in data)
    vals = " & ".join(f"{d['success']:.1f}\\%" for d in data)
    body = (
        "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{l" +
        "c" * len(data) + "}\n\\toprule\n"
        f"Iterations & {head} \\\\\n\\midrule\n"
        f"Success Rate & {vals} \\\\\n\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Extraction success rate by GCG iteration ({_disp(model)}).}}\n"
        "\\label{tab:convergence}\n\\end{table}\n"
    )
    _w(run_dir, "tab_convergence.tex", body)


def t_linguistic(store, run_dir, model) -> None:
    res = analysis.content_predictor_table(store, model)
    if not res:
        return
    feats = list(res["baseline"].keys())
    rows = ""
    for f in feats:
        b, g = res["baseline"][f], res["optimized"][f]
        rows += (f"{f.replace('_', ' ').title()} & {b['beta']:+.2f} & "
                 f"{b['p']:.3f} & {g['beta']:+.2f} & {g['p']:.3f} \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n"
        "Feature & $\\beta_{\\text{base}}$ & $p$ & $\\beta_{\\text{opt}}$ & $p$ "
        "\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Linguistic predictors of extraction success ({_disp(model)}). "
        "Standardized logistic-regression coefficients, controlling for training "
        "frequency.}\n\\label{tab:linguistic}\n\\end{table}\n"
    )
    _w(run_dir, "tab_linguistic.tex", body)


def t_prompts(store, run_dir, model) -> None:
    data = analysis.prompt_property_table(store, model)
    if not data:
        return
    rows = ""
    for r in data:
        rows += (f"{r['feature'].replace('_', ' ').title()} & "
                 f"{r['baseline']:.2f} & {r['optimized']:.2f} & "
                 f"{r['cohens_d']:.2f} \\\\\n")
    body = (
        "\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
        "Feature & Baseline & Optimized & Effect Size \\\\\n\\midrule\n" + rows +
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Comparison of baseline and optimized prompts ({_disp(model)}). "
        "Effect sizes are Cohen's d.}\n\\label{tab:prompts}\n\\end{table}\n"
    )
    _w(run_dir, "tab_prompts.tex", body)


def generate_all(cfg: Dict[str, Any]) -> None:
    run_dir = cfg["output_dir"]
    store = ResultsStore(run_dir)
    primary = cfg["models"][0]["name"]
    fields = cfg["extract"]["fields"]
    record_fields = cfg["extract"]["record_fields"]
    pairs = cfg.get("transfer", {}).get("pairs", [])

    t_main(store, run_dir, record_fields)
    t_frequency(store, run_dir, primary)
    t_field(store, run_dir, primary, fields)
    t_category(store, run_dir, primary)
    t_convergence(store, run_dir, primary)
    if pairs:
        t_transfer(store, run_dir, record_fields, pairs)
    t_validation(store, run_dir)
    t_linguistic(store, run_dir, primary)
    t_prompts(store, run_dir, primary)
    print(f"[tables] all tables in {os.path.join(run_dir, 'tables')}")
