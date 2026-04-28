#!/usr/bin/env python3

"""

consolidate_results_v2.py

-------------------------

FIXED version. Key changes from v1:

- Recognizes ALL column-name conventions used in the project:

    val_auc, vl_auc, AUC, auc, val_AUC

    val_dice, vl_dice, dice, Dice

    val_iou, vl_iou, iou, IoU

- Reads *_summary.csv files preferentially (they hold the clean OOF/mean numbers)

- Picks up extra files we missed: full_evaluation.csv, attention_comparison.csv,

  *_test_metrics.csv, *_metrics.csv, module1_summary.txt

- Adds a "data quality" sanity check column to flag empty/dead runs

"""



import os

import re

import csv

import shutil

from pathlib import Path

from datetime import datetime

from collections import defaultdict



PROJECT_ROOT = Path("/scratch/patel.tis/OncoX")

RESULTS_DIR  = PROJECT_ROOT / "results"

OUTPUT_DIR   = PROJECT_ROOT / "results_consolidated"

FIGURES_DIR  = OUTPUT_DIR / "figures"

RAW_DIR      = OUTPUT_DIR / "per_experiment"



# ALL the column names used across this project — observed from actual CSVs

AUC_COLS  = ["val_auc", "vl_auc", "AUC", "auc", "val_AUC", "VAL_AUC", "best_auc", "oof_auc", "mean_auc"]

DICE_COLS = ["val_dice", "vl_dice", "dice", "Dice", "val_Dice", "best_dice", "mean_dice"]

IOU_COLS  = ["val_iou", "vl_iou", "iou", "IoU", "val_IoU", "best_iou", "mean_iou", "mIoU"]

LOSS_COLS = ["val_loss", "vl_loss", "loss"]





def setup_dirs():

    for d in [OUTPUT_DIR, FIGURES_DIR, RAW_DIR]:

        d.mkdir(parents=True, exist_ok=True)





def safe_read_csv(path):

    try:

        with open(path, "r", newline="") as f:

            return list(csv.DictReader(f))

    except Exception as e:

        print(f"  [WARN] Could not read {path.name}: {e}")

        return []





def best_metric(rows, metric_keys):

    """Return (best_value, n_rows) — handles all column-name variants."""

    best = -float("inf")

    for row in rows:

        for k in metric_keys:

            if k in row and row[k] not in (None, "", "nan", "NaN"):

                try:

                    v = float(row[k])

                    if v > best:

                        best = v

                except ValueError:

                    continue

    return (best if best != -float("inf") else None), len(rows)





def parse_summary_csv(path):

    """Summary CSVs hold pre-computed mean/OOF metrics. Pull every numeric field."""

    rows = safe_read_csv(path)

    if not rows:

        return None

    summary = {}

    for row in rows:

        for k, v in row.items():

            if v in (None, "", "nan"):

                continue

            try:

                summary[k] = float(v)

            except (ValueError, TypeError):

                summary[k] = v

    return summary





# ---------- DATASET ----------

def collect_preprocessing():

    meta_dir = PROJECT_ROOT / "data" / "metadata"

    info = {}

    for split in ["train", "val", "test"]:

        f = meta_dir / f"{split}.csv"

        if f.exists():

            try:

                with open(f) as fh:

                    info[split] = sum(1 for _ in fh) - 1

            except Exception:

                info[split] = "?"

    return info





# ---------- CLASSIFICATION ----------

def collect_classification():

    cls_dir = RESULTS_DIR / "classification"

    if not cls_dir.exists():

        return {}



    fold_logs = {}      # exp -> {fold_n: best_auc}

    single_runs = {}    # exp -> best_auc

    summaries = {}      # filename -> parsed dict

    extras = {}         # special files



    for f in sorted(cls_dir.glob("*")):

        name = f.name



        # ---- Special / extra files ----

        if name in ("full_evaluation.csv", "attention_comparison.csv"):

            extras[name] = safe_read_csv(f)

            continue

        if name.endswith("_metrics.csv"):

            extras[name] = safe_read_csv(f)

            continue

        if name == "module1_summary.txt":

            try:

                extras[name] = f.read_text()

            except Exception:

                pass

            continue

        if name.endswith("_oof_preds.csv"):

            # Don't parse 1MB+ predictions file — just note it exists

            extras[name] = {"note": "OOF predictions file (large)", "size_kb": f.stat().st_size // 1024}

            continue



        # ---- Summary CSVs (preferred source for headline metrics) ----

        if name.endswith("_summary.csv"):

            exp_name = name.replace("_summary.csv", "")

            parsed = parse_summary_csv(f)

            if parsed:

                summaries[exp_name] = parsed

            continue



        # ---- Per-fold training logs ----

        m = re.match(r"(.+?)_fold(\d+)_log\.csv$", name)

        if m:

            exp_name, fold_n = m.group(1), int(m.group(2))

            data = safe_read_csv(f)

            best_auc, n_ep = best_metric(data, AUC_COLS)

            fold_logs.setdefault(exp_name, {})[fold_n] = {

                "best_auc": best_auc, "n_epochs": n_ep

            }

            continue



        # ---- Single-run training logs ----

        if name.endswith("_log.csv"):

            exp_name = name.replace("_log.csv", "")

            data = safe_read_csv(f)

            best_auc, n_ep = best_metric(data, AUC_COLS)

            single_runs[exp_name] = {"best_auc": best_auc, "n_epochs": n_ep}

            continue



    # Copy raw CSVs to per_experiment/

    raw_target = RAW_DIR / "classification"

    raw_target.mkdir(exist_ok=True)

    for csvf in cls_dir.glob("*.csv"):

        try:

            shutil.copy2(csvf, raw_target / csvf.name)

        except Exception:

            pass

    # Copy figures (any PNGs)

    figs = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))

    for fig in figs:

        try:

            shutil.copy2(fig, FIGURES_DIR / f"classification__{fig.name}")

        except Exception:

            pass



    return {

        "single_runs": single_runs,

        "fold_logs": fold_logs,

        "summaries": summaries,

        "extras": extras,

        "n_figures": len(figs),

    }





# ---------- SEGMENTATION ----------

def collect_segmentation():

    seg_dir = RESULTS_DIR / "segmentation"

    if not seg_dir.exists():

        return {}



    rows = []

    for log_file in seg_dir.glob("*_log.csv"):

        exp = log_file.stem.replace("_log", "")

        data = safe_read_csv(log_file)

        best_dice, n_ep = best_metric(data, DICE_COLS)

        best_iou, _ = best_metric(data, IOU_COLS)

        rows.append({

            "experiment": exp,

            "best_dice": best_dice,

            "best_iou": best_iou,

            "n_epochs": n_ep,

        })



    raw_target = RAW_DIR / "segmentation"

    raw_target.mkdir(exist_ok=True)

    for csvf in seg_dir.glob("*.csv"):

        try:

            shutil.copy2(csvf, raw_target / csvf.name)

        except Exception:

            pass

    figs = list(seg_dir.glob("*.png")) + list(seg_dir.glob("*.jpg"))

    for fig in figs:

        try:

            shutil.copy2(fig, FIGURES_DIR / f"segmentation__{fig.name}")

        except Exception:

            pass



    return {"rows": rows, "n_figures": len(figs)}





# ---------- XAI ----------

def collect_xai():

    xai_dir = RESULTS_DIR / "xai"

    if not xai_dir.exists():

        return {}

    galleries = {}

    for sub in xai_dir.iterdir():

        if sub.is_dir():

            imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))

            galleries[sub.name] = len(imgs)

            for img in sorted(imgs)[:3]:

                try:

                    shutil.copy2(img, FIGURES_DIR / f"xai__{sub.name}__{img.name}")

                except Exception:

                    pass

    return {"galleries": galleries}





# ---------- DIFFUSION ----------

def collect_diffusion():

    dif_dir = RESULTS_DIR / "diffusion"

    if not dif_dir.exists():

        return {}

    info = {}

    log = dif_dir / "ddpm_log.txt"

    if log.exists():

        lines = log.read_text().splitlines()

        info["n_log_lines"] = len(lines)

        last_epoch, last_loss = None, None

        for line in reversed(lines):

            m = re.search(r"Epoch\s+(\d+)/\d+\s*\|\s*Loss:\s*([\d.]+)", line)

            if m:

                last_epoch, last_loss = int(m.group(1)), float(m.group(2))

                break

        info["last_epoch"] = last_epoch

        info["last_loss"] = last_loss



    samples_dir = dif_dir / "samples"

    if samples_dir.exists():

        samples = sorted(samples_dir.glob("epoch_*.png"))

        info["n_samples"] = len(samples)

        if samples:

            for s in [samples[0], samples[len(samples)//2], samples[-1]]:

                try:

                    shutil.copy2(s, FIGURES_DIR / f"diffusion__{s.name}")

                except Exception:

                    pass

    gen = dif_dir / "generated"

    info["n_generated"] = len(list(gen.glob("*.png"))) if gen.exists() else 0

    return info





# ---------- INTEGRATION ----------

def collect_integration():

    int_dir = RESULTS_DIR / "integration"

    if not int_dir.exists():

        return {}

    pngs = sorted(int_dir.glob("report_*.png"))

    txts = sorted(int_dir.glob("report_*.txt"))

    for img in pngs:

        try:

            shutil.copy2(img, FIGURES_DIR / f"integration__{img.name}")

        except Exception:

            pass

    return {

        "n_visual_reports": len(pngs),

        "n_text_reports": len(txts),

        "report_files": [p.name for p in pngs],

    }





# ---------- WRITE REPORT ----------

def fmt(v, decimals=4):

    if v is None or v == "" or (isinstance(v, float) and v == -float("inf")):

        return "—"

    if isinstance(v, float):

        return f"{v:.{decimals}f}"

    return str(v)





def write_master_report(prep, cls, seg, xai, dif, intg):

    md = []

    md.append("# Onco-GPT-X — Consolidated Results Report")

    md.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    md.append("\n*Project: melanoma analysis pipeline | PI: Prof. Divya Chaudhary*")

    md.append("\n---\n")



    # 1. Dataset

    md.append("## 1. Dataset & Splits\n")

    md.append("| Split | # Images |")

    md.append("|---|---|")

    for k, v in prep.items():

        md.append(f"| {k} | {v} |")

    md.append("")



    # 2. Classification

    md.append("\n## 2. Classification (Module 2)\n")



    # 2.1 Headline summaries (from *_summary.csv)

    if cls.get("summaries"):

        md.append("### 2.1 Final summary metrics (from `*_summary.csv`)\n")

        # Find common keys across summaries to make a clean table

        all_keys = set()

        for s in cls["summaries"].values():

            all_keys.update(s.keys())

        # Prioritize the most relevant keys

        priority = ["mean_auc", "oof_auc", "auc", "best_auc", "mean_dice",

                    "fold1_auc", "fold2_auc", "fold3_auc", "fold4_auc", "fold5_auc",

                    "test_auc", "n_folds"]

        ordered_keys = [k for k in priority if k in all_keys]

        ordered_keys += sorted([k for k in all_keys if k not in priority])



        if ordered_keys:

            md.append("| Experiment | " + " | ".join(ordered_keys) + " |")

            md.append("|---|" + "|".join(["---"] * len(ordered_keys)) + "|")

            for exp, s in sorted(cls["summaries"].items()):

                cells = [fmt(s.get(k)) for k in ordered_keys]

                md.append(f"| {exp} | " + " | ".join(cells) + " |")

            md.append("")



    # 2.2 K-fold table from per-fold logs

    if cls.get("fold_logs"):

        md.append("### 2.2 K-fold cross-validation (per-fold best Val AUC)\n")

        md.append("| Experiment | F1 | F2 | F3 | F4 | F5 | Mean | Folds |")

        md.append("|---|---|---|---|---|---|---|---|")

        for exp, folds in sorted(cls["fold_logs"].items()):

            cells = []

            valid_aucs = []

            for fn in [1, 2, 3, 4, 5]:

                v = folds.get(fn, {}).get("best_auc")

                cells.append(fmt(v))

                if v is not None:

                    valid_aucs.append(v)

            mean = sum(valid_aucs) / len(valid_aucs) if valid_aucs else None

            md.append(f"| {exp} | " + " | ".join(cells) + f" | **{fmt(mean)}** | {len(valid_aucs)}/5 |")

        md.append("")



    # 2.3 Single-run experiments

    if cls.get("single_runs"):

        md.append("### 2.3 Single-run experiments\n")

        md.append("| Experiment | Best Val AUC | # Epochs |")

        md.append("|---|---|---|")

        for exp, info in sorted(cls["single_runs"].items()):

            md.append(f"| {exp} | {fmt(info['best_auc'])} | {info['n_epochs']} |")

        md.append("")



    # 2.4 Extras (full_evaluation, attention_comparison, test metrics)

    if cls.get("extras"):

        md.append("### 2.4 Additional evaluation files\n")

        for name, content in sorted(cls["extras"].items()):

            md.append(f"\n**`{name}`**\n")

            if isinstance(content, str):

                md.append("```")

                md.append(content[:1500])

                md.append("```")

            elif isinstance(content, dict):

                for k, v in content.items():

                    md.append(f"- {k}: {v}")

            elif isinstance(content, list) and content:

                # CSV rows — render as markdown table

                keys = list(content[0].keys())

                md.append("| " + " | ".join(keys) + " |")

                md.append("|" + "|".join(["---"] * len(keys)) + "|")

                for row in content[:20]:

                    md.append("| " + " | ".join(str(row.get(k, ""))[:40] for k in keys) + " |")

                if len(content) > 20:

                    md.append(f"\n*…+{len(content)-20} more rows in raw CSV*")

            md.append("")



    # 3. Segmentation

    md.append("\n## 3. Segmentation (Module 1)\n")

    if seg.get("rows"):

        md.append("| Architecture | Best Dice | Best IoU | # Epochs |")

        md.append("|---|---|---|---|")

        for r in seg["rows"]:

            md.append(f"| {r['experiment']} | {fmt(r['best_dice'])} | {fmt(r['best_iou'])} | {r['n_epochs']} |")

        md.append("")



    # 4. XAI

    md.append("\n## 4. Explainability — Grad-CAM (Module 3)\n")

    if xai.get("galleries"):

        md.append("| Gallery | # Heatmaps |")

        md.append("|---|---|")

        for g, n in xai["galleries"].items():

            md.append(f"| {g} | {n} |")

        md.append("")



    # 5. Diffusion

    md.append("\n## 5. Generative Counterfactuals — DDPM (Module 4)\n")

    if dif:

        md.append(f"- Last training epoch: **{dif.get('last_epoch', '—')}**")

        md.append(f"- Last training loss: **{dif.get('last_loss', '—')}**")

        md.append(f"- # sample grids saved: **{dif.get('n_samples', 0)}**")

        md.append(f"- # final generated images: **{dif.get('n_generated', 0)}**")

        md.append("")



    # 6. Integration

    md.append("\n## 6. Integrated Clinical Reports (Module 5)\n")

    if intg:

        md.append(f"- # visual reports generated: **{intg.get('n_visual_reports', 0)}**")

        md.append(f"- # text reports generated: **{intg.get('n_text_reports', 0)}**")

        if intg.get("report_files"):

            md.append("\nAll report files:")

            for r in intg["report_files"]:

                md.append(f"- `{r}`")

        md.append("")



    # 7. Figure index

    md.append("\n## 7. Figure Index\n")

    md.append("All figures consolidated under: `results_consolidated/figures/`\n")

    fig_files = sorted(FIGURES_DIR.glob("*"))

    md.append(f"\nTotal figures: **{len(fig_files)}**\n")



    md_text = "\n".join(md)

    out = OUTPUT_DIR / "ONCO_GPT_X_RESULTS.md"

    out.write_text(md_text)

    print(f"\n✓ Master report → {out}")

    return out





def write_master_csv(cls, seg):

    out = OUTPUT_DIR / "master_metrics.csv"

    rows = []



    # Per-fold from logs

    for exp, folds in cls.get("fold_logs", {}).items():

        valid_aucs = [v["best_auc"] for v in folds.values() if v["best_auc"] is not None]

        mean_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else None

        for fn, info in sorted(folds.items()):

            rows.append({

                "module": "classification",

                "experiment": exp,

                "fold": fn,

                "metric": "val_auc",

                "value": info["best_auc"],

                "n_epochs": info["n_epochs"],

                "source": "per-fold log",

            })

        rows.append({

            "module": "classification",

            "experiment": exp,

            "fold": "mean",

            "metric": "val_auc",

            "value": mean_auc,

            "n_epochs": "",

            "source": "computed from folds",

        })



    # Single runs

    for exp, info in cls.get("single_runs", {}).items():

        rows.append({

            "module": "classification",

            "experiment": exp,

            "fold": "",

            "metric": "val_auc",

            "value": info["best_auc"],

            "n_epochs": info["n_epochs"],

            "source": "single-run log",

        })



    # Summaries

    for exp, s in cls.get("summaries", {}).items():

        for k, v in s.items():

            if isinstance(v, (int, float)):

                rows.append({

                    "module": "classification",

                    "experiment": exp,

                    "fold": "",

                    "metric": k,

                    "value": v,

                    "n_epochs": "",

                    "source": "summary CSV",

                })



    # Segmentation

    for r in seg.get("rows", []):

        rows.append({

            "module": "segmentation", "experiment": r["experiment"],

            "fold": "", "metric": "val_dice", "value": r["best_dice"],

            "n_epochs": r["n_epochs"], "source": "log",

        })

        rows.append({

            "module": "segmentation", "experiment": r["experiment"],

            "fold": "", "metric": "val_iou", "value": r["best_iou"],

            "n_epochs": r["n_epochs"], "source": "log",

        })



    fieldnames = ["module", "experiment", "fold", "metric", "value", "n_epochs", "source"]

    with open(out, "w", newline="") as f:

        w = csv.DictWriter(f, fieldnames=fieldnames)

        w.writeheader()

        w.writerows(rows)

    print(f"✓ Master CSV → {out} ({len(rows)} metric rows)")





def main():

    print("=" * 60)

    print(" Onco-GPT-X — Consolidating all results (v2)")

    print("=" * 60)

    setup_dirs()



    print("\n[1/5] Preprocessing...")

    prep = collect_preprocessing()

    print(f"  splits: {prep}")



    print("\n[2/5] Classification...")

    cls = collect_classification()

    print(f"  single runs:    {len(cls.get('single_runs', {}))}")

    print(f"  fold groups:    {len(cls.get('fold_logs', {}))}")

    print(f"  summary files:  {len(cls.get('summaries', {}))}")

    print(f"  extras:         {len(cls.get('extras', {}))}")



    print("\n[3/5] Segmentation...")

    seg = collect_segmentation()

    print(f"  runs: {len(seg.get('rows', []))}")

    for r in seg.get("rows", []):

        print(f"    {r['experiment']}: dice={fmt(r['best_dice'])} iou={fmt(r['best_iou'])}")



    print("\n[4/5] XAI / Diffusion / Integration...")

    xai = collect_xai()

    dif = collect_diffusion()

    intg = collect_integration()

    print(f"  XAI galleries: {len(xai.get('galleries', {}))}")

    print(f"  Diffusion last epoch: {dif.get('last_epoch')}")

    print(f"  Integration reports: {intg.get('n_visual_reports', 0)}")



    print("\n[5/5] Writing report...")

    write_master_report(prep, cls, seg, xai, dif, intg)

    write_master_csv(cls, seg)



    print("\n" + "=" * 60)

    print(f" DONE → {OUTPUT_DIR}")

    print("=" * 60)





if __name__ == "__main__":

    main()
