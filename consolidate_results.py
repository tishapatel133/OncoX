#!/usr/bin/env python3

"""

consolidate_results.py

----------------------

Walks /scratch/patel.tis/OncoX/results/ and produces a SINGLE paper-ready

summary report (markdown + PDF) with all metrics, tables, and figures from

every module of the Onco-GPT-X pipeline.



USAGE (on the Explorer cluster):

    cd /scratch/patel.tis/OncoX/

    python consolidate_results.py



OUTPUT:

    /scratch/patel.tis/OncoX/results_consolidated/

    ├── ONCO_GPT_X_RESULTS.md          ← paper-ready master report

    ├── ONCO_GPT_X_RESULTS.pdf         ← same, as PDF (if pandoc available)

    ├── master_metrics.csv             ← all metrics in one CSV

    ├── figures/                       ← every figure copied here, renamed

    └── per_experiment/                ← raw CSVs, organized

"""



import os

import re

import csv

import glob

import json

import shutil

import argparse

from pathlib import Path

from datetime import datetime

from collections import defaultdict



# ----- CONFIG -----

PROJECT_ROOT = Path("/scratch/patel.tis/OncoX")

RESULTS_DIR  = PROJECT_ROOT / "results"

LOGS_DIR     = PROJECT_ROOT / "models" / "logs"

CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints"

OUTPUT_DIR   = PROJECT_ROOT / "results_consolidated"

FIGURES_DIR  = OUTPUT_DIR / "figures"

RAW_DIR      = OUTPUT_DIR / "per_experiment"

# -------------------





def setup_output_dirs():

    for d in [OUTPUT_DIR, FIGURES_DIR, RAW_DIR]:

        d.mkdir(parents=True, exist_ok=True)





def safe_read_csv(path):

    """Read a CSV defensively — returns list of dict rows, or [] if unreadable."""

    try:

        with open(path, "r", newline="") as f:

            reader = csv.DictReader(f)

            return list(reader)

    except Exception as e:

        print(f"  [WARN] Could not read {path}: {e}")

        return []





def best_row(rows, metric_keys):

    """Find the row with the best (max) value for any of the metric keys."""

    best = None

    best_val = -float("inf")

    best_metric = None

    for row in rows:

        for k in metric_keys:

            if k in row and row[k] not in (None, "", "nan"):

                try:

                    v = float(row[k])

                    if v > best_val:

                        best_val = v

                        best = row

                        best_metric = k

                except ValueError:

                    continue

    return best, best_val, best_metric





# ---------- MODULE 1: PREPROCESSING ----------

def collect_preprocessing():

    """Pull dataset split stats from data/metadata/."""

    meta_dir = PROJECT_ROOT / "data" / "metadata"

    info = {}

    for split in ["train", "val", "test"]:

        f = meta_dir / f"{split}.csv"

        if f.exists():

            try:

                with open(f) as fh:

                    n = sum(1 for _ in fh) - 1

                info[split] = n

            except Exception:

                info[split] = "?"

    return info





# ---------- MODULE 2: CLASSIFICATION ----------

def collect_classification():

    """Walk results/classification/ for all fold logs, ablation summaries."""

    cls_dir = RESULTS_DIR / "classification"

    if not cls_dir.exists():

        return {}



    rows = []  # one row per (experiment, fold)

    

    # Pattern 1: per-fold logs like efficientnet_b3_kfold5_meta0_384_fold1_log.csv

    fold_logs = sorted(cls_dir.glob("*fold*_log.csv"))

    fold_groups = defaultdict(list)

    for fl in fold_logs:

        # Extract experiment prefix and fold number

        m = re.match(r"(.+?)_fold(\d+)_log\.csv", fl.name)

        if not m:

            continue

        exp_name, fold_n = m.group(1), int(m.group(2))

        data = safe_read_csv(fl)

        if not data:

            continue

        best, best_val, _ = best_row(data, ["val_auc", "val_AUC", "auc", "AUC"])

        rows.append({

            "module": "classification",

            "experiment": exp_name,

            "fold": fold_n,

            "best_val_auc": best_val if best_val != -float("inf") else None,

            "n_epochs": len(data),

            "source_file": fl.name,

        })

        fold_groups[exp_name].append(best_val)



    # Pattern 2: single-run logs (CBAM, SwinV2, PVTv2, etc.)

    single_logs = [

        f for f in cls_dir.glob("*_log.csv")

        if "fold" not in f.name

    ]

    for sl in single_logs:

        exp_name = sl.stem.replace("_log", "")

        data = safe_read_csv(sl)

        if not data:

            continue

        best, best_val, _ = best_row(data, ["val_auc", "val_AUC", "auc", "AUC"])

        rows.append({

            "module": "classification",

            "experiment": exp_name,

            "fold": None,

            "best_val_auc": best_val if best_val != -float("inf") else None,

            "n_epochs": len(data),

            "source_file": sl.name,

        })



    # Pattern 3: OOF/summary CSVs

    summary_files = list(cls_dir.glob("*summary*.csv")) + list(cls_dir.glob("*oof*.csv"))

    summaries = {}

    for sf in summary_files:

        data = safe_read_csv(sf)

        summaries[sf.name] = data



    # Pattern 4: ablation experiment results (Exp 1-6)

    # Look for two_stage, ohem, smote, ddpm_aug, gan_aug etc.

    ablation_results = {}

    for keyword in ["two_stage", "ohem", "smote", "oversample", "ddpm_aug", "gan_aug"]:

        files = list(cls_dir.glob(f"*{keyword}*.csv"))

        for f in files:

            data = safe_read_csv(f)

            if data:

                best, best_val, _ = best_row(data, ["val_auc", "auc", "AUC"])

                ablation_results[keyword] = {

                    "best_auc": best_val if best_val != -float("inf") else None,

                    "n_epochs": len(data),

                    "file": f.name,

                }



    # Copy figures

    figs = []

    for ext in ["png", "jpg", "pdf"]:

        figs.extend(cls_dir.glob(f"*.{ext}"))

        figs.extend(cls_dir.glob(f"**/*.{ext}"))

    for fig in figs:

        target = FIGURES_DIR / f"classification__{fig.name}"

        try:

            shutil.copy2(fig, target)

        except Exception as e:

            print(f"  [WARN] Could not copy {fig}: {e}")



    # Copy raw CSVs

    raw_target = RAW_DIR / "classification"

    raw_target.mkdir(exist_ok=True)

    for csvf in cls_dir.glob("*.csv"):

        try:

            shutil.copy2(csvf, raw_target / csvf.name)

        except Exception:

            pass



    return {

        "rows": rows,

        "fold_groups": {k: v for k, v in fold_groups.items()},

        "summaries": summaries,

        "ablation": ablation_results,

        "n_figures": len(figs),

    }





# ---------- MODULE 1 (per professor): SEGMENTATION ----------

def collect_segmentation():

    seg_dir = RESULTS_DIR / "segmentation"

    if not seg_dir.exists():

        return {}



    rows = []

    for log_file in seg_dir.glob("*_log.csv"):

        exp_name = log_file.stem.replace("_log", "")

        data = safe_read_csv(log_file)

        if not data:

            continue

        best, best_dice, _ = best_row(data, ["val_dice", "dice", "Dice", "val_Dice"])

        _, best_iou, _ = best_row(data, ["val_iou", "iou", "IoU", "mIoU"])

        rows.append({

            "module": "segmentation",

            "experiment": exp_name,

            "best_dice": best_dice if best_dice != -float("inf") else None,

            "best_iou": best_iou if best_iou != -float("inf") else None,

            "n_epochs": len(data),

            "source_file": log_file.name,

        })



    # Copy figures

    figs = list(seg_dir.glob("*.png")) + list(seg_dir.glob("*.jpg"))

    for fig in figs:

        try:

            shutil.copy2(fig, FIGURES_DIR / f"segmentation__{fig.name}")

        except Exception:

            pass



    raw_target = RAW_DIR / "segmentation"

    raw_target.mkdir(exist_ok=True)

    for csvf in seg_dir.glob("*.csv"):

        try:

            shutil.copy2(csvf, raw_target / csvf.name)

        except Exception:

            pass



    return {"rows": rows, "n_figures": len(figs)}





# ---------- MODULE 3: XAI ----------

def collect_xai():

    xai_dir = RESULTS_DIR / "xai"

    if not xai_dir.exists():

        return {}



    sample_galleries = {}

    for subdir in xai_dir.iterdir():

        if subdir.is_dir():

            imgs = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg"))

            sample_galleries[subdir.name] = len(imgs)

            # Copy first 3 as representative samples

            for i, img in enumerate(sorted(imgs)[:3]):

                try:

                    shutil.copy2(img, FIGURES_DIR / f"xai__{subdir.name}__{img.name}")

                except Exception:

                    pass



    return {"galleries": sample_galleries}





# ---------- MODULE 4: DIFFUSION ----------

def collect_diffusion():

    dif_dir = RESULTS_DIR / "diffusion"

    if not dif_dir.exists():

        return {}



    info = {}

    log = dif_dir / "ddpm_log.txt"

    if log.exists():

        with open(log) as f:

            lines = f.readlines()

        info["n_log_lines"] = len(lines)

        info["last_lines"] = lines[-5:] if lines else []



        # Parse last epoch + loss

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

        # Copy first, middle, last as progression

        if samples:

            for s in [samples[0], samples[len(samples)//2], samples[-1]]:

                try:

                    shutil.copy2(s, FIGURES_DIR / f"diffusion__{s.name}")

                except Exception:

                    pass



    generated_dir = dif_dir / "generated"

    if generated_dir.exists():

        info["n_generated"] = len(list(generated_dir.glob("*.png")))



    return info





# ---------- MODULE 5: INTEGRATION ----------

def collect_integration():

    int_dir = RESULTS_DIR / "integration"

    if not int_dir.exists():

        return {}



    reports_png = sorted(int_dir.glob("report_*.png"))

    reports_txt = sorted(int_dir.glob("report_*.txt"))



    # Copy ALL integration figures (these are the key visual outputs)

    for img in reports_png:

        try:

            shutil.copy2(img, FIGURES_DIR / f"integration__{img.name}")

        except Exception:

            pass



    # Read text reports for summary

    text_summaries = []

    for txt in reports_txt:

        try:

            with open(txt) as f:

                text_summaries.append({"file": txt.name, "content": f.read()[:500]})

        except Exception:

            pass



    return {

        "n_visual_reports": len(reports_png),

        "n_text_reports": len(reports_txt),

        "report_files": [r.name for r in reports_png],

        "text_summaries": text_summaries,

    }





# ---------- WRITE MASTER REPORT ----------

def write_master_report(prep, cls, seg, xai, dif, intg):

    md = []

    md.append(f"# Onco-GPT-X — Consolidated Results Report")

    md.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    md.append(f"\n*Project: melanoma analysis pipeline | PI: Prof. Divya Chaudhary*")

    md.append("\n---\n")



    # Dataset stats

    md.append("## 1. Dataset & Splits\n")

    if prep:

        md.append("| Split | # Images |")

        md.append("|---|---|")

        for k, v in prep.items():

            md.append(f"| {k} | {v} |")

    else:

        md.append("*No metadata files found.*")

    md.append("")



    # Classification (Module 2)

    md.append("\n## 2. Classification (Module 2)\n")

    if cls.get("rows"):

        md.append("### 2.1 Single-run experiments\n")

        md.append("| Experiment | Best Val AUC | # Epochs |")

        md.append("|---|---|---|")

        for r in [x for x in cls["rows"] if x["fold"] is None]:

            auc = f"{r['best_val_auc']:.4f}" if r['best_val_auc'] is not None else "—"

            md.append(f"| {r['experiment']} | {auc} | {r['n_epochs']} |")

        md.append("")



        # K-fold per-fold

        if cls.get("fold_groups"):

            md.append("### 2.2 K-fold cross-validation (per-fold)\n")

            md.append("| Experiment | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |")

            md.append("|---|---|---|---|---|---|---|")

            for exp, vals in cls["fold_groups"].items():

                vals = [v for v in vals if v != -float("inf")]

                cells = [f"{v:.4f}" for v in vals] + ["—"] * (5 - len(vals))

                mean = sum(vals)/len(vals) if vals else 0

                md.append(f"| {exp} | " + " | ".join(cells[:5]) + f" | **{mean:.4f}** |")

            md.append("")



        # Ablation

        if cls.get("ablation"):

            md.append("### 2.3 Ablation: class-balancing techniques\n")

            md.append("| Technique | Best Val AUC | # Epochs | Source |")

            md.append("|---|---|---|---|")

            for k, v in cls["ablation"].items():

                auc = f"{v['best_auc']:.4f}" if v['best_auc'] is not None else "—"

                md.append(f"| {k} | {auc} | {v['n_epochs']} | `{v['file']}` |")

            md.append("")



    # Segmentation (Module 1)

    md.append("\n## 3. Segmentation (Module 1)\n")

    if seg.get("rows"):

        md.append("| Architecture | Best Dice | Best IoU | # Epochs |")

        md.append("|---|---|---|---|")

        for r in seg["rows"]:

            dice = f"{r['best_dice']:.4f}" if r['best_dice'] is not None else "—"

            iou = f"{r['best_iou']:.4f}" if r['best_iou'] is not None else "—"

            md.append(f"| {r['experiment']} | {dice} | {iou} | {r['n_epochs']} |")

    else:

        md.append("*No segmentation logs found.*")

    md.append("")



    # XAI (Module 3)

    md.append("\n## 4. Explainability — Grad-CAM (Module 3)\n")

    if xai.get("galleries"):

        md.append("| Gallery | # Heatmaps |")

        md.append("|---|---|")

        for g, n in xai["galleries"].items():

            md.append(f"| {g} | {n} |")

    md.append("")



    # Diffusion (Module 4)

    md.append("\n## 5. Generative Counterfactuals — DDPM (Module 4)\n")

    if dif:

        md.append(f"- Last training epoch: **{dif.get('last_epoch', '—')}**")

        md.append(f"- Last training loss: **{dif.get('last_loss', '—')}**")

        md.append(f"- # sample grids saved: **{dif.get('n_samples', 0)}**")

        md.append(f"- # final generated images: **{dif.get('n_generated', 0)}**")

    md.append("")



    # Integration (Module 5)

    md.append("\n## 6. Integrated Clinical Reports (Module 5)\n")

    if intg:

        md.append(f"- # visual reports generated: **{intg.get('n_visual_reports', 0)}**")

        md.append(f"- # text reports generated: **{intg.get('n_text_reports', 0)}**")

        if intg.get("report_files"):

            md.append("\n**Sample reports:**")

            for r in intg["report_files"][:5]:

                md.append(f"- `{r}`")

    md.append("")



    # Figure index

    md.append("\n## 7. Figure Index\n")

    md.append(f"All figures consolidated under: `results_consolidated/figures/`\n")

    fig_files = sorted(FIGURES_DIR.glob("*"))

    md.append("| Module Prefix | Filename |")

    md.append("|---|---|")

    for f in fig_files[:50]:  # cap at 50 to keep report readable

        prefix = f.name.split("__")[0] if "__" in f.name else "?"

        md.append(f"| {prefix} | `{f.name}` |")

    if len(fig_files) > 50:

        md.append(f"\n*…and {len(fig_files)-50} more in the figures/ folder.*")

    md.append("")



    md.append("\n---\n")

    md.append(f"\n*Total figures collected: {len(fig_files)}*")



    md_text = "\n".join(md)

    out = OUTPUT_DIR / "ONCO_GPT_X_RESULTS.md"

    with open(out, "w") as f:

        f.write(md_text)

    print(f"\n✓ Master report written to: {out}")

    return out





def write_master_csv(cls, seg):

    """Flatten all metrics into a single CSV."""

    out = OUTPUT_DIR / "master_metrics.csv"

    fieldnames = ["module", "experiment", "fold", "best_val_auc",

                  "best_dice", "best_iou", "n_epochs", "source_file"]

    with open(out, "w", newline="") as f:

        w = csv.DictWriter(f, fieldnames=fieldnames)

        w.writeheader()

        for r in cls.get("rows", []):

            w.writerow({k: r.get(k) for k in fieldnames})

        for r in seg.get("rows", []):

            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"✓ Master CSV written to: {out}")





def try_pdf_conversion(md_path):

    """Attempt MD→PDF conversion via pandoc if available."""

    pdf_path = md_path.with_suffix(".pdf")

    rc = os.system(f"pandoc {md_path} -o {pdf_path} 2>/dev/null")

    if rc == 0 and pdf_path.exists():

        print(f"✓ PDF written to: {pdf_path}")

    else:

        print("  (pandoc not available — skip PDF; markdown is ready)")





def main():

    print("=" * 60)

    print(" Onco-GPT-X — Consolidating all results")

    print("=" * 60)

    setup_output_dirs()



    print("\n[1/5] Preprocessing / dataset splits...")

    prep = collect_preprocessing()

    print(f"  Found splits: {prep}")



    print("\n[2/5] Classification (Module 2)...")

    cls = collect_classification()

    print(f"  Found {len(cls.get('rows', []))} classification runs")

    print(f"  Found {len(cls.get('ablation', {}))} ablation experiments")



    print("\n[3/5] Segmentation (Module 1)...")

    seg = collect_segmentation()

    print(f"  Found {len(seg.get('rows', []))} segmentation runs")



    print("\n[4/5] XAI / Diffusion / Integration (Modules 3-5)...")

    xai = collect_xai()

    dif = collect_diffusion()

    intg = collect_integration()

    print(f"  XAI galleries: {len(xai.get('galleries', {}))}")

    print(f"  Diffusion last epoch: {dif.get('last_epoch')}")

    print(f"  Integration reports: {intg.get('n_visual_reports', 0)}")



    print("\n[5/5] Writing consolidated report...")

    md_path = write_master_report(prep, cls, seg, xai, dif, intg)

    write_master_csv(cls, seg)

    try_pdf_conversion(md_path)



    print("\n" + "=" * 60)

    print(" DONE.")

    print(f" Output dir: {OUTPUT_DIR}")

    print("=" * 60)





if __name__ == "__main__":

    main()
