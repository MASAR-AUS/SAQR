"""Architecture B under blur: does the deployed layer stop blurred crops, and are the ones it lets through still named
correctly and covered by the CP set? Report only; nothing is chosen here.

The d1_gc7_fix test half (2,752 crops, 87 families), blurred synthetically at increasing strength:
  gaussian  sigma 0.5, 1, 1.5, 2, 3, 4 px   (PIL GaussianBlur; pixels of the 224 px crop)
  motion    5, 9, 15, 21 px streak          (archA_gates.motion: a line kernel at a random angle per crop)
Gaussian 2 / 4 px and motion 9 / 21 px are architecture A's "s1" / "s2" corruptions (results/archA/step2/).
Every blurred crop goes through gc7 (8-view features, MLP heads, kNN vote) and architecture B as deployed
(calibration_gc7.json: OOD gate, temperature, CP set, visor; no quality gate). For reference only, architecture A's
legacy quality gate (calibration.json, alpha 0.05) is run on the same pixels; B does not use it.
Real blur: the untouched defect crops split into fifths by their own sharpness (variance of the Laplacian).

From 1_main_model_resnet18/ (~15 min on an M-series Mac):
  python enhanced_res/arch_a/experiments/archB_blur.py [--figure blur_examples.png]
Writes enhanced_res/arch_a/results/archB/blur/README.md and results.json. The optional figure shows D1 crops, so it
is written only where --figure points (the repo is public; D1 images are not committed).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score

import archB_common as BC
from archA_gates import motion, quality_stats
from archB_common import R

OUT = BC.HERE.parent / "results" / "archB" / "blur"
LEVELS = ([("clean", None)] + [(f"gaussian {s:g} px", ("g", s)) for s in (0.5, 1, 1.5, 2, 3, 4)]
          + [(f"motion {k} px", ("m", k)) for k in (5, 9, 15, 21)])
FIG_LEVELS = ["clean", "gaussian 1 px", "gaussian 2 px", "gaussian 4 px", "motion 9 px", "motion 21 px"]


def blur(im, how, rng):
    if how is None:
        return im
    kind, s = how
    return im.filter(ImageFilter.GaussianBlur(s)) if kind == "g" else motion(im, int(s), rng)


def visor(d):
    return {R.NOTHING: "(nothing shown)", R.REAIM: "RE-AIM"}.get(d.action) or d.text()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", help="PNG of example crops at each blur level (D1 images; keep it out of git)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    B = R.ArchA(R.CALIBRATIONS["gc7"])
    A = R.ArchA(R.CALIBRATIONS["v2"])                      # only for its legacy quality gate
    va, cal, tst = BC.halves(B.cal["data"]["split"])
    cid = {c: i for i, c in enumerate(B.classes)}
    y, g = tst.cls.map(cid).to_numpy(), tst.src_group.to_numpy()
    nd = B.nd
    defect, clean_skin = y != nd, y == nd
    base = [Image.open(p).convert("RGB") for p in tst.crop_path]

    rows, per = [], {}
    for name, how in LEVELS:
        rng = np.random.default_rng(0)
        ims = [blur(im, how, rng) for im in base]
        F, P, D = BC.outputs("gc7", None, ims)
        dec = B.decide(P, knn_dist=D, feats=F)
        act = np.array([d.action for d in dec])
        p, S = B.sets(P)
        top, n_set = p.argmax(1), S.sum(1)
        hit, right = S[np.arange(len(y)), y], p.argmax(1) == y
        passed = act != R.REAIM
        sharp = quality_stats(ims).sharpness.to_numpy()
        if how is None:
            z = np.load(BC.C.CACHE["gc7"], allow_pickle=True)
            assert list(z["crop_path"][len(va) + len(cal):]) == list(tst.crop_path)
            drift = float(np.abs(z["probs"][len(va) + len(cal):] - P).max())
            sharp0 = sharp
        alo, ahi = BC.family_ci(right, g)
        clo, chi = BC.family_ci(hit[passed], g[passed])
        rows.append({
            "level": name,
            "median sharpness": float(np.median(sharp)),
            "sharpness left (median of blurred / clean, per crop)": float(np.median(sharp / np.maximum(sharp0, 1e-6))),
            "legacy quality gate would stop": float(np.mean(A.quality(ims)[0] <= A.alpha_q)),
            "re-aim (OOD gate)": float(np.mean(act == R.REAIM)),
            "top-class accuracy": float(right.mean()), "accuracy 95% family interval": [alo, ahi],
            "7-class macro-F1": float(f1_score(y, top, average="macro", labels=list(range(len(B.classes))))),
            "CP coverage of passed crops": float(hit[passed].mean()), "coverage 95% family interval": [clo, chi],
            "mean set size, passed": float(n_set[passed].mean()),
            "single class, passed": float((n_set[passed] == 1).mean()),
            "shown one class and right": float(np.mean((act == R.SHOW) & hit)),
            "single wrong defect class shown without warning": float(np.mean((act == R.SHOW) & ~hit)),
            "flagged for the inspector (ambiguous / possible / re-aim)": float(np.mean(np.isin(act, (R.AMBIGUOUS, R.POSSIBLE, R.UNCLEAR, R.REAIM)))),
            "silent miss (defect shown nothing)": float(np.mean(act[defect] == R.NOTHING)),
            "visor speaks on no-defect crops": float(np.mean(act[clean_skin] != R.NOTHING)),
        })
        per[name] = dict(dec=dec, act=act, hit=hit, right=right, sharp=sharp, ims=ims if name in FIG_LEVELS else None)
        print(f"{name:16s} acc {right.mean():.3f}  re-aim {np.mean(act == R.REAIM):.3f}  cov {hit[passed].mean():.3f}  "
              f"size {n_set[passed].mean():.2f}  silent {np.mean(act[defect] == R.NOTHING):.3f}", flush=True)

    # real blur: untouched defect crops by their own sharpness
    c0 = per["clean"]
    q = pd.qcut(pd.Series(sharp0[defect]), 5, labels=False).to_numpy()
    real = []
    for k in range(5):
        m = np.flatnonzero(defect)[q == k]
        pa = m[c0["act"][m] != R.REAIM]
        real.append({"fifth": k + 1, "crops": int(len(m)), "families": int(len(set(g[m]))),
                     "sharpness range": [float(sharp0[m].min()), float(sharp0[m].max())],
                     "top-class accuracy": float(c0["right"][m].mean()), "re-aim": float(np.mean(c0["act"][m] == R.REAIM)),
                     "CP coverage of passed": float(c0["hit"][pa].mean()),
                     "silent miss": float(np.mean(c0["act"][m] == R.NOTHING))})

    (OUT / "results.json").write_text(json.dumps({"crops": int(len(y)), "families": int(len(set(g))),
                                                  "defect_crops": int(defect.sum()), "no_defect_crops": int(clean_skin.sum()),
                                                  "clean_vs_cache_max_prob_diff": drift, "levels": rows,
                                                  "real_blur_defect_fifths": real}, indent=1))
    pct = lambda v: f"{v * 100:.1f}%"                                                     # noqa: E731
    md = ["# Architecture B under blur", "",
          f"The d1_gc7_fix test half ({len(y):,} crops: {defect.sum():,} defect, {clean_skin.sum()} no-defect; "
          f"{len(set(g))} families), blurred synthetically, through gc7 and architecture B as deployed (OOD α "
          f"{B.alpha_ood}, T {B.T:.6f}, CP α {B.alpha}; no quality gate). Report only: nothing is chosen here. Blur is in "
          "pixels of the 224 px crop. Gaussian 2 / 4 px and motion 9 / 21 px are architecture A's s1 / s2 "
          "([`../../archA/step2/`](../../archA/step2/README.md)). \"Legacy quality gate\" = architecture A's (α 0.05) on "
          "the same pixels, for reference; B does not run it. The training augmentation has no blur. Clean pass vs "
          f"calibrate.py cache: max probability difference {drift:.1e}.", "",
          "## What B does with blurred crops", "",
          "| blur | sharpness left | legacy quality gate would stop | re-aim (OOD) | top-class accuracy (95% family "
          "interval) | 7-class macro-F1 | CP coverage of passed (95% family interval) | mean set size | single class "
          "| shown one class, right | single wrong class, no warning | silent miss | visor speaks on no-defect |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['level']} | {pct(r['sharpness left (median of blurred / clean, per crop)'])} | "
                  f"{pct(r['legacy quality gate would stop'])} | {pct(r['re-aim (OOD gate)'])} | "
                  f"{pct(r['top-class accuracy'])} ({pct(r['accuracy 95% family interval'][0])}–{pct(r['accuracy 95% family interval'][1])}) | "
                  f"{r['7-class macro-F1']:.3f} | {pct(r['CP coverage of passed crops'])} "
                  f"({pct(r['coverage 95% family interval'][0])}–{pct(r['coverage 95% family interval'][1])}) | "
                  f"{r['mean set size, passed']:.2f} | {pct(r['single class, passed'])} | {pct(r['shown one class and right'])} | "
                  f"{pct(r['single wrong defect class shown without warning'])} | {pct(r['silent miss (defect shown nothing)'])} | "
                  f"{pct(r['visor speaks on no-defect crops'])} |")
    md += ["", "Sharpness left = median over crops of (variance of the Laplacian after blur) / (before). CP coverage "
           "is only guaranteed for crops like the calibration crops; under blur it is a measurement.", "",
           "## Real blur: untouched defect crops by their own sharpness", "",
           "Fifths of the test-half defect crops by variance of the Laplacian (1 = least sharp). Low sharpness also "
           "means plain surfaces, not only blur.", "",
           "| fifth | crops (families) | sharpness | top-class accuracy | re-aim | CP coverage of passed | silent miss |",
           "|---:|---:|---|---:|---:|---:|---:|"]
    for r in real:
        md.append(f"| {r['fifth']} | {r['crops']} ({r['families']}) | {r['sharpness range'][0]:.0f}–{r['sharpness range'][1]:.0f} | "
                  f"{pct(r['top-class accuracy'])} | {pct(r['re-aim'])} | {pct(r['CP coverage of passed'])} | {pct(r['silent miss'])} |")
    md += ["", "Run from 1_main_model_resnet18/: `python enhanced_res/arch_a/experiments/archB_blur.py`."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")

    if a.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ex = []
        for cls in ("crack", "dent"):                     # a crop named right when clean, of median sharpness
            m = np.flatnonzero((y == cid[cls]) & (c0["act"] == R.SHOW) & c0["hit"])
            ex.append(m[np.argsort(sharp0[m])[len(m) // 2]])
        soft = np.flatnonzero(defect)[np.argsort(sharp0[defect])[:len(FIG_LEVELS)]]   # the least sharp untouched
        fig, ax = plt.subplots(3, len(FIG_LEVELS), figsize=(3.1 * len(FIG_LEVELS), 10.2))
        for r, i in enumerate(ex):
            for c, lv in enumerate(FIG_LEVELS):
                d = per[lv]
                ax[r, c].imshow(d["ims"][i])
                ax[r, c].set_title(f"{lv} · sharpness {d['sharp'][i]:.0f}\n{visor(d['dec'][i])[:34]}", fontsize=8)
        for c, i in enumerate(soft):
            ax[2, c].imshow(base[i])
            ax[2, c].set_title(f"real, {tst.cls[i]} · sharpness {sharp0[i]:.0f}\n{visor(c0['dec'][i])[:34]}", fontsize=8)
        for x in ax.ravel():
            x.axis("off")
        fig.suptitle(f"Rows 1–2: a {tst.cls[ex[0]]} and a {tst.cls[ex[1]]} test crop blurred; row 3: the least sharp "
                     "untouched test defect crops. Titles: architecture B's visor output", fontsize=10)
        fig.tight_layout()
        fig.savefig(a.figure, dpi=110)
        print(f"figure: {a.figure}")


if __name__ == "__main__":
    main()
