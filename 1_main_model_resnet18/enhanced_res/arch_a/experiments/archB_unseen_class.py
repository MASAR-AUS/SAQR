"""Architecture B on a damage class D1 does not have: D5's Rupture. Does the deployed layer show a confident wrong class,
hedge (AMBIGUOUS / POSSIBLE), stay silent, or refuse (re-aim)? Report only; nothing is chosen here.

Crops: D5 Rupture crops from D5 photos that are not in D1 and one photo per D5 near-duplicate group, by the same rule as
the D5 outside test (enhanced_res/external_d5.py, results/d5/d5_images.csv: kept). Reference: D5's own dents and
fastener damage (the classes D1 has, archA_d5.d5_crops) and the d1_gc7_fix test-half defect crops, through the same
layer. Every rupture answer is wrong by construction (there is no rupture class), so the question is how it is wrong.
Also: how D1 labels the D5 rupture photos it contains (the matched D1 photo's dominant class in d1_grouped/groups.csv).

From 1_main_model_resnet18/ (a few minutes; model outputs cached under data/work/enhanced_res/archB/):
  python enhanced_res/arch_a/experiments/archB_unseen_class.py [--examples examples.json]
Writes enhanced_res/arch_a/results/archB/unseen_class/README.md and results.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import archB_common as BC
from archB_common import R

OUT = BC.HERE.parent / "results" / "archB" / "unseen_class"
HEDGE = (R.AMBIGUOUS, R.POSSIBLE, R.UNCLEAR)


def summarise(B, P, D, F, groups, y=None):
    dec = B.decide(P, knn_dist=D, feats=F)
    act = np.array([d.action for d in dec])
    p, S = B.sets(P)
    n_set = S.sum(1)
    show = act == R.SHOW
    conf = np.array([d.scores[0] if d.action == R.SHOW else np.nan for d in dec])
    shown_cls = pd.Series([d.classes[0] for d in dec if d.action == R.SHOW]).value_counts()
    out = {
        "crops": int(len(act)), "groups": int(len(set(groups))),
        "re-aim": float(np.mean(act == R.REAIM)), "nothing shown": float(np.mean(act == R.NOTHING)),
        "ambiguous": float(np.mean(act == R.AMBIGUOUS)), "possible": float(np.mean(act == R.POSSIBLE)),
        "one class shown": float(show.mean()),
        "one class shown at 80%+": float(np.mean(show & (np.nan_to_num(conf) >= 0.8))),
        "mean confidence when one class is shown": float(np.nanmean(conf)) if show.any() else None,
        "classes shown alone": {k: int(v) for k, v in shown_cls.items()},
        "mean set size (not stopped)": float(n_set[act != R.REAIM].mean()),
        "top class (all crops)": {k: int(v) for k, v in pd.Series(np.array(B.classes)[p.argmax(1)]).value_counts().items()},
        "median OOD p-value": float(np.median([d.p_ood for d in dec])),
    }
    for k in ("re-aim", "nothing shown", "one class shown"):
        m = {"re-aim": act == R.REAIM, "nothing shown": act == R.NOTHING, "one class shown": show}[k]
        out[k + " 95% family interval"] = list(BC.family_ci(m, np.asarray(groups)))
    if y is not None:
        hit = S[np.arange(len(y)), y]
        out["one class shown and right"] = float(np.mean(show & hit))
        out["one class shown and wrong"] = float(np.mean(show & ~hit))
    return dec, act, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", help="write example crops (path, visor text) to this JSON, for slides")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    B = R.ArchA(R.CALIBRATIONS["gc7"])
    cid = {c: i for i, c in enumerate(B.classes)}

    # D5 rupture, unseen by D1, one photo per D5 group
    imgs = pd.read_csv(BC.MAIN / "enhanced_res/results/d5/d5_images.csv")
    crops = pd.read_csv(BC.MAIN / "data/work/crops/D5.csv")
    rup_all = crops[crops.cls == "Rupture"].merge(imgs, on="image", how="left", validate="m:1")
    rup = rup_all[rup_all.kept == True].reset_index(drop=True)            # noqa: E712
    paths = [str(BC.MAIN / p) for p in rup.crop_path]
    F, P, D = BC.outputs("gc7", "d5_rupture", paths)
    dec, act, res = summarise(B, P, D, F, rup.d5_group.to_numpy())

    # how D1 labels the rupture photos it contains: the matched D1 photo's dominant class
    seen = rup_all[rup_all.seen_in_d1 == True].drop_duplicates("image")   # noqa: E712
    match = seen.phash_match.where(seen.phash <= 8, seen.cos_match)
    dom = pd.read_csv(BC.MAIN / "d1_grouped/groups.csv").set_index("key").dominant
    d1_label = match.map(dom).value_counts()

    # references: D5's known classes and the D1 test half (defect crops)
    d5, _ = BC.A5.d5_crops()
    F5, P5, D5_ = BC.outputs("gc7", "d5", list(d5.crop_path))
    _, _, ref_d5 = summarise(B, P5, D5_, F5, d5.d5_group.to_numpy(), y=d5.cls.map(cid).to_numpy())
    va, cal, tst = BC.halves(B.cal["data"]["split"])
    z = np.load(BC.C.CACHE["gc7"], allow_pickle=True)
    s = slice(len(va) + len(cal), None)
    yt = tst.cls.map(cid).to_numpy()
    d = yt != B.nd
    _, _, ref_d1 = summarise(B, z["probs"][s][d], z["knn_dist"][s][d], z["feats"][s][d],
                             tst.src_group.to_numpy()[d], y=yt[d])

    res_all = {"rupture_crops_total": int(len(rup_all)), "rupture_photos_in_d1": int(rup_all.drop_duplicates("image").seen_in_d1.sum()),
               "d1_label_of_rupture_photos": {k: int(v) for k, v in d1_label.items()},
               "rupture": res, "d5_known_classes": ref_d5, "d1_test_half_defects": ref_d1}
    (OUT / "results.json").write_text(json.dumps(res_all, indent=1))

    pct = lambda v: f"{v * 100:.1f}%"                                                     # noqa: E731
    ci = lambda r, k: f"{pct(r[k])} ({pct(r[k + ' 95% family interval'][0])}–{pct(r[k + ' 95% family interval'][1])})"  # noqa: E731
    cols = [("D5 rupture (no class)", res), ("D5 dent + fastener damage", ref_d5), ("D1 test-half defects", ref_d1)]
    md = ["# Architecture B on damage it has no class for: D5 rupture", "",
          f"D5 has {len(rup_all):,} Rupture crops; {res_all['rupture_photos_in_d1']:,} of their photos are also in D1. "
          f"Kept, as for the D5 outside test: {res['crops']} crops from {res['groups']} D5 photos not in D1, one per "
          f"near-duplicate group. Deployed layer (OOD α {B.alpha_ood}, T {B.T:.6f}, CP α {B.alpha}). Report only. "
          "Every rupture answer is wrong by construction; the table says how.", "",
          "| visor outcome | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    rows = [("re-aim (refused)", "re-aim"), ("nothing shown (silent)", "nothing shown"),
            ("AMBIGUOUS (two or more defects)", "ambiguous"), ("POSSIBLE …: check", "possible"),
            ("one class shown, no warning", "one class shown"), ("… of which at 80% or more", "one class shown at 80%+")]
    for lab, k in rows:
        cells = [ci(r, k) if k + " 95% family interval" in r else pct(r[k]) for _, r in cols]
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md.append("| mean confidence when one class is shown | " + " | ".join(pct(r["mean confidence when one class is shown"]) for _, r in cols) + " |")
    md.append("| mean set size, not stopped | " + " | ".join(f"{r['mean set size (not stopped)']:.2f}" for _, r in cols) + " |")
    md.append("| median OOD p-value | " + " | ".join(f"{r['median OOD p-value']:.2f}" for _, r in cols) + " |")
    md.append("| one class shown and right / wrong | – | " + " | ".join(
        f"{pct(r['one class shown and right'])} / {pct(r['one class shown and wrong'])}" for _, r in cols[1:]) + " |")
    md += ["", "Rupture answers shown as one class: " + ", ".join(f"{k} {v}" for k, v in res["classes shown alone"].items())
           + f". Top class over all {res['crops']} rupture crops: " + ", ".join(f"{k} {v}" for k, v in res["top class (all crops)"].items()) + ".",
           "", "How D1 labels the D5 rupture photos it contains (dominant class of the matched D1 photo): "
           + ", ".join(f"{k} {v}" for k, v in d1_label.items()) + ".", "",
           "95% intervals resample D5 photo groups (rupture, D5) or D1 photo groups. Run from 1_main_model_resnet18/: "
           "`python enhanced_res/arch_a/experiments/archB_unseen_class.py`."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    if a.examples:
        rng = np.random.default_rng(0)
        picks = []
        for outcome in (R.SHOW, R.AMBIGUOUS, R.POSSIBLE, R.REAIM, R.NOTHING):
            idx = np.flatnonzero(act == outcome)
            for i in rng.choice(idx, min(2, len(idx)), replace=False):
                picks.append({"path": paths[i], "outcome": outcome, "text": dec[i].text() or "(nothing shown)"})
        Path(a.examples).write_text(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
