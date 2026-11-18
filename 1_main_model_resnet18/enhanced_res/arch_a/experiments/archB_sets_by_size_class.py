"""Architecture B on the test half: coverage per set size, coverage per class (the deployed global LAC cut-off against
per-class "Mondrian" cut-offs), and which accuracy figure is which. Report only; nothing is chosen here.

Per set size: the 2,711 test-half crops the OOD gate lets through (stopped crops never reach the visor). Per class:
every test-half crop, as in results/archB/README.md. Mondrian CP (Vovk et al. 2005): one LAC cut-off per true class,
fitted like the deployed one (family-weighted, alpha 0.10, gate-passed cal-half crops), but only on that class's crops.

From 1_main_model_resnet18/ (seconds; uses the cached gc7 outputs from calibrate.py):
  python enhanced_res/arch_a/experiments/archB_sets_by_size_class.py
Writes enhanced_res/arch_a/results/archB/sets_by_size_class/README.md and results.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import f1_score

import archB_common as BC
from archB_common import R

OUT = BC.HERE.parent / "results" / "archB" / "sets_by_size_class"


def sets_with(p, cut):
    """LAC sets with a cut-off per class (cut: [n_classes] score thresholds); an empty set keeps its top class."""
    S = p >= 1 - np.asarray(cut)[None, :]
    S[np.arange(len(p)), p.argmax(1)] |= ~S.any(1)
    return S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    B = R.ArchA(R.CALIBRATIONS["gc7"])
    classes, nd, K = list(B.classes), B.nd, len(B.classes)
    cid = {c: i for i, c in enumerate(classes)}
    va, cal, tst = BC.halves(B.cal["data"]["split"])
    z = np.load(BC.C.CACHE["gc7"], allow_pickle=True)
    sl = {"cal": slice(len(va), len(va) + len(cal)), "test": slice(len(va) + len(cal), None)}
    df = {"cal": cal, "test": tst}
    y = {h: df[h].cls.map(cid).to_numpy() for h in df}
    g = {h: df[h].src_group.to_numpy() for h in df}
    P = {h: z["probs"][sl[h]] for h in df}
    dec = {h: B.decide(P[h], knn_dist=z["knn_dist"][sl[h]], feats=z["feats"][sl[h]]) for h in df}
    passed = {h: np.array([d.action != R.REAIM for d in dec[h]]) for h in df}

    # the deployed cut-off, rebuilt from the cal half: must match calibration_gc7.json
    pc = R.temper(P["cal"], B.T)
    q_global = R.family_threshold(pc[passed["cal"]], y["cal"][passed["cal"]], g["cal"][passed["cal"]], B.alpha)
    assert abs(q_global - B.q) < 1e-12, (q_global, B.q)

    # ---- which accuracy is which (top class of the model's probabilities; temperature never changes it)
    def acc(Pm, yy, defects_only=False, best_defect=False):
        m = yy != nd if defects_only else np.ones(len(yy), bool)
        Q = Pm[m].copy()
        if best_defect:
            Q[:, nd] = -1
        return float((Q.argmax(1) == yy[m]).mean()), int(m.sum())
    Pf, yf = np.vstack([P["cal"], P["test"]]), np.concatenate([y["cal"], y["test"]])
    accs = {
        "full test, 7 classes, every crop": acc(Pf, yf),
        "full test, defect crops, top answer of 7": acc(Pf, yf, True),
        "full test, defect crops, best of the 6 defect classes": acc(Pf, yf, True, True),
        "test half, 7 classes, every crop": acc(P["test"], y["test"]),
        "test half, defect crops, best of the 6 defect classes": acc(P["test"], y["test"], True, True),
        "cal half, 7 classes, every crop": acc(P["cal"], y["cal"]),
    }
    macro_f1_full = float(f1_score(yf, Pf.argmax(1), average="macro"))

    # ---- per set size, crops the gate lets through
    p_t, S_t = B.sets(P["test"])
    yt, ok = y["test"], passed["test"]
    hit = S_t[np.arange(len(yt)), yt]
    size = S_t.sum(1)
    top_right = p_t.argmax(1) == yt
    act = np.array([d.action for d in dec["test"]])
    by_size = []
    for k in range(1, K + 1):
        m = ok & (size == k)
        if not m.any():
            continue
        by_size.append({"size": k, "crops": int(m.sum()), "share": float(m.sum() / ok.sum()),
                        "coverage": float(hit[m].mean()), "coverage_95": list(BC.family_ci(hit[m], g["test"][m])),
                        "top_class_right": float(top_right[m].mean()),
                        "visor": {a: int((act[m] == a).sum()) for a in (R.SHOW, R.AMBIGUOUS, R.POSSIBLE, R.NOTHING)
                                  if (act[m] == a).any()}})
    passed_cov = float(hit[ok].mean())

    # ---- per class: global cut-off (deployed) against one cut-off per class (Mondrian)
    pc_ok, yc_ok, gc_ok = pc[passed["cal"]], y["cal"][passed["cal"]], g["cal"][passed["cal"]]
    q_class = [R.family_threshold(pc_ok[yc_ok == c], yc_ok[yc_ok == c], gc_ok[yc_ok == c], B.alpha) for c in range(K)]
    S_m = sets_with(p_t, q_class)
    hit_m = S_m[np.arange(len(yt)), yt]
    per_class = []
    for c in range(K):
        m = yt == c
        per_class.append({"class": classes[c], "test_crops": int(m.sum()), "test_families": int(len(set(g["test"][m]))),
                          "cal_crops": int((yc_ok == c).sum()), "cal_families": int(len(set(gc_ok[yc_ok == c]))),
                          "global": {"coverage": float(hit[m].mean()), "mean_set": float(size[m].mean()),
                                     "single": float((size[m] == 1).mean())},
                          "mondrian": {"cut_off": float(1 - q_class[c]), "coverage": float(hit_m[m].mean()),
                                       "mean_set": float(S_m[m].sum(1).mean()), "single": float((S_m[m].sum(1) == 1).mean())}})
    overall = {"global": {"cut_off": float(1 - B.q), "coverage": float(hit.mean()), "mean_set": float(size.mean()),
                          "single": float((size == 1).mean()),
                          "lowest_class": min(per_class, key=lambda r: r["global"]["coverage"])["class"]},
               "mondrian": {"coverage": float(hit_m.mean()), "mean_set": float(S_m.sum(1).mean()),
                            "single": float((S_m.sum(1) == 1).mean()),
                            "lowest_class": min(per_class, key=lambda r: r["mondrian"]["coverage"])["class"]}}

    res = {"model": "gc7", "data": B.cal["data"]["split"], "T": B.T, "alpha": B.alpha, "alpha_ood": B.alpha_ood,
           "accuracy": {k: {"value": v[0], "crops": v[1]} for k, v in accs.items()}, "macro_f1_full_test": macro_f1_full,
           "passed_test_crops": int(ok.sum()), "coverage_passed": passed_cov, "coverage_all": float(hit.mean()),
           "by_set_size": by_size, "per_class": per_class, "overall": overall}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    pct = lambda v: f"{v * 100:.1f}%"                                                     # noqa: E731
    md = ["# Architecture B: coverage per set size and per class, and which accuracy is which", "",
          f"gc7 on {res['data']}, T {B.T:.6f}, OOD α {B.alpha_ood}, CP α {B.alpha} (LAC, family-weighted). Test half, read "
          "once; report only. Coverage = the set holds the true class. Per-size and per-class coverage are measurements; "
          "conformal prediction guarantees 90% only on average over crops.", "",
          "## Which accuracy is which", "",
          "Top class of the model's probabilities (temperature never changes it).", "",
          "| population | crops | accuracy |", "|---|---:|---:|"]
    md += [f"| {k} | {v[1]:,} | {pct(v[0])} |" for k, v in accs.items()]
    md += ["", f"Macro-F1 over the full test, 7 classes: {macro_f1_full:.3f}. The full test is the cal half plus the test "
           "half. The uncertainty numbers (ECE, coverage) can only use the test half, because the cal half fits the cut-off.", "",
           f"## Coverage per set size ({ok.sum():,} test-half crops that pass the gate)", "",
           "| set size | crops | share | coverage (95% interval over families) | top class right | visor |",
           "|---:|---:|---:|---:|---:|---|"]
    for r in by_size:
        lo, hi = r["coverage_95"]
        vis = ", ".join(f"{a} {n}" for a, n in r["visor"].items())
        md.append(f"| {r['size']} | {r['crops']:,} | {pct(r['share'])} | {pct(r['coverage'])} ({pct(lo)}–{pct(hi)}) | "
                  f"{pct(r['top_class_right'])} | {vis} |")
    md += ["", f"All gate-passed crops: coverage {pct(passed_cov)}. Every test-half crop (stopped ones included, as in "
           f"results/archB/README.md): {pct(hit.mean())}.", "",
           "## Coverage per class: one global cut-off (deployed) against one cut-off per class (Mondrian)", "",
           f"Global cut-off {pct(1 - B.q)} of tempered probability. Mondrian: each class's own cut-off from its gate-passed "
           "cal-half crops, families weighted equally. Every test-half crop.", "",
           "| class | test crops | cal families | global: coverage / mean set / single | Mondrian cut-off | "
           "Mondrian: coverage / mean set / single |", "|---|---:|---:|---:|---:|---:|"]
    for r in per_class:
        a, b = r["global"], r["mondrian"]
        md.append(f"| {r['class']} | {r['test_crops']:,} | {r['cal_families']} | {pct(a['coverage'])} / {a['mean_set']:.2f} / "
                  f"{pct(a['single'])} | {pct(b['cut_off'])} | {pct(b['coverage'])} / {b['mean_set']:.2f} / {pct(b['single'])} |")
    o = overall
    md.append(f"| **all** | {len(yt):,} | {len(set(gc_ok))} | {pct(o['global']['coverage'])} / {o['global']['mean_set']:.2f} / "
              f"{pct(o['global']['single'])} | | {pct(o['mondrian']['coverage'])} / {o['mondrian']['mean_set']:.2f} / "
              f"{pct(o['mondrian']['single'])} |")
    md += ["", "95% intervals resample photo families. Run from 1_main_model_resnet18/: "
           "`python enhanced_res/arch_a/experiments/archB_sets_by_size_class.py`."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
