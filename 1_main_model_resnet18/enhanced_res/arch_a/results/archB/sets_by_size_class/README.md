# Architecture B: coverage per set size and per class, and which accuracy is which

gc7 on d1_gc7_fix, T 1.089562, OOD α 0.05, CP α 0.1 (LAC, family-weighted). Test half, read once; report only. Coverage = the set holds the true class. Per-size and per-class coverage are measurements; conformal prediction guarantees 90% only on average over crops.

## Which accuracy is which

Top class of the model's probabilities (temperature never changes it).

| population | crops | accuracy |
|---|---:|---:|
| full test, 7 classes, every crop | 4,868 | 80.4% |
| full test, defect crops, top answer of 7 | 4,253 | 78.8% |
| full test, defect crops, best of the 6 defect classes | 4,253 | 81.1% |
| test half, 7 classes, every crop | 2,752 | 79.4% |
| test half, defect crops, best of the 6 defect classes | 2,417 | 80.0% |
| cal half, 7 classes, every crop | 2,116 | 81.7% |

Macro-F1 over the full test, 7 classes: 0.752. The full test is the cal half plus the test half. The uncertainty numbers (ECE, coverage) can only use the test half, because the cal half fits the cut-off.

## Coverage per set size (2,711 test-half crops that pass the gate)

| set size | crops | share | coverage (95% interval over families) | top class right | visor |
|---:|---:|---:|---:|---:|---|
| 1 | 1,878 | 69.3% | 90.6% (83.0%–95.4%) | 90.6% | show class 1629, nothing shown 249 |
| 2 | 468 | 17.3% | 89.7% (83.7%–94.9%) | 65.6% | ambiguous 351, possible defect 117 |
| 3 | 211 | 7.8% | 92.9% (87.3%–97.0%) | 42.2% | ambiguous 120, possible defect 91 |
| 4 | 105 | 3.9% | 96.2% (92.1%–99.1%) | 46.7% | ambiguous 55, possible defect 50 |
| 5 | 45 | 1.7% | 97.8% (91.9%–100.0%) | 42.2% | ambiguous 22, possible defect 23 |
| 6 | 4 | 0.1% | 100.0% (100.0%–100.0%) | 0.0% | possible defect 4 |

All gate-passed crops: coverage 91.0%. Every test-half crop (stopped ones included, as in results/archB/README.md): 91.0%.

## Coverage per class: one global cut-off (deployed) against one cut-off per class (Mondrian)

Global cut-off 6.3% of tempered probability. Mondrian: each class's own cut-off from its gate-passed cal-half crops, families weighted equally. Every test-half crop.

| class | test crops | cal families | global: coverage / mean set / single | Mondrian cut-off | Mondrian: coverage / mean set / single |
|---|---:|---:|---:|---:|---:|
| corrosion | 257 | 13 | 80.2% / 1.64 / 66.9% | 1.0% | 100.0% / 2.21 / 47.5% |
| crack | 636 | 34 | 87.3% / 1.89 / 55.7% | 1.4% | 96.2% / 3.47 / 0.2% |
| dent | 1,129 | 34 | 95.3% / 1.34 / 76.4% | 5.1% | 95.6% / 3.48 / 0.3% |
| missing-head | 132 | 12 | 97.0% / 1.14 / 88.6% | 16.3% | 95.5% / 2.14 / 1.5% |
| paint-peel-off | 142 | 21 | 85.9% / 1.75 / 58.5% | 6.8% | 85.2% / 2.75 / 12.0% |
| scratch | 121 | 16 | 76.0% / 1.87 / 46.3% | 2.2% | 89.3% / 2.68 / 26.4% |
| no-defect | 335 | 80 | 97.3% / 1.40 / 74.3% | 36.3% | 91.0% / 2.73 / 1.2% |
| **all** | 2,752 | 88 | 91.0% / 1.54 / 68.8% | | 94.8% / 3.13 / 6.6% |

95% intervals resample photo families. Run from 1_main_model_resnet18/: `python enhanced_res/arch_a/experiments/archB_sets_by_size_class.py`.
