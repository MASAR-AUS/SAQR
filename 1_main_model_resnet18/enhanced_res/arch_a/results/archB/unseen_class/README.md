# Architecture B on damage it has no class for: D5 rupture

D5 has 3,788 Rupture crops; 2,411 of their photos are also in D1. Kept, as for the D5 outside test: 490 crops from 295 D5 photos not in D1, one per near-duplicate group. Deployed layer (OOD α 0.05, T 1.089562, CP α 0.1). Report only. Every rupture answer is wrong by construction; the table says how.

| visor outcome | D5 rupture (no class) | D5 dent + fastener damage | D1 test-half defects |
|---|---:|---:|---:|
| re-aim (refused) | 3.1% (1.6%–4.7%) | 0.3% (0.0%–1.0%) | 1.4% (0.5%–2.8%) |
| nothing shown (silent) | 2.4% (1.1%–4.2%) | 2.2% (0.5%–4.7%) | 0.3% (0.1%–0.6%) |
| AMBIGUOUS (two or more defects) | 40.0% | 25.2% | 22.6% |
| POSSIBLE …: check | 34.7% | 5.9% | 8.6% |
| one class shown, no warning | 19.8% (16.0%–23.4%) | 66.5% (60.7%–71.5%) | 67.1% (57.9%–73.8%) |
| … of which at 80% or more | 18.8% | 65.7% | 66.2% |
| mean confidence when one class is shown | 86.9% | 91.7% | 87.8% |
| mean set size, not stopped | 2.57 | 1.55 | 1.54 |
| median OOD p-value | 0.44 | 1.00 | 0.91 |
| one class shown and right / wrong | – | 64.1% / 2.4% | 60.4% / 6.7% |

Rupture answers shown as one class: crack 60, scratch 19, dent 11, corrosion 4, paint-peel-off 2, missing-head 1. Top class over all 490 rupture crops: crack 202, dent 77, scratch 74, no-defect 72, corrosion 31, paint-peel-off 25, missing-head 9.

How D1 labels the D5 rupture photos it contains (dominant class of the matched D1 photo): crack 2026, dent 212, corrosion 125, background 19, paint-peel-off 17, scratch 11, missing-head 1.

95% intervals resample D5 photo groups (rupture, D5) or D1 photo groups. Run from 1_main_model_resnet18/: `python enhanced_res/arch_a/experiments/archB_unseen_class.py`.
