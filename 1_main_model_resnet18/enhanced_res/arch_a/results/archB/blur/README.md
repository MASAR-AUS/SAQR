# Architecture B under blur

The d1_gc7_fix test half (2,752 crops: 2,417 defect, 335 no-defect; 87 families), blurred synthetically, through gc7 and architecture B as deployed (OOD α 0.05, T 1.089562, CP α 0.1; no quality gate). Report only: nothing is chosen here. Blur is in pixels of the 224 px crop. Gaussian 2 / 4 px and motion 9 / 21 px are architecture A's s1 / s2 ([`../../archA/step2/`](../../archA/step2/README.md)). "Legacy quality gate" = architecture A's (α 0.05) on the same pixels, for reference; B does not run it. The training augmentation has no blur. Clean pass vs calibrate.py cache: max probability difference 6.0e-08.

## What B does with blurred crops

| blur | sharpness left | legacy quality gate would stop | re-aim (OOD) | top-class accuracy (95% family interval) | 7-class macro-F1 | CP coverage of passed (95% family interval) | mean set size | single class | shown one class, right | single wrong class, no warning | silent miss | visor speaks on no-defect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 100.0% | 4.6% | 1.5% | 79.4% (71.3%–85.4%) | 0.745 | 91.0% (85.4%–95.1%) | 1.52 | 69.3% | 53.1% | 6.1% | 0.3% | 28.1% |
| gaussian 0.5 px | 72.2% | 4.0% | 1.5% | 78.8% (70.6%–85.1%) | 0.739 | 90.7% (85.2%–94.8%) | 1.51 | 69.4% | 53.0% | 6.2% | 0.3% | 27.5% |
| gaussian 1 px | 22.7% | 4.3% | 1.8% | 77.0% (67.3%–84.1%) | 0.712 | 88.8% (82.4%–93.5%) | 1.51 | 68.4% | 50.2% | 6.4% | 1.5% | 24.5% |
| gaussian 1.5 px | 9.1% | 6.7% | 2.0% | 76.4% (65.7%–84.4%) | 0.691 | 86.9% (79.2%–92.5%) | 1.56 | 65.4% | 46.3% | 6.6% | 2.2% | 24.2% |
| gaussian 2 px | 5.8% | 6.8% | 2.8% | 74.2% (62.6%–82.7%) | 0.664 | 86.8% (78.7%–92.7%) | 1.65 | 61.0% | 42.4% | 5.9% | 2.3% | 26.6% |
| gaussian 3 px | 3.7% | 7.0% | 3.6% | 70.4% (59.3%–79.0%) | 0.625 | 85.0% (76.3%–91.6%) | 1.77 | 55.0% | 37.5% | 6.5% | 1.2% | 35.2% |
| gaussian 4 px | 2.9% | 7.9% | 4.0% | 66.6% (56.4%–73.9%) | 0.582 | 83.3% (75.1%–89.6%) | 2.01 | 43.3% | 28.3% | 5.3% | 1.3% | 43.9% |
| motion 5 px | 25.8% | 5.1% | 1.9% | 76.9% (67.0%–84.3%) | 0.706 | 89.3% (83.2%–94.0%) | 1.53 | 67.3% | 49.9% | 6.0% | 1.3% | 25.7% |
| motion 9 px | 15.5% | 5.0% | 3.0% | 75.4% (64.3%–83.9%) | 0.679 | 87.1% (80.2%–92.8%) | 1.59 | 63.5% | 45.9% | 5.7% | 1.7% | 30.4% |
| motion 15 px | 10.2% | 5.5% | 4.8% | 71.1% (58.9%–80.1%) | 0.614 | 83.5% (75.0%–90.5%) | 1.67 | 59.8% | 41.0% | 6.8% | 1.7% | 37.6% |
| motion 21 px | 8.2% | 6.1% | 6.8% | 67.9% (55.6%–77.5%) | 0.570 | 82.2% (72.7%–89.1%) | 1.76 | 55.2% | 35.8% | 7.1% | 1.7% | 41.5% |

Sharpness left = median over crops of (variance of the Laplacian after blur) / (before). CP coverage is only guaranteed for crops like the calibration crops; under blur it is a measurement.

## Real blur: untouched defect crops by their own sharpness

Fifths of the test-half defect crops by variance of the Laplacian (1 = least sharp). Low sharpness also means plain surfaces, not only blur.

| fifth | crops (families) | sharpness | top-class accuracy | re-aim | CP coverage of passed | silent miss |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 484 (43) | 0–4 | 73.6% | 0.2% | 85.9% | 0.6% |
| 2 | 483 (61) | 4–34 | 67.1% | 2.3% | 87.7% | 0.6% |
| 3 | 483 (58) | 34–146 | 76.8% | 0.4% | 88.4% | 0.0% |
| 4 | 483 (37) | 147–327 | 82.2% | 0.4% | 91.1% | 0.2% |
| 5 | 484 (30) | 328–3978 | 89.5% | 3.5% | 97.6% | 0.2% |

Run from 1_main_model_resnet18/: `python enhanced_res/arch_a/experiments/archB_blur.py`.
