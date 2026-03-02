# Statistical Significance Report

## openvla_ood
- **steering_vs_mppi**
  - pooled diff: -0.83 pp (95% CI [-13.43, +11.76])
  - p(z-test): 0.8968, p(Fisher): 1
  - paired runs (n=3): mean delta -0.83 pp, p(t-test)=0.874, p(Wilcoxon)=1
- **steering_vs_vanilla**
  - pooled diff: +0.83 pp (95% CI [-11.78, +13.45])
  - p(z-test): 0.897, p(Fisher): 1
  - paired runs (n=3): mean delta +0.83 pp, p(t-test)=0.8675, p(Wilcoxon)=1
- **mppi_vs_vanilla**
  - pooled diff: +1.67 pp (95% CI [-10.94, +14.27])
  - p(z-test): 0.7956, p(Fisher): 0.897
  - paired runs (n=3): mean delta +1.67 pp, p(t-test)=0.1835, p(Wilcoxon)=0.5

## openvla_sweep
- **steering_vs_mppi**
  - pooled diff: +0.14 pp (95% CI [-1.82, +2.10])
  - p(z-test): 0.8879, p(Fisher): 0.9038
  - paired runs (n=25): mean delta +0.11 pp, p(t-test)=0.8648, p(Wilcoxon)=0.8549
- **steering_vs_vanilla**
  - pooled diff: -0.14 pp (95% CI [-2.10, +1.82])
  - p(z-test): 0.8878, p(Fisher): 0.9038
  - paired runs (n=25): mean delta -0.12 pp, p(t-test)=0.8573, p(Wilcoxon)=0.7377
- **mppi_vs_vanilla**
  - pooled diff: -0.28 pp (95% CI [-2.24, +1.68])
  - p(z-test): 0.7779, p(Fisher): 0.7934
  - paired runs (n=25): mean delta -0.23 pp, p(t-test)=0.648, p(Wilcoxon)=0.3349

## act_ood_baselines
- **steering_vs_mppi**
  - pooled diff: -3.57 pp (95% CI [-11.85, +4.70])
  - p(z-test): 0.3978, p(Fisher): 0.4467
  - paired runs (n=3): mean delta -0.33 pp, p(t-test)=0.9261, p(Wilcoxon)=1
- **steering_vs_vanilla**
  - pooled diff: -0.36 pp (95% CI [-8.62, +7.91])
  - p(z-test): 0.9325, p(Fisher): 1
  - paired runs (n=3): mean delta +5.17 pp, p(t-test)=0.4134, p(Wilcoxon)=0.5
- **mppi_vs_vanilla**
  - pooled diff: +3.21 pp (95% CI [-5.06, +11.49])
  - p(z-test): 0.4467, p(Fisher): 0.4989
  - paired runs (n=3): mean delta +5.50 pp, p(t-test)=0.1107, p(Wilcoxon)=0.25

## act_sweep
- **steering_vs_mppi**
  - pooled diff: -0.72 pp (95% CI [-2.83, +1.39])
  - p(z-test): 0.5047, p(Fisher): 0.5186
  - paired runs (n=22): mean delta -0.73 pp, p(t-test)=0.241, p(Wilcoxon)=0.2048
- **steering_vs_vanilla**
  - pooled diff: -0.62 pp (95% CI [-2.73, +1.48])
  - p(z-test): 0.5612, p(Fisher): 0.5758
  - paired runs (n=22): mean delta -0.60 pp, p(t-test)=0.3392, p(Wilcoxon)=0.3046
- **mppi_vs_vanilla**
  - pooled diff: +0.09 pp (95% CI [-2.02, +2.20])
  - p(z-test): 0.9314, p(Fisher): 0.9485
  - paired runs (n=22): mean delta +0.13 pp, p(t-test)=0.8647, p(Wilcoxon)=0.9094

## act_zero_shot_ood
- **steering_vs_mppi**
  - pooled diff: -2.14 pp (95% CI [-10.23, +5.95])
  - p(z-test): 0.6037, p(Fisher): 0.6653
  - paired runs (n=6): mean delta -2.71 pp, p(t-test)=0.2005, p(Wilcoxon)=0.3125
- **steering_vs_vanilla**
  - pooled diff: +0.71 pp (95% CI [-7.32, +8.75])
  - p(z-test): 0.8617, p(Fisher): 0.9306
  - paired runs (n=6): mean delta +1.04 pp, p(t-test)=0.6487, p(Wilcoxon)=0.875
- **mppi_vs_vanilla**
  - pooled diff: +2.86 pp (95% CI [-5.22, +10.93])
  - p(z-test): 0.4881, p(Fisher): 0.5441
  - paired runs (n=6): mean delta +3.75 pp, p(t-test)=0.1067, p(Wilcoxon)=0.1875

## paper_curated
- **steering_vs_mppi**
  - pooled diff: -0.41 pp (95% CI [-1.80, +0.98])
  - p(z-test): 0.5609, p(Fisher): 0.5705
  - paired runs (n=59): mean delta -0.56 pp, p(t-test)=0.2352, p(Wilcoxon)=0.2174
- **steering_vs_vanilla**
  - pooled diff: -0.32 pp (95% CI [-1.71, +1.07])
  - p(z-test): 0.65, p(Fisher): 0.6602
  - paired runs (n=59): mean delta +0.14 pp, p(t-test)=0.7988, p(Wilcoxon)=0.7776
- **mppi_vs_vanilla**
  - pooled diff: +0.09 pp (95% CI [-1.30, +1.48])
  - p(z-test): 0.8984, p(Fisher): 0.9097
  - paired runs (n=59): mean delta +0.69 pp, p(t-test)=0.1337, p(Wilcoxon)=0.2971

