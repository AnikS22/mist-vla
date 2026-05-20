# ADVERSARIAL CoRL REVIEW — PULSE: Probing Underlying Latent Safety Embeddings in Robot Foundation Policies

**Reviewer posture:** Elite CoRL reviewer. Maximally skeptical. Scientifically precise. No softening.

---

## PHASE 1 — CONTRIBUTION TRIAGE

### 1.1 Actual Contributions

**Genuinely Novel (thin but real):**
- Demonstrating that an MLP probe over frozen VLA/ACT hidden states gives +0.047/+0.101 AUC over a linear probe baseline for binary failure detection. This is the only result that survives all the negative-result disclosures intact.
- Honest negative result: learned correction directions do not outperform random-magnitude-matched perturbation (latent jiggle). This is scientifically informative and unusually candid.
- The early-rollout directionality probe (Appendix) showing the hidden state anticipates spatial failure direction before kinematics commit — a narrow but interesting mechanistic insight.

**Incremental Extensions:**
- TTF regression head extending binary detection to continuous time — **immediately undermined** by the authors' own disclosure that a raw step counter achieves r≈1.0 on this benchmark (vs. PULSE's r=0.86). This is not a contribution; it is a demonstration that LIBERO failure rollouts run to fixed episode length. The paper correctly downgrades this but still lists it as C1.
- MLP probe template working on two architectures (OpenVLA 4096-d, ACT 256-d) without per-architecture tuning — this is an engineering observation, not a scientific result. The safety *directions* explicitly do not transfer (ACT→OpenVLA AUC 0.27, below chance).

**Engineering Observations:**
- Single-step querying of the PULSE failure head is 6.1× faster than K-sample MPPI querying of the **same** failure head. This is a tautologically expected result: of course querying a function once is faster than querying it K=16 times. The paper frames this as a "low-latency safety planning" result, which significantly overstates what it is.

**Negative Results (scientifically valuable):**
- Correction direction does not help. Latent jiggle (same gating, random direction) ties or beats learned steering. Correctly reported.
- Cross-architecture safety direction does not transfer linearly. Correctly reported.
- Random noise (no probe, Gaussian σ=0.05, every step) is the **highest-success row** at 53.98%, beating all PULSE variants. This is the most damaging finding in the paper and is buried in a paragraph within the results section.

**Unsupported "Marketing" Claims:**
- The title says "Safety Embeddings" — the probe detects failure; it does not enforce safety in any meaningful sense. The paper itself clarifies this in the Discussion ("safety-specific framing would need conformal calibration and validation against a real safety specification"), but the title and abstract language does not match this hedge.
- "Self-calibrating to task difficulty" (r=−0.98) — this is an emergent property of the failure detector firing more on hard tasks, not a designed calibration mechanism. Calling it "self-calibration" implies intentional adaptive behavior.
- "Architecture-agnostic probe template" — the template trains separately on each architecture from scratch and the safety signal does not transfer. The architecture-agnostic claim covers only the MLP topology, not the learned representations.

### 1.2 Is the Paper's Framing Honest?

**Title:** "PULSE: Probing Underlying Latent Safety Embeddings in Robot Foundation Policies" — "Safety Embeddings" overstates. The probe predicts task failure, not safety constraint violation. A robot that fails to place a bowl is not necessarily unsafe.

**Abstract:** Unusually honest for a top-venue submission — it discloses the TTF clock equivalence, the jiggle negative result, the calibration failure (ECE 0.22), and the simulation-only scope. However, the abstract is extremely dense (one paragraph approaching ~350 words) and buries the devastating Random Noise result until the Discussion section. A reader skimming the abstract gets a more positive picture than the full paper supports.

**Introduction:** The "Contributions" bullet structure (C1/C2/C3) makes weak results look like strong contributions. C1 includes TTF despite the clock baseline matching it. C2 calls single-step gating a "Low-latency safety planning" result, which obscures that it is a trivial inference-count comparison against the same head. C3 ("Architecture-agnostic probe template") is the definition of incremental.

**Generalization claims:** "Zero-shot generalization" means training on 8 LIBERO-Spatial tasks and testing on the other 2 LIBERO-Spatial tasks in the same simulator with the same physics. This is within-distribution interpolation, not generalization in any meaningful robotics sense.

**"Safety" claims vs. capability claims:** The paper deploys the word "safety" throughout but the evaluation metric is binary task success on a tabletop pick-and-place benchmark. There is no safety specification, no constraint set, no injury/damage metric, no safety-relevant failure mode. This is capability detection dressed as safety research.

### 1.3 True Scientific Story

**Real contribution in one sentence:** A multi-head MLP probe over frozen robot policy hidden states improves failure-detection AUC by ~0.05–0.10 over a linear probe on LIBERO-Spatial, and single-step thresholding on the failure logit is 6.1× faster than K-sample MPPI over the same logit while achieving statistically equivalent task success on the same simulator benchmark.

**What is NOT supported in one sentence:** The correction-direction head does not work, the TTF head is a clock surrogate, a probe-free random Gaussian perturbation outperforms all PULSE controllers numerically, no real-robot validation exists, "safety" framing is not justified by any safety specification, and generalization beyond LIBERO-Spatial is entirely untested.

---

## PHASE 2 — SECTION-BY-SECTION REVIEW

### ABSTRACT

**Overclaiming:** The abstract correctly hedges most results, but the phrase "low-latency drop-in for sampling-based safety planners" frames a trivial inference-count reduction as a systems contribution. The 6.1× speedup is not a property of the method — it is a consequence of calling a function 1 time vs. 16 times.

**Consistency with Tables:** The pooled success numbers (51.7%, 52.1%) are consistent with tab_final_pooled_results. The AUCs (0.83/0.89) are consistent with tab_safety_head_metrics. The TTF r=0.86 is consistent. The TOST p=0.011 is consistent. No inconsistencies detected.

**Omitted negatives:** The abstract mentions the jiggle negative result and TTF clock issue explicitly — unusual and commendable. However, the Random Noise being the highest-success row (53.98%) does not appear until the Discussion. A reader of only the abstract concludes the probe is a net positive; a reader of the full Results section learns the best method is no probe at all.

**Structural problem:** The abstract is ~340 words and contains multiple parenthetical technical caveats (TOST, ECE, Platt scaling). This is not an abstract — it is a compressed version of the Introduction. At CoRL, reviewers see the abstract first, and this will immediately signal a paper that cannot identify its own headline result.

**Causal claims:** No unjustified causal claims detected. The framing is appropriately correlational for probe results.

### INTRODUCTION

**Contributions novelty:**
- C1 partially supported but the TTF component is undermined within the same bullet point by the authors' own temporal baseline disclosure. A contribution that the authors themselves invalidate within its own statement is not a contribution — it is a caveat.
- C2 describes single-step gating as achieving "low-latency safety planning." This is a false framing. Low-latency planning implies a planning algorithm was improved. What actually happened: the authors compared querying a neural network once vs. 16 times and observed the expected speedup. This should be framed as a deployment engineering observation.
- C3 ("Architecture-agnostic probe template") — the probe template is a three-layer MLP with LayerNorm. Demonstrating that a generic MLP architecture can be trained on two different input dimensionalities (256 vs. 4096) from scratch is not a scientific contribution. Every MLP is trivially "architecture-agnostic" in this sense.

**Related work representation:** The introduction cites SAFE, FAIL-Detect, and FIPER but explicitly states these are not evaluated baselines. This is a significant weakness flagged below.

**Precision:** The "self-calibrates to task difficulty (r=−0.98 between intervention rate and vanilla success)" claim needs scrutiny. If hard tasks fail more often and the failure detector fires more on failing steps, then intervention rate correlates with task difficulty *by construction* — this is not a probe property, it is arithmetic.

**Scope paragraph:** The paragraph explicitly states "All quantitative claims in this submission are empirically evaluated on LIBERO-Spatial only." This is appropriately scoped, but placing this scope statement at the bottom of the Introduction (after three contribution bullets that imply broader applicability) is structurally misleading. It should be the first sentence of the Introduction.

### RELATED WORK

**Missing papers that should be baselines or comparisons:**
1. **FIPER** (cited) — random network distillation + action-chunk entropy for failure detection. This is directly competitive and should be a numeric baseline. The authors acknowledge this ("a head-to-head numeric comparison is a clear next step") but do not run it.
2. **FAIL-Detect** (cited) — OOD detection without labeled failure data. Also directly competitive. Not run.
3. **DreamerV3 / world-model-based safety** — not cited. Latent safety signals have been studied in model-based RL settings.
4. **IRL/inverse reward-based failure detection** — no citation to work using reward signal for failure detection.
5. **Conformal prediction for robotics** — cited briefly but no baseline using conformal guarantees (SAFE's primary approach) in the controller evaluation.

**Honest representation of competing methods:** The paper concedes all three main competitors (SAFE, FIPER, FAIL-Detect) are not experimentally compared. For a CoRL submission, this is a critical gap. A reviewer can legitimately reject on this basis alone.

**Missing citation to latent space control literature:** Activation engineering (cited), but no mention of DiffusionPolicy's noise-space interventions, RT-2's language conditioning of latent, or similar inference-time modification work that is highly relevant.

### METHOD

**Mathematical clarity:**
The problem formulation is clear. The three-headed loss (Eq. 1) is well-specified. The inference equations (Eqs. 2–5) are standard (EMA, clamp, gate, apply).

**Critical hidden assumption — correction label construction:**
The correction label Δp_t = p^succ_t − p^fail_t is computed from nearest-neighbor task-matched success trajectories. This is a significant methodological weakness:
1. The "nearest neighbor" is time-index matched (step t of failure trajectory paired with step t of success trajectory). This is not a geometrically meaningful nearest-neighbor match — it is just temporal alignment within the same task. A failure at step 80 is paired with the success trajectory's step-80 EEF position, regardless of where the success trajectory actually is in the workspace at that moment.
2. This label construction almost certainly produces noisy, geometrically inconsistent correction vectors, which is the most likely explanation for why the correction head fails. The paper identifies this in the Conclusion ("recovery-conditioned labels") but does not present it as a *methodological flaw* — it is described as future work, which undersells the seriousness of the problem.
3. No ablation of label construction strategies is presented. Would task-conditioned optimal-transport matching, or DTW alignment, produce better correction labels? Unknown.

**Reward hacking / gating analysis:**
The double gating (Eq. 4) fires on 40% of timesteps on average (range 10–66%). A "safety" controller that modifies the policy 40% of the time is not a safety intervention — it is a continuous policy modification. This undermines the "lightweight drop-in safety layer" framing. The paper correctly reports this number but does not adequately discuss its implications for the safety framing.

**Architecture definition:**
The 108K/1.1M parameter count is reported. The encoder is specified (m→256→128→64). No specification of training duration, learning rate schedule, or convergence criteria beyond "AdamW, early stopping on val AUC." This is insufficient for reproducibility. Specifically:
- What is the learning rate? (Not stated.)
- How many epochs before early stopping typically triggers?
- Are the three heads trained simultaneously or sequentially?
- What are λ_f, λ_τ, λ_c (the loss weights)? They appear in Eq. 1 but are never specified in the text.

**State/action definitions:**
The correction vector Δp ∈ ℝ³ is Cartesian EEF position. The policy outputs action a_t ∈ ℝ^d. The correction is applied to the XYZ components of a_t (Eq. 5). For ACT with 8-step action chunks, which chunk step does the correction apply to? Step 1 of the chunk? All steps? This is not specified and matters for the evaluation.

**Temporal leakage risk:**
The trajectory-level 75/15/10 split (stratified by task and label) is correct and prevents trajectory-level leakage. However, the correction label Δp_t = p^succ_t − p^fail_t is computed using success trajectories that are in the training set, and applied as targets for training the correction head on the same failure trajectories. If the success trajectory used for label generation for a failure trajectory in the *test* set happens to be a success trajectory in the *training* set (which it almost certainly is, since the label-generation pool and the training pool both draw from the same rollout corpus), then the correction head's labels for test failures are derived from training data. This is a subtle form of leakage for the correction head specifically. The paper does not discuss this.

**MPPI implementation:**
The K-sample MPPI uses K=16 samples per step with "feature-space perturbation σ=0.01." Feature-space perturbation of the hidden state is not a standard MPPI perturbation — normally MPPI perturbs action-space trajectories. The authors are perturbing the *input features* to the probe and scoring the resulting failure logit. This is not the same as sampling candidate action trajectories and scoring them with a cost function. The framing as "MPPI" is misleading. This should be described as "K-sample probe-score averaging over feature-space perturbations."

### EXPERIMENTS

**Benchmark diversity — catastrophically narrow:**
All quantitative results come from LIBERO-Spatial: one simulator (MuJoCo), one task suite (tabletop pick-and-place), one episode structure (300-step, binary success). The paper does not test on:
- LIBERO-Goal, LIBERO-Object, LIBERO-Long (three immediately adjacent benchmark suites)
- RoboCasa
- Any real robot
- Any environment outside tabletop pick-and-place
- Any policy outside OpenVLA-7B OFT and ACT

For a CoRL paper, results on a single simulation environment with no real-robot validation are insufficient for acceptance under current standards.

**Realism:**
The sim-to-real gap is noted but entirely unaddressed. The paper includes a blue-colored placeholder box for "SO-101 Robot Setup Photo" in the Physical Robot Evaluation subsection of the Results section. This should not exist in a submitted paper. It signals the paper was submitted incomplete.

**Baseline quality — critical failure:**
The primary comparison is PULSE steering vs. K-sample MPPI over the same PULSE failure head vs. vanilla. The baselines do not include:
1. **SAFE** (the closest prior work) — not re-implemented and compared.
2. **FIPER** — not implemented.
3. **FAIL-Detect** — not implemented.
4. **Any classical safety filter** (CBF, shielded RL) — not compared.
5. **A threshold on prediction error** — e.g., model-error-based detection like ensemble disagreement.
6. **The random-noise "baseline"** (Gaussian σ=0.05) is the best-performing method at 53.98%, and it requires zero probe training. This is presented as a transparency note rather than the primary result it actually is.

The absence of FIPER and FAIL-Detect is especially damning given that both are cited in related work, the rollout pool already exists to evaluate them, and the authors themselves acknowledge "a head-to-head numeric comparison is a clear next step we did not run here." Running these comparisons is not optional at CoRL.

**Seed counts and variance:**
N=20 episodes/mode/task in the main pool. This is insufficient for tasks with binary outcomes at ~50% success rates. A single episode flip changes a task-level success rate by 5pp. The paper pools across all tasks and seeds to reach n≈10,000, which gives adequate power for pooled equivalence testing, but per-task and per-seed variance is high. The gating ablation (Tab. A7) uses n=5 per cell, which the paper correctly flags as directional-only but still includes in a main-paper appendix table — this is borderline misleading.

**Confidence intervals:**
Wilson CIs are reported in the appendix (tab_wilson_paper_curated). However, the main text reports only point estimates and p-values. CoRL readers expect CIs alongside effect sizes.

**Significance testing:**
The TOST equivalence framework is correctly applied with Holm correction on the primary contrast family. The ±2pp margin is self-justified ("the tightest honest claim our data support") — while defensible, a reviewer will observe that this margin was chosen *after* seeing the data, which raises the question of whether the equivalence conclusion is pre-registered or post-hoc. The paper provides no pre-registration.

**The intervention rate at 40%:**
A 40% average intervention rate means the PULSE controller modifies the policy action at nearly half of all timesteps. This fundamentally undermines the "lightweight safety overlay" framing. If the controller fires on 40% of steps, it is not a safety intervention — it is a co-policy. The correct framing would be "joint policy" or "action-space additive controller," not "safety probe."

**Sample complexity:**
~2,000 GPU-hours on A100 for training and evaluation. The data pool requires 699 success + 874 failure trajectories (1,573 total rollouts) of labeled data collected by running the frozen policy and labeling with LIBERO's predefined success criterion. This is a non-trivial data requirement that is not discussed in terms of practical deployment cost. In a real deployment, how do you collect labeled failure trajectories before deployment? The paper assumes a training phase with access to failure data, which is a strong assumption not acknowledged.

**Ablation completeness:**
- Correction label construction: **not ablated**
- Loss weights (λ_f, λ_τ, λ_c): **not ablated**
- Bottleneck dimensionality (256→128→64 vs. alternatives): **not ablated**
- Number of MPPI samples K (only K=16 tested): **not ablated**
- EMA smoothing β=0.9: **not ablated**
- Training data size: **not ablated** (how much labeled failure data is needed?)
- Layer from which hidden states are extracted (which layer of OpenVLA/ACT?): **not stated**

**Conclusions matching results:**
The conclusion states PULSE "self-calibrates to difficulty and generalizes zero-shot." The zero-shot result (37.5% vs. 38.2%, p=0.86) shows the PULSE controller is indistinguishable from vanilla on held-out LIBERO-Spatial tasks. Calling this "generalizes zero-shot" is technically accurate (TOST equivalent) but frames a null result as a positive generalization claim. Zero-shot on 2 held-out tasks from the same simulator is the weakest possible generalization test.

### DISCUSSION / CONCLUSION

**Unsupported speculation:**
- "Three directions may address the correction bottleneck" — recovery-conditioned labels, multi-step correction planning, task-specific heads. These are reasonable suggestions but are pure speculation with no preliminary evidence. At CoRL, three future-work bullet points dressed up as a contribution direction is recognized as filler.
- "The probe self-calibrates to task difficulty" — as noted above, this is arithmetic, not a designed property of the probe.

**Future-work fluff:**
The conclusion contains six future-work items marked in blue (physical robot, nonlinear transfer, conformal prediction, variable-length TTF, etc.). Blue-colored text in the main paper body is editorially inappropriate for a final submission. It signals the paper was written in draft mode and never properly finalized.

**Hidden limitations:**
1. The learning rate, loss weights, and training convergence details are missing from the method section.
2. The correction label construction is acknowledged as a problem but not adequately analyzed as a methodological flaw.
3. The 40% intervention rate's implications for the safety framing are not directly confronted.
4. The fact that the best baseline requires no probe training (Random Noise) is not placed in the limitations section — it is placed in a results subsection paragraph where it can be easily missed.

**Overgeneralization:**
"The jiggle ablation simplifies the safety-adapter design space and identifies correction fidelity as the clear bottleneck." This is an overconfident causal claim from a single experiment on one benchmark. The correction head failing might be an artifact of the correction label construction (temporal nearest-neighbor matching), the specific task structure of LIBERO-Spatial (where failures are geometrically stereotyped), or the gating threshold firing after the informative window has closed (as the paper's own directionality probe suggests). "Correction fidelity is the bottleneck" is one of several competing explanations and cannot be claimed as the "clear" bottleneck from this evidence.

---

## PHASE 3 — AGGRESSIVE REVIEWER OBJECTIONS

### R1 — No real-robot results, and the physical robot section is a placeholder. [REJECT-LEVEL]

**The problem:** Section 4.8 "Physical Robot Evaluation" in the submitted paper contains a \fbox with the text "SO-101 Robot Setup Photo — Placeholder." The results reported in this section consist of three sentences in blue text announcing that physical evaluation is "planned." This is not a Results subsection — it is a project roadmap item. Submitting a paper with a placeholder figure box in the main results section is a sign of premature submission.

**Why it threatens rejection:** CoRL 2026 is a robot learning conference. Simulation-only results for a method targeting robot safety deployment are insufficient unless the paper makes an unusually strong mechanistic or theoretical contribution. PULSE does not make such a contribution — it is an empirical method. Simulation-only empirical safety papers need to demonstrate at minimum that (a) the probe generalizes across simulation environments and (b) there is a credible sim-to-real path. Neither is demonstrated.

**Rebuttal potential:** Extremely limited. Running physical robot experiments in rebuttal period is not feasible. The authors can argue the simulation results are sufficient given the scope framing; reviewers at CoRL are unlikely to find this argument persuasive.

**What's needed:** 50 episodes/mode on SO-101 or equivalent with blind success labeling before submission.

---

### R2 — FIPER and FAIL-Detect are omitted as baselines despite being cited. [MAJOR]

**The problem:** The two most directly comparable prior methods for failure detection from policy representations (FIPER and FAIL-Detect) are cited in related work but not experimentally compared. The paper explicitly acknowledges "a head-to-head numeric comparison is a clear next step we did not run here." The authors have the rollout data; they simply chose not to run these comparisons.

**Why it threatens rejection:** Without these comparisons, the +0.047/+0.101 AUC improvement over a linear probe baseline is uncontextualized. A reviewer cannot determine whether PULSE's improvements are meaningful relative to the state of the art. SAFE, FIPER, and FAIL-Detect could all achieve similar or better AUCs; we do not know.

**Rebuttal potential:** Moderate. If the authors can implement FIPER and FAIL-Detect on their existing rollout pool during the rebuttal period and show PULSE compares favorably, this objection can be partially addressed.

**What's needed:** Port FIPER's random network distillation + entropy baseline and FAIL-Detect's OOD approach to the existing LIBERO rollout pool. Report failure AUC alongside PULSE.

---

### R3 — The "latency advantage" is a trivial inference-count comparison. [MAJOR]

**The problem:** The headlined "6.1× latency advantage" compares: (A) querying the PULSE failure head once per step (0.9ms) vs. (B) querying the PULSE failure head 16 times per step (5.6ms). Multiplying 0.9ms × 6.1 ≈ 5.5ms, which is approximately 16 × 0.9ms ÷ some parallelization overhead. This is not a discovery — it is the definition of what happens when you call a function fewer times. No alternative safety method is compared. No hand-designed cost function is compared. The only comparison is inference count on the same function.

**Why it threatens rejection:** If a reviewer realizes the latency comparison is purely an inference-count ablation on the same head, the paper loses its only unambiguously positive empirical result. The AUC improvement over linear probe is small; the correction direction fails; the TTF is a clock; Random Noise beats everything. The latency result is the paper's remaining positive claim, and it is not a meaningful one.

**Rebuttal potential:** Low. This is a framing problem baked into the paper's core contribution structure. The authors could acknowledge this more directly, but cannot reframe it as a genuine algorithmic contribution.

**What's needed:** Compare single-step PULSE gating against SAFE's actual inference-time pipeline, FIPER's entropy-based controller, and at least one CBF-based safety filter. Show whether PULSE's latency advantage holds relative to real competing methods, not just K-sample self-comparison.

---

### R4 — Random Noise is the best controller and it requires no probe. [REJECT-LEVEL for safety framing]

**The problem:** The best-performing method in Table 1 is "Random Noise" (Gaussian action-space perturbation σ=0.05, no gating, no PULSE, OpenVLA-only n=5,080) at 53.98%, statistically significantly above PULSE steering (z=−2.60, p=0.009 before multiplicity correction). This means a probe-free additive noise baseline beats all PULSE variants on the one benchmark where PULSE is evaluated. This result is buried in a paragraph titled "Random-noise note (highest-success row)" in the middle of the Results section.

**Why it threatens rejection:** If random perturbation is the best controller, the entire premise of the paper — that learned latent safety signals improve or preserve performance relative to no intervention — is falsified on this benchmark. The honest conclusion is that LIBERO-Spatial's ceiling is the base policy's ceiling, and any sufficiently mild perturbation achieves equivalent or better success by escaping local basins of failure. The probe is not adding value.

**Rebuttal potential:** Low. The result exists in the paper's own tables. The authors argue (correctly) that the contribution is latency reduction, not success-rate improvement. But this means the paper is not a safety paper — it is a latency optimization paper for an unnecessary detector.

**What's needed:** Either (a) demonstrate a setting where PULSE steering provides a statistically significant success improvement over random noise, or (b) reframe the paper entirely as a mechanistic analysis of failure representations, dropping the controller framing.

---

### R5 — Correction label construction is a methodological flaw, not future work. [MAJOR]

**The problem:** Correction labels are computed as Δp_t = p^succ_t − p^fail_t using temporal nearest-neighbor matching: the step-t position of a failure trajectory is subtracted from the step-t position of the nearest success trajectory. This produces labels that are geometrically meaningless in many cases — at step 80 of a failure trajectory, the arm might be in mid-air; the success trajectory at step 80 might be reaching for the object from a different approach angle. The resulting "correction vector" is not a recovery direction; it is a displacement between two unrelated configurations.

**Why it threatens rejection:** If the correction labels are geometrically meaningless, the fact that the correction head "fails" is not a discovery about latent representations — it is a discovery about bad labels. The paper cannot distinguish these two explanations. It claims the latent does not encode corrective direction; the equally valid explanation is that the labels do not encode corrective direction.

**Rebuttal potential:** Moderate if the authors run an ablation with DTW-aligned labels or task-conditioned trajectory matching during the rebuttal period.

**What's needed:** At minimum, visualize the distribution of correction vectors |Δp| and their cosine alignment with actual recovery directions from human demonstrations. Show the labels are geometrically sensible before claiming the latent cannot encode them.

---

### R6 — 40% intervention rate undermines the safety framing. [MODERATE]

**The problem:** The PULSE controller modifies policy actions at 40% of timesteps on average (range 10–66%, median 37%). A controller that modifies the action at 40% of timesteps is a co-policy, not a safety overlay. This is especially concerning because (a) the modified actions are not validated to be safe or recovery-directed (the correction head fails), and (b) the intervention rate on the hardest task reaches 97.5% of timesteps — essentially replacing the policy entirely. The paper describes this as "self-calibration," which is a framing that hides the severity.

**Rebuttal potential:** Moderate. The authors can argue (correctly) that the gating fires on failure-probable steps and that the intervention, while frequent, causes no degradation (TOST equivalent to vanilla). But the safety framing requires justification beyond "no worse."

**What's needed:** Show that the 40% intervention rate does not introduce safety-relevant artifacts (joint limit violations, self-collisions, workspace exits). On real hardware, frequent uncontrolled corrections are a safety concern, not a safety solution.

---

### R7 — Single-benchmark evaluation makes all generalization claims unverifiable. [MAJOR]

**The problem:** All quantitative results are from LIBERO-Spatial only. The "zero-shot" result covers 2 held-out tasks from the same suite. The "cross-architecture" result covers two models on the same suite. The "OOD" result covers disturbance injection within the same suite. These are not independent tests of generalization; they are sub-experiments within a single evaluation environment.

**Why it threatens rejection:** CoRL reviewers expect evidence across multiple environments. LIBERO-Goal, LIBERO-Object, LIBERO-Long are adjacent and immediately available. RoboCasa provides a more realistic tabletop environment. The fact that none of these are tested suggests the results may be LIBERO-Spatial-specific.

**Rebuttal potential:** Low. Running additional environments in rebuttal period is not feasible for a full evaluation.

---

## PHASE 4 — CLAIM VERIFICATION

### SUPPORTED CLAIMS

1. **Failure-detection AUC (0.83 OpenVLA, 0.89 ACT)** — supported by tab_safety_head_metrics, trajectory-disjoint test split, data audit confirming consistency.
2. **+0.047/+0.101 AUC over linear probe baseline** — supported by same table. Linear probe is a legitimate and well-defined baseline.
3. **TOST equivalence (vanilla ≡ K-sample MPPI ≡ single-step gating) at ±2pp, p∈{0.003, 0.008, 0.011}** — supported by tab_equivalence_tost. Holm correction applied correctly to the primary contrast family.
4. **6.1× latency reduction (0.9ms vs. 5.6ms)** — supported by tab_final_pooled_results and secondary measurement in tab_latency. Both measurements are consistent in direction.
5. **Latent jiggle TOST fails vs. single-step gating (p_TOST=0.119) and vs. vanilla (p_TOST=0.053)** — supported. Correctly flagged as the negative-result framing motivation.
6. **TTF clock equivalence (step counter achieves r≈1.0 vs. PULSE's 0.86 on LIBERO-Spatial failure rollouts)** — supported by tab_temporal_baseline and authors' own disclosure.
7. **Cross-architecture linear transfer fails (ACT→OpenVLA AUC 0.27, OpenVLA→ACT AUC 0.42)** — supported by results in §4.7.
8. **Random Noise achieves 53.98% on OpenVLA sub-pool** — supported by tab_final_pooled_results, z=−2.60 vs. steering.
9. **Intervention rate r=−0.981 correlation with vanilla success (p=5.2×10⁻⁷, n=10)** — supported by adaptive gating statistics.

### PARTIALLY SUPPORTED CLAIMS

1. **"Hidden states encode failure timing" (TTF r=0.86)** — directionally true but fully undermined by the clock equivalence. The r=0.86 figure is reported alongside the admission that r≈1.0 is achievable with just t. The substantive version of this claim (that the hidden state encodes TTF *beyond* what a clock provides) is *not* supported on LIBERO-Spatial.

2. **"The probe self-calibrates to task difficulty"** — the correlation (r=−0.98) is real, but calling it "self-calibration" overstates its significance. The correlation is an arithmetic consequence of failure detection firing more on hard tasks. The mechanism is not designed self-calibration; it is passive emergent behavior. Partially supported as a behavioral observation; not supported as a designed property.

3. **"Architecture-agnostic"** — the MLP template trains on two architectures without modification. Supported as an engineering observation. Not supported as a claim about the safety manifold being shared (cross-architecture transfer fails).

4. **Early-rollout directionality (hidden state predicts failure direction before kinematics commit)** — directional displacement error improves from 8.7→7.6cm (OpenVLA) and 9.4→7.7cm (ACT) in the first 25% of failure trajectories. Partially supported: the improvement exists, but the comparison to a current-EEF baseline is close enough that the effect may be modest and sensitive to analysis choices. n is not independently reported for this sub-analysis.

### UNSUPPORTED CLAIMS

1. **"PULSE achieves... a +0.05 to +0.10 gain over a linear-probe baseline"** (Abstract) — the abstract's range notation implies a consistent +0.05 to +0.10 band across conditions. In reality: OpenVLA gains +0.047, ACT gains +0.101. The upper end of the range (0.10) is from a single architecture; the framing suggests a generalizable range.

2. **"The probe predicts when to intervene but does not predict where to push"** — this is stated as a factual conclusion, but the correct mechanistic claim is weaker: "on LIBERO-Spatial with temporal-nearest-neighbor correction labels, the correction head's output does not outperform random direction." The inference that the latent does not encode corrective direction is not validated — the directionality probe (Appendix) shows the latent *does* encode spatial failure direction early in rollouts. The paper's own evidence contradicts this claim.

   **Problematic text:** "The probe predicts *when* to intervene but does not predict *where to push*." (Introduction, negative result paragraph; repeated in Discussion and Conclusion.)
   
   **Why invalid:** The authors' own Appendix spatial failure analysis shows that hidden states *do* encode failure direction early in rollouts (8.7→7.6cm, 9.4→7.7cm improvement). The failure is not that the latent lacks directional information — it is that (a) the controller threshold fires after the informative window closes, and (b) the correction labels may not capture true recovery direction. "Doesn't predict where to push" is too strong a conclusion.

3. **"Generalizes zero-shot"** — the result is p=0.86 (vanilla parity), TOST near-equivalent. Vanilla parity is not generalization — it is non-degradation. Saying the probe "generalizes zero-shot" implies it provides some benefit in the zero-shot setting; the evidence shows it provides no benefit (which may actually be the correct interpretation for safety deployment, but should be framed as "maintains performance" not "generalizes").

   **Problematic text:** "self-calibrating to difficulty and generalizing zero-shot" (Conclusion).

### MISLEADING CLAIMS

1. **"Using the same PULSE failure head as a controller cost in two ways—a K-sample MPPI baseline... and direct single-step gating"** (Abstract) — the word "MPPI" implies a principled sampling-based planning algorithm operating in the action-trajectory space. What the paper describes is K independent perturbations of the *feature space* input to the failure head, with the lowest-cost perturbation selected. This is not MPPI; it is K-sample probe evaluation with feature-space noise. Using "MPPI" suggests the method is comparable to literature on MPPI-based robot planning; it is not.

   **Problematic text:** "K-sample Model Predictive Path Integral (MPPI) baseline that scores candidate corrections by querying the head K times" — this description, combined with the citation to Williams et al., implies a trajectory optimization connection that is not present.

2. **"6.1× lower per-step latency"** (Abstract and throughout) — framed as a property of the method, but it is solely a function of K=16 vs. K=1 inference calls. The latency of any probe-based method with K=1 calls would achieve the same speedup over its own K=16 variant. This is not a discovery about PULSE specifically.

3. **"A low-latency drop-in for sampling-based safety planners"** (Abstract) — implies PULSE is a replacement for class of existing safety planners. No safety planner (CBF, shielded RL, MPPI-based) is actually compared. The "drop-in" claim has no empirical support relative to the field.

4. **"The corrective steering result is negative and we report it as the central caveat"** (Abstract) — presenting the failure of a full head as the "central caveat" rather than a "primary result" softens what is actually a fundamental result: the method's most ambitious component (the correction head) completely fails. The caveat framing implies this is a minor qualification on an otherwise positive paper; it is not.

---

## PHASE 5 — REQUIRED FIXES BEFORE SUBMISSION

### Must-Fix (Paper Likely Rejected Without These)

**MF1. Remove the Physical Robot Evaluation placeholder section entirely.**
The `\fbox{\parbox{...}}` placeholder for "SO-101 Robot Setup Photo" and the blue-text "planned" subsection in §4.8 must be deleted from the submitted paper. This is a results section, not a roadmap. Replace with a one-sentence forward reference to the conclusion or delete the subsection and fold its content into the conclusion's future work item (i). **File: `sections/results.tex`, §4.8.**

**MF2. Add FIPER and FAIL-Detect as experimental baselines.**
Port both methods to the existing LIBERO rollout pool. Report failure AUC alongside PULSE in tab_safety_head_metrics. This requires:
- Implementing FIPER's random network distillation + action-chunk entropy baseline on OpenVLA and ACT hidden states
- Implementing FAIL-Detect's OOD approach (e.g., Mahalanobis distance or KDE on hidden states) without labeled failure data
- Adding rows to tab_safety_head_metrics with FIPER AUC, FAIL-Detect AUC, Linear Probe AUC, PULSE AUC
**Files: `tables/tab_safety_head_metrics.tex`, `scripts/train_eef_correction_mlp.py`, new scripts.**

**MF3. Remove all blue-colored "planned" text from the main body.**
`\color{blue}` appears in: Introduction (scope paragraph), §4.8 (physical robot section), Conclusion (future work (i)), §2 (experimental setup compute paragraph). Blue text in a research paper signals draft status. All future-work items must be written as regular prose. Convert to past tense for completed items, future tense without color for planned items.
**Files: `sections/introduction.tex`, `sections/results.tex`, `sections/conclusion.tex`, `sections/experimental_setup.tex`.**

**MF4. Reframe C2 ("Low-latency safety planning") as what it actually is.**
Replace the planning-horizon ablation framing with an honest description: "We show that single-step thresholding on the PULSE failure logit achieves statistically equivalent task success to K-sample feature-space probe evaluation (K=16) at 6.1× lower per-step latency, enabling deployment in latency-constrained control loops." Remove the word "MPPI" from C2 or add a clear parenthetical that the K-sample baseline is feature-space probe scoring, not trajectory-space MPPI.
**Files: `sections/introduction.tex`, `sections/abstract.tex`, all table captions referencing "MPPI".**

**MF5. Report the Random Noise result prominently, not buried.**
The fact that probe-free Gaussian noise is the highest-success row is a primary result, not a "note." Move this to a dedicated paragraph in §4.2 with equal prominence to the TOST equivalence paragraph. Add a sentence in the abstract acknowledging this. The current structure allows a reader to miss the most damaging finding entirely.
**Files: `sections/results.tex`, `sections/abstract.tex`.**

**MF6. Specify all hyperparameters for reproducibility.**
Add to the experimental setup: (a) learning rate for AdamW, (b) loss weights λ_f, λ_τ, λ_c, (c) number of training epochs (median/range before early stopping), (d) which layer of OpenVLA/ACT is used for hidden state extraction, (e) whether heads are trained jointly or sequentially.
**Files: `sections/experimental_setup.tex`, `sections/method.tex`.**

**MF7. Broaden to at least one additional LIBERO suite.**
Run PULSE on LIBERO-Goal or LIBERO-Object. 10 tasks, same framework. This is the minimum credibility threshold for generalization claims. The current zero-shot result (2 held-out tasks from LIBERO-Spatial) is within-distribution interpolation, not generalization.
**No existing file — requires new experiment and results section.**

---

### Important Fixes (Major Score Improvement)

**IF1. Reframe or remove the TTF head from C1.**
The TTF result is not a contribution on LIBERO-Spatial because a step counter matches it. Options:
- (a) Remove TTF from C1 entirely; keep it as an architectural detail reported in tab_safety_head_metrics
- (b) Reframe C1 as "failure detection AUC" only and move TTF to a footnote/appendix with the clock baseline
- (c) Rename C1 to "Hidden states encode failure risk" with AUC as the only supported submetric
**Files: `sections/introduction.tex`, `sections/abstract.tex`, `sections/results.tex`.**

**IF2. Ablate correction label construction.**
Show at minimum: (a) distribution of |Δp| labels, (b) cosine alignment between label direction and actual recovery direction from a small set of human demonstrations, (c) comparison with DTW-aligned labeling. This distinguishes "the latent doesn't encode direction" from "the labels don't encode direction."
**No existing file — requires new experiment.**

**IF3. Correct the "does not predict where to push" claim.**
Given the Appendix directionality probe shows hidden states do encode spatial failure direction early in rollouts, the conclusion "does not predict where to push" is too strong. Replace with: "On LIBERO-Spatial with the current label construction and gating thresholds, the correction head's output does not improve on random direction, likely because the gating fires after the early-rollout informative window has closed."
**Files: `sections/introduction.tex`, `sections/discussion.tex`, `sections/conclusion.tex`.**

**IF4. Add the SAFE controller as an explicit experimental baseline.**
SAFE is the closest prior work and is extensively discussed. It should appear as a row in the controller comparison table with its AUC and task success rate reported.
**Files: `tables/tab_final_pooled_results.tex` (new row), new experiment.**

**IF5. Reduce gating threshold ablation n from 5 to ≥20 per cell.**
Tab. A7 (gating ablation) with n=5 per cell is directional noise, not an ablation. Either (a) rerun with n=20 per cell and report results, or (b) remove the table entirely and replace with a sensitivity range statement based on the pooled results across the hyperparameter sweep.
**Files: `tables/tab_gating_ablation.tex`.**

**IF6. Add confidence intervals to main-text tables.**
Tab. 1 (tab_final_pooled_results) reports only point estimates and success counts. Add Wilson CIs (already computed in tab_wilson_paper_curated) as a column.
**Files: `tables/tab_final_pooled_results.tex`.**

**IF7. Address the 40% intervention rate directly in the Discussion.**
Add a paragraph in §5 (Discussion) confronting the 40% intervention rate and its implications for the "safety overlay" framing. Acknowledge that a controller modifying actions at 40% of timesteps is a co-policy. Reframe accordingly.
**Files: `sections/discussion.tex`.**

---

### Minor Fixes (Clarity and Polish)

**mF1. Abstract length.** The abstract exceeds ~340 words with multiple nested parentheticals. Target ≤250 words. Prioritize: AUC gain, latency result, negative correction result, simulation-only scope. Cut: TTF specific numbers (move to intro), TOST p-values (move to results), calibration ECE (move to results), Appendix cross-references.

**mF2. Pearson p precision.** The adaptive gating correlation is stated as p<10⁻⁶ in one location and p=5.2×10⁻⁷ in the data audit. Unify to p=5.2×10⁻⁷ throughout.

**mF3. Keywords.** The keywords are generic ("failure detection, latency-efficient safety probes, inference-time steering, vision-language-action models"). Replace with: "latent failure detection, robot policy probing, inference-time intervention, LIBERO benchmark, imitation learning safety."

**mF4. MPPI citation clarification.** Add a parenthetical in §2 and §4 clarifying the K-sample implementation is feature-space probe scoring, not Williams et al. (2017)'s gradient-free trajectory optimization in action space.

**mF5. Missing figure.** `figures/27_pipeline_overview.png` is referenced in the method section but not present in the figures directory listing. Confirm the file exists and is included in the compiled PDF.

**mF6. Acknowledgments placeholder.** `% \acknowledgments{TODO for camera-ready.}` is commented out with "TODO for camera-ready" — this is fine for anonymous submission but should be noted on the checklist.

**mF7. "Latent Stop" baseline.** Tab. 1 includes "Latent Stop" at 24.44% success (176/720, n=720). This baseline is never defined in the experimental setup section. What is Latent Stop? Add a definition.

**mF8. Hyperparameter averaging note.** The pooled table "averages over the sweep" (correction gain α∈{0.10, 0.12, 0.15, 0.18}, etc.). Averaging over hyperparameter sweep configurations conflates best-configuration performance with average-configuration performance. The table caption should explicitly state this is average over sweep, not best-found hyperparameter performance.

**mF9. ACT action chunk correction ambiguity.** For ACT with 8-step action chunks: specify which step(s) of the chunk receive the correction. This is a reproducibility gap.

**mF10. Passive-voice "cut_content.tex" entries.** The cut_content.tex file contains commented-out sections including "CUT 4: Non-overlap statement paragraph from related_work.tex — Reason: Defensive framing that weakens the paper." Including this reasoning in the submitted repository is fine for internal purposes, but should not be in the submission archive (it reveals authorial intent that should be transparent in the paper itself, not as a comment explaining why a defensive statement was removed).

---

## PHASE 6 — FINAL CoRL ASSESSMENT

### Reviewer Scores (1–5 scale, 5=best)

| Dimension | Score | Justification |
|---|---|---|
| **Originality** | 2/5 | MLP over linear probe for failure detection is incremental over SAFE. TTF is undermined by clock baseline. Correction head fails. Latency result is trivial. |
| **Technical Quality** | 3/5 | Statistics are rigorous (TOST, Holm correction, Wilson CIs). Ablations are partially complete. Critical hyperparameters missing. Label construction is flawed and unanalyzed. |
| **Empirical Rigor** | 2/5 | One simulator, one task suite, no real robot, key baselines (FIPER, FAIL-Detect, SAFE controller) omitted. Random noise beats all methods. Gating ablation n=5. |
| **Clarity** | 3/5 | Honest about negative results (unusual and commendable). Abstract too long and dense. Blue-colored future work in results section is editorially inappropriate. |
| **Reproducibility** | 3/5 | Code/data release promised. Claim-evidence map and audit reports are exemplary. But learning rate, loss weights, layer selection, and chunk-correction ambiguity are missing. |
| **Significance** | 2/5 | If the best baseline requires no probe, and the correction head fails, and the latency result is trivial, and the benchmark is single-environment simulation, the method's practical significance is unclear. |

### Estimated Outcome: **WEAK REJECT / BORDERLINE**

The paper is unusually honest about its own limitations, which is rare and scientifically admirable. However, honesty about failures does not substitute for positive evidence. After accounting for all the disclosures, the remaining positive contributions are:

1. MLP gives +0.047/+0.101 AUC over linear probe for failure detection on one simulation benchmark.
2. Single-step inference is faster than K-sample inference on the same function (trivially true).

These contributions are insufficient for CoRL acceptance in their current form. The paper needs real-robot results, competitive baselines, and at minimum one additional simulation environment.

### Most Likely Reviewer Consensus

- **Reviewer 1 (methodology-focused):** Weak Reject. Missing baselines, single benchmark, missing hyperparameters.
- **Reviewer 2 (applications-focused):** Reject. No real robot, placeholder in results section, safety framing not justified.
- **Reviewer 3 (statistics-focused):** Borderline. Statistical framework is rigorous; honesty about negative results is commendable. But scale and scope are inadequate.
- **Meta-reviewer consensus:** Weak Reject. Revise and resubmit with real-robot results and competitive baselines.

### Best Alternative Venue If Not CoRL Main Track

| Venue | Fit | Why |
|---|---|---|
| **CoRL 2026 Workshop on Safe Robot Learning** | High | The mechanistic analysis and honest negative results are exactly what workshop discussions are for |
| **RA-L (IEEE Robotics and Automation Letters)** | High | RA-L accepts simulation-only empirical work with appropriate scope framing; the statistical rigor is RA-L quality |
| **IROS 2026** | Moderate | Good match for the detection/latency results; less emphasis on significance threshold |
| **NeurIPS 2026 Workshop on Robot Foundation Models** | High | The mechanistic interpretation angle and honest negative results fit workshop format well |
| **ICLR 2026 Workshop on Mechanistic Interpretability** | Moderate | The directionality probe and cross-architecture analysis fit; robot learning frame is secondary |

### Most Important Single Experiment That Could Change the Decision

**Physical robot validation on SO-101 (or any real robot):** 50 episodes/mode/task with blind success labeling, showing PULSE failure detection AUC > 0.70 on real-world hidden states from a sim-trained probe. This single experiment would elevate the paper to borderline-accept because:
1. It validates the core detection claim beyond simulation
2. It provides a credible sim-to-real story for the 40% intervention rate
3. It demonstrates that the paper is finished, not a work-in-progress

Without real-robot validation, no amount of additional simulation analysis will move a CoRL reviewer from Weak Reject to Accept.

---

## FORMATTING AUDIT

### Issues Found

| Issue | Location | Severity | Action |
|---|---|---|---|
| `\fbox{\parbox{...}}` placeholder in Results | `sections/results.tex` §4.8 | **Critical** | Delete entire placeholder block |
| `\color{blue}` text in main body | Introduction scope ¶, §4.8, §5 limitations, Conclusion future work, §2 compute ¶ | **High** | Convert to regular prose or delete |
| Abstract >340 words | `sections/abstract.tex` | **High** | Cut to ≤250 words |
| "Latent Stop" undefined in experimental setup | `sections/experimental_setup.tex` | **High** | Add definition |
| Loss weights λ_f, λ_τ, λ_c not specified | `sections/method.tex` | **High** | Add to Eq. 1 caption or training paragraph |
| Learning rate not stated | `sections/experimental_setup.tex` | **High** | Add AdamW lr=X to safety probe paragraph |
| Hidden state layer not specified | `sections/experimental_setup.tex` | **High** | Specify which layer (last hidden, penultimate, etc.) |
| ACT chunk correction ambiguity | `sections/method.tex` Eq. 5 | **High** | Specify which chunk step receives correction |
| Gating ablation n=5 in main appendix | `tables/tab_gating_ablation.tex` | **Moderate** | Increase n to ≥20 or remove from submitted appendix |
| Pearson p string inconsistency (8×10⁻⁷ vs. 5.2×10⁻⁷) | `sections/results.tex` §4.3 | **Moderate** | Unify to 5.2×10⁻⁷ |
| Keywords too generic | `main.tex` | **Low** | Revise to specific terms |
| Hyperparameter sweep averaging not flagged in Tab. 1 caption | `tables/tab_final_pooled_results.tex` | **Moderate** | Add "averaged over hyperparameter sweep" to caption |
| OpenVLA AUC note in DATA_AUDIT: 0.832 (trainer) vs. 0.925 (threshold replay) — resolved but inconsistency should be documented in paper | Appendix | **Low** | Already resolved in data audit; no action needed if single AUC pair is correct |
| `% \acknowledgments{TODO for camera-ready.}` in main.tex | `main.tex` line 46 | **Low** | Fine for anon submission; checklist item |
| `\usepackage[final]{corl_2026}` commented out | `main.tex` line 4 | **Low** | Switch to final for camera-ready |

---

## UNNECESSARY FLUFF IDENTIFICATION

The following content adds length without adding scientific value and should be cut or compressed:

### High-Priority Cuts

**FLUFF-1: Failure subtype taxonomy in Experimental Setup (§2, "Failure definition" paragraph)**
The four-category taxonomy (collision / grasp slip / wrong-receptacle / timeout) is explicitly labeled "not used as probe labels" and is described as "reader context only." It occupies ~8 lines including the parenthetical n=650 contact-logging note and the median step 107. None of this information affects any experimental result. Cut entirely; replace with: "Any rollout where the LIBERO success predicate does not fire by step 300 is labeled failure."

**FLUFF-2: Concrete Task 0 walkthrough paragraph in Experimental Setup**
"Concrete example (LIBERO-Spatial Task 0, 'pick up the akita_black_bowl...')": this paragraph (~8 lines) describes one specific task's success/failure pattern in narrative form. It adds zero information not already present in the filmstrip figure in the Appendix. Cut entirely; the Appendix figure provides better evidence with less space. Keep a one-line cross-reference to the figure: "See Appendix Fig. X for a paired success/failure rollout on Task 0."

**FLUFF-3: Three "Improving the correction head" bullet points in Conclusion**
Recovery-conditioned labels, multi-step correction planning, task-specific correction heads — these are reasonable suggestions but are pure speculation with no preliminary evidence. At CoRL, three bullet points of "here's what we would do next" is recognized as filler. Cut to one sentence: "Recovery-conditioned labels, multi-step planning, and task-specific heads are candidate next steps for improving correction fidelity." Save space for the discussion of what is actually known.

**FLUFF-4: Repeated negative-result disclaimers**
The paper states the correction-direction result is negative in: the abstract (2× mentions), the introduction (1 paragraph), §4.5 (the mechanism subsection), the discussion (1 paragraph), and the conclusion (2× mentions). This is five separate restatements of the same finding. Pick two locations (abstract and §4.5) and eliminate the others.

**FLUFF-5: §4.8 "Physical Robot Evaluation" subsection**
As noted in MF1 — delete entirely. It contains no results.

**FLUFF-6: "Scope" paragraph in Introduction**
The scope paragraph ("The PULSE probe template is architecture-agnostic... All quantitative claims in this submission are empirically evaluated on LIBERO-Spatial only...") restates what C1/C2/C3 already imply and what the experimental setup will specify. Move the key scope statement ("LIBERO-Spatial only") to the abstract as a one-clause hedge, and delete the scope paragraph.

### Medium-Priority Cuts

**FLUFF-7: PCA figure pair in §4.1**
The PCA visualization (Fig. 2a) shows success/failure separate in the first two principal components. This is a qualitative illustration with no quantitative claim attached. The separation is already implied by the AUC. Consider removing from the main body and moving to Appendix.

**FLUFF-8: Directionality probe paragraph in §4.5**
The "Directionality probe: where will the rollout fail?" paragraph is an interesting mechanistic analysis, but it raises a question ("does the latent encode failure direction?") and provides a qualified answer ("yes, but only early in the rollout, and only before kinematics commit") that ultimately supports the negative result without changing any contribution. This can be compressed from ~15 lines to 4 lines without loss of scientific content, with the full Table A8 (tab_spatial_failure) remaining in the Appendix.

**FLUFF-9: MPPI hyperparameter specification in Experimental Setup**
"MPPI uses K=16 samples per step (default in `scripts/eval_tuning.py`, `--mppi-samples 16`), feature-space perturbation σ=0.01, softmin temperature τ=5.0." The backtick code reference in the main text is unusual for an academic paper. Remove the script/flag reference; keep the numerical values.

---

## NEGATIVE RESULTS: CHANGE PLAN

The paper reports four negative results. The question is which should be changed (i.e., reframed, removed, or replaced with positive evidence) and which should be kept as published negative results.

### Negative Result Assessment

| Negative Result | Keep As-Is? | Change? | Rationale |
|---|---|---|---|
| **NR1: Correction direction does not outperform latent jiggle** | **Keep** | No | This is the most scientifically valuable result. It is reproducible, clearly set up, correctly interpreted, and changes the field's understanding of what latent failure representations encode. Removing it would be scientifically dishonest. |
| **NR2: TTF head is equivalent to a step counter** | **Change framing** | Yes | Currently listed as part of C1, which is confusing. Downgrade TTF to an architectural detail. Remove from the contribution bullets entirely. Keep the clock baseline in the Appendix but do not present TTF as a contribution. |
| **NR3: Random noise (no probe) achieves highest success rate** | **Change placement** | Yes | Currently buried in a "note" paragraph. Should be elevated to a primary result that motivates the paper's reframing toward latency and mechanistic understanding rather than success-rate improvement. Make it the first finding in §4.2, not a footnote. |
| **NR4: Cross-architecture linear transfer fails** | **Keep** | No | This is informative and correctly placed in §4.7. The authors correctly distinguish it from the "architecture-agnostic template" claim (C3). No change needed. |

### Specific Reframing Actions for Negative Results

**NR2 (TTF):** Remove TTF from Contribution C1. Rewrite C1 as: "**Hidden states encode failure risk.** We train an MLP probe on frozen policy hidden states and achieve failure-detection AUC 0.83/0.89 on OpenVLA/ACT, gaining +0.047/+0.101 over a linear probe baseline. This is the primary empirical finding. We also train a TTF regression head (r=0.86), but on LIBERO-Spatial where failure rollouts run to a fixed episode length, a raw step counter achieves r≈1.0; the TTF head's value on variable-length-failure settings is future work (§A.2)." This keeps the honest disclosure without listing a debunked result as a contribution.

**NR3 (Random Noise):** Move the random-noise finding to the top of §4.2, before the TOST equivalence results. Suggested placement: "Before reporting the PULSE controller results, we note the key calibration point: a probe-free Gaussian perturbation (σ=0.05, every step, no PULSE involvement) achieves the highest success rate in our evaluation at 53.98% on the OpenVLA sub-pool (n=5,080). This establishes that LIBERO-Spatial's task-success ceiling is essentially the base policy's ceiling, and the contribution of PULSE is inference-cost reduction and mechanistic understanding of failure representations, not a task-success lift over any perturbation baseline." Then proceed to the TOST equivalence results.

---

*Review completed by adversarial CoRL reviewer. All findings based on full reading of main.tex, all section files, key tables, and internal audit documents (DATA_AUDIT_REPORT.md, CLAIM_EVIDENCE_MAP.md, SIM_PRIMARY_CLAIM.md).*
