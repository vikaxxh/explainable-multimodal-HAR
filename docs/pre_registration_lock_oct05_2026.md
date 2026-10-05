# Pre-Registration Lock & Scientific Protocol Commitment
## Controlled Counterfactual Diagnostics for Pedestrian Intention Prediction

**Candidate**: Vikash Paigamber (Scholar ID: `25212031117`)  
**Department**: Department of Computer Science & Engineering, MANIT Bhopal  
**Target Venues**: IEEE Transactions on Intelligent Vehicles (T-IV) / IEEE T-ITS / Premier Transportation Safety Conferences  
**Date of Pre-Registration Lock**: **October 5, 2026 (11:50 IST)**  
**Document Status**: **LOCKED PRIOR TO BLIND TEST SET (SET 03) EVALUATION**  

---

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PRE-REGISTRATION AUDIT HASH & COMMITMENT                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Protocol Status:    LOCKED & TIME-STAMPED BEFORE TEST SET (SET 03) EVALUATION                                   │
│ Target Test Split:  PIE Official Set 03 (719 Tracks | 294,219 Frames | 68,060 Sliding Windows)                   │
│ Locked Checkpoint:  Epoch 21 (experiments/checkpoints/pie/checkpoint_best.pt)                                    │
│ Selection Metric:   Minimum Total Validation Loss on Sets 05 & 06 (0.9537)                                      │
│ Pre-Registered MMES: ΔAUC >= +0.015 (+1.5% ROC-AUC over certified isolated baseline)                            │
│ Scientific Stance:  All outcomes (Positive, Proximity Shortcut, or Null Result) are valid and publishable.      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Checkpoint Selection & Early-Stopping Rule (Locked Today)

To prevent post-hoc cherry-picking or $p$-hacking, we formally record our checkpoint selection rule **before running any inference on held-out test Set 03**:

1. **The Checkpoint Rule**:
   * The model evaluated on held-out Set 03 must be selected strictly by **minimum total validation loss** on the validation split (Sets 05 & 06).
   * **Selected Model**: **Epoch 21** (`checkpoint_best.pt`).
   * **Frozen Validation Metrics (Epoch 21)**:
     * Validation Total Loss: **`0.9537`** (all-time minimum across all completed epochs).
     * Validation Crossing Accuracy: **`89.0%`**.
     * Validation Pedestrian Action Accuracy: **`63.6%`**.
2. **Explicit Rejection of Late-Epoch Models**:
   * Epochs 22 through 37 are explicitly **rejected** for test evaluation due to diagnosed capacity saturation:
     * Training loss dropped to `0.3738` (97.3% training crossing accuracy).
     * Validation loss inflated to `1.4118` due to logit overconfidence on boundary cases.
   * Early stopping was functionally triggered at **Epoch 31** (10 consecutive epochs without validation loss improvement past Epoch 21).
   * **Rule Commitment**: No results from Epoch 37 or other late epochs will be substituted for test reporting.

---

## 2. Pre-Registered Hypotheses (H2 – H4) & Acceptance Criteria

All inferential tests are evaluated on held-out **Set 03 (68,060 sequences across 719 pedestrian tracks)**. Because overlapping sliding windows are non-i.i.d., all standard errors and confidence intervals use **Track-Clustered Bootstrapping** (1,000 resamples clustered at the 719 pedestrian track level).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HYPOTHESIS DECISION CRITERIA SUMMARY                                      │
├────┬─────────────────────────────┬──────────────────────────────────────────┬───────────────────────────────────┤
│ Hyp│ Focus                       │ Primary Statistical Criterion            │ Substantive / Physical Criterion  │
├────┼─────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────┤
│ H2 │ Interaction-Specific Utility│ Two-sided 95% Bootstrap CI(ΔAUC) > 0     │ ΔAUC >= +0.015 (MMES Threshold)   │
│    │ (Full vs. Isolated Baseline)│ at Holm-Bonferroni α = 0.05              │                                   │
├────┼─────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────┤
│ H3 │ Attribution Faithfulness    │ Paired Wilcoxon Signed-Rank Test on      │ Median attribution difference     │
│    │ (vs. Matched Control Agent) │ track medians δ_i (p_Holm < 0.05)        │ δ > 0 over matched candidate      │
├────┼─────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────┤
│ H4 │ Kinematic Urgency           │ Linear Mixed-Effects Model fixed slope   │ Randomized-bearing placebo slope  │
│    │ Concordance (v_closing grid)│ β_α < 0 (p_Holm < 0.05, D_M <= Q_0.95)   │ β_placebo ≈ 0 (p >= 0.05)         │
└────┴─────────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────┘
```

### Detailed Decision Rules:

* **Hypothesis H2 (Interaction-Specific Predictive Utility)**:
  * Metric: $\Delta\text{AUC} = \text{ROC-AUC}_{\text{multi}} - \text{ROC-AUC}_{\text{isolated}}$ on held-out Set 03.
  * Baseline Anchor: Certified isolated baseline (`checkpoint_isolated.pt`, **87.73% ROC-AUC**, 64.26% Crossing F1).
  * **Pass Condition**: $95\%\text{ CI}(\Delta\text{AUC}) > 0$ **AND** $\Delta\text{AUC} \ge +0.015$.
  * **Fail / Null Condition**: If $\Delta\text{AUC} < +0.015$ or the $95\%$ CI includes zero, H2 is declared **UNSUPPORTED (Null Result)**.

* **Hypothesis H3 (Counterfactual Attribution Faithfulness)**:
  * For each frame where top-attended neighbor $N_1$ exists, identify candidate control neighbor $N_{\text{ctrl}}$ matched within:
    * Distance: $\pm 15\%$
    * Relative Speed: $\pm 20\%$
    * Relative Heading: $\pm 30^\circ$
  * Compute attribution difference $\delta_{i,t} = A(N_1) - A(N_{\text{ctrl}})$, aggregated to track medians $\bar{\delta}_i$.
  * **Pass Condition**: Paired Wilcoxon signed-rank test across 719 tracks yields $p_{\text{Holm}} < 0.05$ with $\text{median}(\bar{\delta}_i) > 0$.
  * **Fail Condition**: If $p \ge 0.05$, attention is declared **Unfaithful** (driven by spatial proximity rather than interactive priority).

* **Hypothesis H4 (Kinematic Urgency Concordance)**:
  * Scale relative closing velocity $v_{\text{closing}} = -\vec{v}_{\text{rel}} \cdot \vec{u}_{\text{rel}}$ across 11-point grid:
    $$\alpha \in \{0.50, 0.65, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.50, 2.00\}$$
  * Fit Linear Mixed-Effects Model: $P(\text{cross})_{i,t,\alpha} = \beta_0 + \beta_{\alpha}\alpha + u_i + \epsilon_{i,t,\alpha}$.
  * Constrained by Ledoit-Wolf shrinkage Mahalanobis distance ($D_M \le Q_{0.95}$) to ensure counterfactual in-distribution validity.
  * **Pass Condition**: Fixed effect slope $\beta_{\alpha} < 0$ ($p < 0.05$) AND placebo control slope $\beta_{\text{placebo}} \approx 0$ ($p \ge 0.05$).
  * **Fail Condition**: $\beta_{\alpha} \ge 0$ indicates model fails to obey physical collision closing laws.

---

## 3. The Three Legitimate Scientific Interpretations After H2

We pre-commit to publishing whichever outcome is observed without shifting criteria:

```
                                      ┌────────────────────────┐
                                      │   EXECUTE H2 ON SET 03 │
                                      └───────────┬────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
             ΔAUC >= +0.015 (Significant)                      ΔAUC < +0.015 (Null Effect)
                         │                                                 │
            ┌────────────┴────────────┐                                    │
            ▼                         ▼                                    ▼
       H3 & H4 PASS              H3 or H4 FAIL                         OUTCOME C:
        OUTCOME A:                OUTCOME B:                     "Interaction Modules
   "Interaction Modeling    "Proximity Shortcut Exposed:           Add Little Over
    Provides Genuine          Model Gains Accuracy via              Pedestrian's Own
    Behavioral Signal"        Spatial Artifacts, Not Physics"       Kinematic Cues"
   [Lead with Protocol &     [Flagship Critique Paper on            [Diagnostic Audit Paper
    Interaction Mechanism]    Popular Graph Attention Flaws]         on Model Complexity]
```

* **Outcome A (Interaction is Genuine)**:
  * Multi-agent modeling provides both statistical gain and physical adherence. Lead with the diagnostic protocol as the first certified evaluation of relational attention.
* **Outcome B (Proximity Shortcut Exposed)**:
  * The model gains statistical accuracy, but fails counterfactual matched-pair attribution or kinematic scaling. This is a high-impact critique proving that benchmark gains in literature stem from proximity shortcuts rather than right-of-way negotiation.
* **Outcome C (Interaction Adds Little / Null Result)**:
  * $\Delta\text{AUC}$ fails to meet the $+0.015$ MMES threshold. The pedestrian's own kinematics, pose articulation, and visual context account for virtually all predictive capacity. Framed as an empirical analysis paper exposing overparameterization in urban VRU prediction.

---

## 4. Comprehensive Seven-Step Action Roadmap

```
┌───────┬───────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ STEP  │ TIMEFRAME                 │ ACTIONABLE SPECIFICATION & DELIVERABLES                                │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 1│ TODAY (Oct 5, 2026)       │ • Lock Pre-Registration Document & git commit to repository.           │
│       │                           │ • Terminate DGX job at Epoch 37; verify Epoch 21 checkpoint on disk.   │
│       │                           │ • Freeze test Set 03 evaluation until protocol is committed.           │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 2│ Week 1 (Regularization)   │ • Implement EarlyStopping(patience=8, metric='val_auc').               │
│       │                           │ • Increase weight decay (1e-4 -> 1e-2) and dropout (0.1 -> 0.3).       │
│       │                           │ • Add graph edge dropout (p=0.2) to prevent topology memorization.     │
│       │                           │ • Train across 3 random seeds (seeds 42, 100, 2024).                   │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 3│ Fair Baseline Retraining  │ • Retrain Isolated Baseline under identical regularization recipe,     │
│       │                           │   same tuning budget, same random seeds, and same checkpoint rule.     │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 4│ Parked-Car Confound Audit │ • Algorithmically label neighbors: static (speed < 0.5 m/s across track)│
│       │                           │   versus dynamic (speed >= 0.5 m/s).                                   │
│       │                           │ • Re-run H1 density audit with parked vehicles excluded.               │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 5│ Protocol Validation       │ • Train Proximity-Only Variant (edges carry Euclidean distance only).  │
│       │                           │ • Train Urgency-Aware Variant (edges carry closing velocity & TTC).    │
│       │                           │ • Shuffled-Neighbor Control: Attach neighbors from random pedestrians  │
│       │                           │   to test if attention degrades to zero.                               │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 6│ Single Test Evaluation    │ • Evaluate frozen checkpoints on Set 03 once and report all metrics    │
│       │                           │   (Accuracy, ROC-AUC, PR-AUC, Crossing F1, Precision, Recall).         │
├───────┼───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Step 7│ Cross-Dataset & Writing   │ • Replicate protocol on JAAD (jaad_beh official split).                │
│       │                           │ • Draft conference paper first (IEEE IV / ITSC), then extend to        │
│       │                           │   flagship journal (IEEE T-IV / IEEE T-ITS).                           │
└───────┴───────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

*This document represents the immutable pre-registration audit for the research project conducted by Vikash Paigamber at Maulana Azad National Institute of Technology (MANIT), Bhopal. Locked on October 5, 2026.*
