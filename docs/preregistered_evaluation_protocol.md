# Pre-Registered Evaluation Protocol & Hypothesis Registry
**Project:** Empirical Evaluation of Pedestrian Crossing Intention & Relational Interaction Modeling  
**Date of Pre-Registration:** September 28, 2026  
**Status:** Pre-Registered Prior to Clean Benchmark Execution  

---

## 1. Core Research Question
> **"Do vulnerable-road-user (VRU) crossing-intention models actively rely on surrounding traffic participants, and is that reliance driven by behavioral interaction rather than mere spatial proximity or model capacity?"**

---

## 2. Experimental Parameters & Analysis Rules (Locked In)

### 2.1 Benchmark Splits & Datasets
* **PIE (Pedestrian Intent Estimation):**
  * Official Benchmark Partition:
    * **Train:** Set 01, Set 02, Set 04
    * **Val:** Set 05, Set 06
    * **Test:** Set 03 exclusively (719 standard benchmark pedestrian tracks)
  * Observation length: $T_{obs} = 16$ frames (~0.53s at 30 fps)
  * Time-to-Event (TTE) prediction horizon: $T_{TTE} \in [30, 60]$ frames (1.0s to 2.0s before the event point)
* **JAAD (Joint Attention in Autonomous Driving):**
  * Official video-disjoint partitions: `jaad_beh` (behavior-annotated, 117 test videos) and `jaad_all` (126 test videos), reported separately.
  * Observation length: $T_{obs} = 16$ frames; $T_{TTE} \in [30, 60]$ frames.
* **Separation Rule:** Datasets are analyzed and reported strictly independently. PIE and JAAD are never pooled into a single composite score.

### 2.2 Replicates, Random Seeds & Inference Unit
* **Random Seeds:** 5 fixed independent seeds: `[42, 100, 2024, 7, 13]`.
* **Primary Unit of Analysis & Inference:** The **Pedestrian Track** (cluster-level). Individual sliding windows from the same pedestrian track are correlated; therefore, all inferential confidence intervals and hypothesis tests use **cluster bootstrap resampling ($B=1000$) clustered on pedestrian track ID (`ped_id`)**.
* **Multiple Testing Correction:** Family-wise error rate across hypotheses H1–H4 controlled via the **Holm-Bonferroni step-down procedure** ($\alpha = 0.05$).
* **Minimum Stratum Sample Size:** $N_{min} = 25$ independent pedestrian tracks. Any stratum with $<25$ tracks is reported descriptively only, with explicit disclosure of sample insufficiency.

### 2.3 Coordinate System & Metric Honesty
* **Monocular Image Space Disclosure:** Inputs from PIE and JAAD are monocular dashcam recordings. Unless explicitly recovered via metric depth sensors or ground-plane homography, spatial distances, velocities, and closing-speed time-to-event surrogates are defined within **normalized image coordinates** $[0, 1]$. 
* **Terminology:** Perturbations are termed **"kinematically consistent within normalized feature space"** rather than "physically metric," reflecting data realities accurately.

---

## 3. The 4 Pre-Registered Hypotheses

### Hypothesis 1 (H1): Neighbor Availability Measurement
* **Prediction:** Annotated non-ego dynamic traffic participants ($K \ge 1$) exist in a minority of test windows on PIE Set 03 and JAAD_beh.
* **Metric:** Empirical distribution and percentage of windows and tracks across neighbor strata:
  $$\text{Strata: } K=0 \text{ (isolated)}, \quad K=1 \text{ (dyadic)}, \quad K \ge 2 \text{ (multi-agent)}$$
* **Falsification / Outcome Interpretation:**
  * If $K \ge 1$ constitutes $>50\%$ of benchmark windows $\implies$ The sparsity hypothesis is refuted; benchmarks have abundant multi-agent interaction.
  * If $K \ge 1$ constitutes $<50\%$ $\implies$ Sparsity is confirmed, and interaction claims must be interpreted within this empirical boundary.

---

### Hypothesis 2 (H2): Stratified Utility of the Interaction Branch
* **Prediction:** An interaction-aware model outperforms its matched no-neighbor ablation ($\text{Full} - \text{NoNeighbor}$) primarily in $K \ge 1$ strata, and shows no statistically significant gain on $K=0$.
* **Metric:** Difference in ROC-AUC ($\Delta \text{AUC}$) and Binary F1 ($\Delta \text{F1}$) evaluated separately per stratum:
  $$\Delta M_K = M_{\text{Full}}(K) - M_{\text{NoNeighbor}}(K)$$
* **Statistical Criterion for Support:**
  * 95% cluster-bootstrap confidence interval for $\Delta M_{K \ge 1}$ strictly excludes zero ($CI_{lower} > 0$).
  * 95% cluster-bootstrap confidence interval for $\Delta M_{K=0}$ contains zero.
* **What Counts Against It:**
  * **Significant gain on $K=0$:** Indicates performance improvement stems from general model capacity or regularizer effects rather than neighbor modeling.
  * **No significant gain across any stratum:** The interaction module is dormant on these benchmarks. This constitutes an empirical **null result** to be reported transparently.

---

### Hypothesis 3 (H3): Targeted vs. Matched-Control Neighbor Removal
* **Prediction:** Removing the model-attributed primary interacting neighbor induces a larger reduction in predicted crossing probability than removing a distance-, speed-, and heading-matched control neighbor.
* **Intervention & Matching Protocol (for samples with $N \ge 2$ valid neighbors):**
  * Target neighbor: Top-attributed neighbor by the interaction attention weights.
  * Matched control candidate: Neighbor satisfying:
    $$|\text{dist}_{ctrl} - \text{dist}_{targ}| \le 0.15 \cdot \text{dist}_{targ} \quad (\text{or } \pm 0.05 \text{ normalized})$$
    $$|\text{speed}_{ctrl} - \text{speed}_{targ}| \le 0.20 \cdot \text{speed}_{targ}$$
    $$|\text{heading}_{ctrl} - \text{heading}_{targ}| \le 30^\circ \quad (\pi/6 \text{ rad})$$
  * Selection Bias Control: A random pick among all candidates satisfying matching tolerances.
  * Oracle Sensitivity Control: Exhaustive removal of each neighbor in turn to identify empirical maximum sensitivity $\Delta p_{oracle}$.
* **Metric:** Paired difference per sample:
  $$\delta_{attribution} = |\Delta p(y_{pred} \mid \text{remove target})| - |\Delta p(y_{pred} \mid \text{remove matched control})|$$
* **Statistical Criterion for Support:**
  * Median paired difference is positive ($\text{Median}(\delta_{attribution}) > 0$).
  * Track-clustered paired Wilcoxon signed-rank test achieves $p < 0.05$ after Holm-Bonferroni correction.
  * 95% bootstrap CI of the median excludes zero.
* **What Counts Against It:**
  * $\delta_{attribution} \approx 0$ or CI includes zero $\implies$ The model responds merely to general agent presence or spatial proximity, and the attention attribution is uninformative.

---

### Hypothesis 4 (H4): Response to Normalized Kinematic Approach Urgency
* **Prediction:** For closing neighbors ($v_{closing} > 0$), scaling neighbor approach velocity by factor $\alpha$ monotonically reduces the predicted crossing probability ($p_{cross}$), whereas separating neighbors ($v_{closing} \le 0$) and placebo feature perturbations do not.
* **Underlying Behavioral Assumption (Stated Explicitly):** An approaching vehicle or dynamic obstacle closing in on the pedestrian's crossing corridor increases collision risk, reducing the probability of human crossing intention.
* **Kinematic Scale Grid (Locked):**
  $$\alpha \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$$
* **Kinematic Co-Rescaling Formula:**
  $$p(t) = p(T_{obs}) - \alpha \cdot (p(T_{obs}) - p(t)), \quad v(t) = \alpha \cdot v(t)$$
* **Closing Vector Definition:**
  $$\vec{r} = \vec{p}_{ped} - \vec{p}_{nbr}, \quad \vec{v}_{rel} = \vec{v}_{ped} - \vec{v}_{nbr}, \quad v_{closing} = -\frac{\vec{r} \cdot \vec{v}_{rel}}{\|\vec{r}\| + \epsilon}$$
  * If $v_{closing} > 0 \implies$ Closing / approaching.
  * If $v_{closing} \le 0 \implies$ Separating / diverging.
* **Metric:**
  * Per-sample Spearman rank correlation $\rho(\alpha, p_{cross})$.
  * Share of closing samples exhibiting expected negative correlation ($\rho < -0.50$).
  * Comparison against placebo perturbation (scrambled agent position coordinates).
* **Statistical Criterion for Support:**
  * Proportion of samples with expected sign significantly exceeds placebo perturbation rate via McNemar's test ($p < 0.05$).
* **What Counts Against It:**
  * Flat, non-monotonic, or inconsistent $\rho$ distribution $\implies$ Feature sensitivity without behavioral or directional consistency.

---

## 4. Distributional Validity Check (Out-of-Distribution Monitoring)
* For every intervention in H3 and H4, we measure the distance of perturbed inputs from the training distribution:
  1. Proportion of perturbed feature values falling outside the empirical $[min, max]$ training range.
  2. Mahalanobis distance relative to the training feature mean and covariance:
     $$D_M(\mathbf{x}_{pert}) = \sqrt{(\mathbf{x}_{pert} - \boldsymbol{\mu}_{tr})^T \boldsymbol{\Sigma}_{tr}^{-1} (\mathbf{x}_{pert} - \boldsymbol{\mu}_{tr})}$$
* Results are reported alongside H3 and H4 to distinguish between learned behavioral sensitivity and out-of-distribution model degradation.

---

## 5. Scope & Boundary Conditions
1. **Single-Model vs. Multi-Model:** Evaluated on the implemented interaction-aware architecture and baselines. Third-party models (e.g. PCPA) are evaluated where runnable code exists; unreleased third-party models (e.g. PIT, ESIA) are not simulated or overclaimed.
2. **TITAN:** Reserved for separate, explicitly labeled exploratory analysis once raw Tokyo annotations are unpacked and verified.
3. **Language Standards:** Prohibited terms in all draft text: "first," "airtight," "proves causality," "revolutionary." All claims are indexed strictly to the empirical outcomes of H1–H4.
