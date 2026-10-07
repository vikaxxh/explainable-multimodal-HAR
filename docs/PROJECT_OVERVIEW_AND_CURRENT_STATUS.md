# X-MIST / EMIT-HAR: Project Master Dossier & Current Status Report
## Explainable Multimodal Interaction-aware Spatio-Temporal Transformer for Pedestrian Intention Prediction

```
====================================================================================================
PROJECT METADATA & COMMITMENT RECORD
====================================================================================================
Author / Candidate:     Vikash Paigamber (Scholar ID: 25212031117)
Department:             Department of Computer Science & Engineering
Institution:            Maulana Azad National Institute of Technology (MANIT), Bhopal
Target Venues:          IEEE Transactions on Intelligent Vehicles (IEEE T-IV)
                        IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)
Current Date:           October 8, 2026
Protocol Status:        Pre-Registration Locked (Held-Out Test Set 03 Quarantined)
Git Branch / Head:      main (Commit 998fb24)
HPC Environment:        DGX A100 Supercomputer (manit.ac.in), Python env: emit-har
====================================================================================================
```

---

## 1. Executive Summary & Research Problem

Vision-based pedestrian intention prediction (deciding whether a roadside human will cross the roadway $1.0\text{s}$ to $3.0\text{s}$ before initiation) is critical for Autonomous Emergency Braking (AEB) and Level 4 urban driving systems. While naturalistic benchmarks such as PIE (*Rasouli et al., IJCV 2021*) and JAAD (*Rasouli et al., ICCVW 2017*) have spurred architectures utilizing Graph Neural Networks (GNNs) and Spatio-Temporal Transformers (e.g., PIT *Zhou et al., IEEE T-ITS 2023*; PedGraph+; IntentFormer), published literature is fraught with three unaddressed methodological vulnerabilities:

1. **The Spatial Proximity Confound**: Graph attention modules routinely reward physical proximity (identifying agents geometrically closest to the ego-vehicle or pedestrian) rather than behavioral/kinematic relevance (agents on a physical collision trajectory or actively negotiating priority).
2. **Static Obstacle & Parked Vehicle Artifacts**: Datasets filmed from urban roadways annotate parked, un-crewed vehicles as dynamic traffic participants. Models often artificially boost "interaction count" by tracking inanimate roadside cars that exert zero behavioral negotiation.
3. **Severe Class Imbalance & Evaluation Masking**: Urban driving is predominantly non-crossing (~80–85% negative frames). A degenerate baseline predicting constant negative intention attains ~82% accuracy while offering zero collision protection (0% recall, 0% F1). Standard accuracy metrics mask severe collapse.

### Methodological Mission
Rather than building an uncalibrated black-box claiming inflated "state-of-the-art" accuracy, **X-MIST** establishes an empirically unassailable, pre-registered experimental test vehicle to execute a **controlled counterfactual diagnostic protocol (Hypotheses H1–H4)**. We definitively test whether multi-agent relational graph attention extracts authentic physical interaction dynamics or merely exploits statistical shortcuts.

---

## 2. Pre-Registered Hypotheses & Acceptance Rules

The experimental protocol is locked prior to running inference on the blind test partition. All inferential evaluations are conducted on **Held-Out Test Set 03 (68,060 sliding windows across 719 pedestrian tracks)**. 

Because consecutive sliding windows share video frames and violate standard independent identically distributed ($i.i.d.$) assumptions, all statistical inferences, standard errors, and confidence intervals use **Track-Clustered Bootstrapping** (1,000 resamples clustered at the pedestrian track level).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  THE PRE-REGISTERED HYPOTHESIS SUITE                                   │
├────┬─────────────────────────────┬──────────────────────────────────────────┬──────────────────────────┤
│ Hyp│ Focus                       │ Primary Statistical Criterion            │ Substantive Rule         │
├────┼─────────────────────────────┼──────────────────────────────────────────┼──────────────────────────┤
│ H1 │ Neighbor Density Audit      │ Stratification across 3 levels:          │ Distinguish dynamic      │
│    │                             │ Track-level, Frame-level, and Static Parked│ traffic from parked cars │
├────┼─────────────────────────────┼──────────────────────────────────────────┼──────────────────────────┤
│ H2 │ Interaction Utility         │ Two-sided 95% Bootstrap CI(ΔAUC) > 0     │ ΔAUC >= +0.015 (+1.5%    │
│    │ (Multi-Agent vs. Isolated)  │ at Holm-Bonferroni α = 0.05              │ ROC-AUC over isolated)   │
├────┼─────────────────────────────┼──────────────────────────────────────────┼──────────────────────────┤
│ H3 │ Attribution Faithfulness    │ Paired Wilcoxon Signed-Rank Test on      │ Median attribution δ > 0 │
│    │ (vs. Matched Control Agent) │ track medians δ_i (p_Holm < 0.05)        │ over distance-matched    │
├────┼─────────────────────────────┼──────────────────────────────────────────┼──────────────────────────┤
│ H4 │ Kinematic Urgency           │ Linear Mixed-Effects Model fixed slope   │ β_α < 0 (p < 0.05),      │
│    │ Concordance (v_closing grid)│ β_α < 0 under Ledoit-Wolf D_M <= Q_0.95  │ β_placebo ≈ 0 (p >= 0.05)│
└────┴─────────────────────────────┴──────────────────────────────────────────┴──────────────────────────┘
```

### The Three Legitimate Scientific Interpretations
- **Outcome A ($\Delta\text{AUC} \ge +0.015$, H3 & H4 Pass)**: Relational graph modeling provides genuine behavioral signal grounded in physical closing kinematics. (Flagship paper on interaction modeling).
- **Outcome B ($\Delta\text{AUC} \ge +0.015$, H3 or H4 Fail)**: Proximity shortcut exposed. GNNs gain accuracy from geometric co-occurrence rather than physics. (Critical diagnostic critique paper).
- **Outcome C ($\Delta\text{AUC} < +0.015$, Null Result)**: Surrounding vehicles offer no incremental predictive power over the pedestrian's own kinematic heading and velocity. (Model parsimony & complexity audit paper).

**Commitment**: All three outcomes are pre-certified as valid, defendable, and publishable.

---

## 3. Benchmark Dataset Partitions & Reconciliation

| Dataset | Split | Partition Source | Tracks / Clips | Frames | Sequences ($T=32$, $\text{stride}=4$) | Purpose |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **PIE** | Train | Sets 01, 02, 04 | 884 tracks | 338,042 | 83,630 windows | Supervised training |
| **PIE** | Validation | Sets 05, 06 | 239 tracks | 82,408 | 19,467 windows | Checkpoint selection & tuning |
| **PIE** | **Held-Out Test** | **Set 03** | **719 tracks** | **294,219** | **68,060 windows** | **Quarantined blind evaluation (H1–H4)** |
| **JAAD** | Cross-Dataset | Official `jaad_beh` | 346 videos | ~82,000 | 4,204 windows | North American domain transfer |
| **TITAN** | Cross-Geography | Tokyo Urban Driving | 700 clips | 107,000 | Variable ($10\text{ Hz}$) | North America $\to$ Japan generalization |
| **IDD-PeD** | Paper 2 Reserve | Unstructured Indian Driving | Dedicated | 100+ videos | Domain adaptation | Reserved for Paper 2 (Unstructured traffic) |

---

## 4. X-MIST Architecture & Mathematical Pipeline

```
                                      X-MIST SYSTEM ARCHITECTURE
                                      
  [ RGB Frames (T, C, H, W) ]   ──>   ResNet18 Backbone       ──>  f_rgb   (B, T, 256) ──┐
  [ 2D Pose Keypoints (T, 36) ] ──>   Linear + LayerNorm      ──>  f_pose  (B, T, 256) ──┼──> [ Cross-Modal Attention ]
  [ 5D Kinematics (T, 5) ]      ──>   MLP + Positional Embed  ──>  f_traj  (B, T, 256) ──┼          (Pairwise Q-K-V)
  [ Scene Semantics (T, 10) ]   ──>   Linear + LayerNorm      ──>  f_scene (B, T, 256) ──┘                 │
                                                                                                           ▼
                                                                                         [ Adaptive Modality Gating ]
                                                                                           f_multi = Σ w_m * f_m
                                                                                                           │
  [ Neighbor Agents (B, T, K, 5) ] ──>  [ Dynamic Relational Graph ]                                       │
  [ Neighbor Mask (B, T, K) ]      ──>    Edges: [dist, Δv, Δθ, TTC]                                       │
                                                     │                                                     │
                                                     └───────────────┬─────────────────────────────────────┘
                                                                     ▼
                                                   [ Dynamic Interaction Transformer ]
                                                       (Multi-Agent Relational Reasoning)
                                                                     │
                                                                     ▼
                                                  [ Spatio-Temporal Transformer (L=3) ]
                                                        Latent Representation Z
                                                                     │
                                       ┌─────────────────────────────┼─────────────────────────────┐
                                       ▼                             ▼                             ▼
                           [ Crossing Intention Head ]     [ Action State Head ]         [ Pedestrian Taxonomy Head ]
                             Query-Attention Decoder         Linear Classifier             Linear Classifier
                                (Crossing vs Not)             (Walk vs Stand)               (8 fine-grained classes)
```

### Core Components
1. **Multimodal Encoders**:
   - `TrajectoryEncoder`: Encodes 5D trajectory $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}, v_{\text{est}}]$ with temporal positional encodings.
   - `RGBEncoder`: Spatial CNN backbone (ResNet18 / Lightweight) with feature projection to $d=256$.
   - `PoseEncoder` & `SceneEncoder`: Linear projections mapping skeletal keypoints and semantic context into $d=256$.
2. **Cross-Modal Attention (`CrossModalFusion`)**:
   - Computes pairwise cross-attention: Query $F^{\text{traj}}$, Key/Value $F^{\text{rgb}}$, $F^{\text{pose}}$, $F^{\text{scene}}$.
   - Enhances trajectory representations while capturing cross-modal attention maps for interpretability.
3. **Adaptive Modality Gating (`ModalityGating`)**:
   - Dynamically computes weights $w_m = \text{softmax}(W_g [F_1; F_2; F_3; F_4])$.
   - Fuses representations: $F_{\text{multi}} = \sum_{m} w_m F_m$.
4. **Dynamic Interaction Graph & Transformer**:
   - Relational edge features: $e_{ij} = [d_{ij}, \Delta v_{ij}, \Delta\theta_{ij}, \text{TTC}_{ij}]$.
   - Relational multi-head attention over up to $K=4$ surrounding agents. Stripped/disabled when `--model isolated`.
5. **Spatio-Temporal Sequence Transformer**:
   - 3-layer Transformer Encoder modeling temporal behavior transitions over 32 frames ($1.07\text{s}$).
6. **Multi-Task Classification Heads**:
   - `CrossingIntentionHead`: Learnable task query token cross-attending over latent representations $Z_{1:T}$ via `BehaviorQueryDecoder`, followed by MLP.
   - `ActionStateHead`: Predicts Walking vs. Standing.
   - `PedestrianBehaviorHead`: 8-class fine-grained ontology classification.

---

## 5. Multi-Task Behavioral Loss Formulation

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cross}} \mathcal{L}_{\text{cross}}^{\text{Focal}} + \lambda_{\text{action}} \mathcal{L}_{\text{action}}^{\text{Focal}} + \lambda_{\text{ped}} \mathcal{L}_{\text{ped}}^{\text{Focal}} + \lambda_{\text{micro}} \mathcal{L}_{\text{micro}} + \lambda_{\text{inter}} \mathcal{L}_{\text{inter}} + \lambda_{\text{xai}} \mathcal{L}_{\text{xai}}$$

### Balanced Parameter Configuration (Commit `998fb24`):
- $\lambda_{\text{cross}} = 5.0$ (Primary crossing intention task; dominant gradient share)
- $\lambda_{\text{action}} = 0.1$ (Secondary action recognition: walking vs standing)
- $\lambda_{\text{ped}} = 0.1$ (Pedestrian behavior taxonomy: 8 classes)
- $\lambda_{\text{micro}} = 0.0$ (Disabled for PIE; no micromobility in dataset)
- $\lambda_{\text{inter}} = 0.0$ (Disabled for isolated baseline; interaction graph off)
- $\lambda_{\text{align}} = 0.0$ (Disabled when interaction branch is off)
- $\lambda_{\text{xai}} = 0.0$ (Disabled for isolated baseline to prevent entropy gradient dilution)
- $\alpha_{\text{cross}} = 0.75$ (Class weighting for minority crossing class in Focal Loss)
- $\gamma = 2.0$ (Focal loss focusing exponent)
- $\text{label\_smoothing} = 0.05$

---

## 6. Project Current Status (October 8, 2026)

### 6.1 Proposed Model (`proposed_seed42`)
- **Status**: Completed training on DGX GPU.
- **Selection Metric**: Minimum Total Validation Loss on Sets 05 & 06 (`0.9537` at **Epoch 21**).
- **Recorded Metrics (Epoch 21 Checkpoint)**:
  - Validation Crossing Accuracy: **`89.0%`**
  - Validation Pedestrian Action Accuracy: **`63.6%`**
  - Validation Crossing ROC-AUC: **`87.57%`**
- **Decision**: Pre-registered and locked as the proposed checkpoint candidate.

### 6.2 Isolated Baseline (`isolated_seed42`)
- **Initial Observation**:
  - Training completed 9 epochs on DGX GPU before early stopping triggered (`Patience: 8/8`).
  - Validation ROC-AUC flatlined at **`50.00%`**, PR-AUC at **`16.11%`**, and F1 at **`0.00%`**.
  - Crossing probabilities across all 16 validation samples were 100% identical: `0.44406` ($\text{std} = 0.00000002$).
- **Root-Cause Diagnostic Audit Results**:
  1. *Dead RGB Gradient*: When dummy zero tensors were passed, autograd was initially disconnected. **Fixed in `cfcdfd3`**; gradient is now healthy (`max = 1.638`).
  2. *Drowned Crossing Gradient*: The 8-class pedestrian action loss ($\mathcal{L}_{\text{ped}} = 1.5251$) and XAI entropy ($\mathcal{L}_{\text{xai}} = 1.3366$) accounted for $>85\%$ of the total loss, drowning the focal crossing loss ($\mathcal{L}_{\text{cross}} = 0.0722$). The optimizer discovered that predicting the dataset's constant marginal class prior ($82.6\%$ negative) minimized total loss to $\ln(2) \approx 0.6910$, eliminating sample-to-sample output variance.
  3. *Un-updated Best Checkpoint*: The trainer only saves `checkpoint_best.pt` when `current_metric > best_metric_val`. Because `val_auc` stayed at 50.00% every epoch, `checkpoint_best.pt` remained frozen at Epoch 0.
- **Remediation Applied & Verified (Commit `998fb24`)**:
  - Re-balanced multi-task weights: $\lambda_{\text{cross}} = 5.0$, $\lambda_{\text{ped}} = 0.1$, $\lambda_{\text{action}} = 0.1$, $\lambda_{\text{xai}} = 0.0$.
  - Module gradient for `crossing_head` jumped from $0.43$ to **$1.346$** ($10\times$ larger than `ped_head`).
  - Total training loss cut from $1.134$ down to $0.576$.
  - Empirical verification (160 validation samples) showed immediate emergence of sample-to-sample discrimination: **ROC-AUC climbed from `50.00%` to `53.30%`** (`std = 0.010895`).

---

## 7. Cluster Execution Directives & Active Scripts

### HPC Environment Spec
- **Host**: `25212031117@hpc.manit.ac.in`
- **Scheduler**: PBS Pro
- **Queue**: `dgx` (Resource limits: `ngpus=1`, `ncpus=20`, `walltime=24:00:00`)
- **Python Interpreter**: `~/.conda/envs/emit-har/bin/python`

### Relevant Script Manifest
| Script Path | Purpose |
| :--- | :--- |
| [`tools/diagnose_isolated_collapse.py`](file:///home/vikash/Desktop/research%20Implementation/tools/diagnose_isolated_collapse.py) | 4-part root-cause diagnostic script (distribution audit, feature audit, module gradient flow) |
| [`scripts_hpc/pbs_train_isolated.pbs`](file:///home/vikash/Desktop/research%20Implementation/scripts_hpc/pbs_train_isolated.pbs) | PBS submission script for isolated baseline training on DGX GPU |
| [`train.py`](file:///home/vikash/Desktop/research%20Implementation/train.py) | Main distributed training driver (`--model isolated`, `--epochs 50`) |
| [`evaluate.py`](file:///home/vikash/Desktop/research%20Implementation/evaluate.py) | Pre-registered inference and bootstrapping script for Test Set 03 |
| [`configs/config.yaml`](file:///home/vikash/Desktop/research%20Implementation/configs/config.yaml) | Master configuration file defining all loss weights and architectural dimensions |

---

## 8. Immediate Action Plan & Roadmap

```
                                          EXECUTION ROADMAP
                                          
  [ Step 1: HPC Diagnostic Audit ] ──> Run diagnose_isolated_collapse.py on HPC with commit 998fb24.
                 │
                 ▼
  [ Step 2: 5-Epoch Cluster Test ] ──> Launch short 5-epoch test run on DGX GPU.
                                       Verify Val AUC moves off 50.00% (expected: 55%–70%).
                 │
                 ▼
  [ Step 3: Full Baseline Train ]  ──> Submit 50-epoch PBS job (pbs_train_isolated.pbs).
                                       Save certified checkpoint_best.pt.
                 │
                 ▼
  [ Step 4: Execute Hypothesis H2] ──> Un-quarantine Held-Out Test Set 03.
                                       Run track-clustered bootstrap test on ΔAUC (Full vs Isolated).
                 │
                 ▼
  [ Step 5: Execute H3 & H4 Tests] ──> Run counterfactual attribution (H3) and urgency grid (H4).
                 │
                 ▼
  [ Step 6: Manuscript Locking ]   ──> Populate locked empirical tables in IEEE T-IV manuscript.
```
