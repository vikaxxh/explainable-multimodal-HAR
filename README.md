# X-MIST / EMIT-HAR: Explainable Multimodal Interaction-aware Spatio-Temporal Transformer

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HPC Ready](https://img.shields.io/badge/HPC-Slurm%20%7C%20DDP-blue.svg)]()

> **Research Thesis & Paper Implementation**: Progressive pipeline for fine-grained multi-agent pedestrian and micromobility behavior recognition with dynamic interaction modeling and space-time-modality-interaction quantitative XAI.

---

## 1. Research Narrative & Problem Formulation

Traditional pedestrian intention models (such as **IntentFormer**) predominantly address a coarse, binary classification task (*cross* vs. *not cross*). In dense urban traffic, however, autonomous systems require fine-grained behavior recognition and relational understanding between heterogeneous agents (pedestrians, bicycles, e-scooters, vehicles).

**X-MIST** (*Explainable Multimodal Interaction-aware Spatio-Temporal Transformer*) extends existing intention frameworks into an interaction-aware, multi-task, and quantitatively explainable research architecture:

```
                      SHARED URBAN VIDEO
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Detection + Tracking │
                  └──────────┬───────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
      RGB / Video          Pose            Trajectory
           │                 │                 │
           ▼                 ▼                 ▼
    Visual Encoder      Pose Encoder     Motion Encoder
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                     Scene Context
                             │
                             ▼
                   Scene/Context Encoder
                             │
                             ▼
               ┌──────────────────────────┐
               │ Multimodal Tokenization  │
               └────────────┬─────────────┘
                            │
                            ▼
                  Cross-Modal Attention
                            │
                            ▼
              Dynamic Interaction Graph (Gt)
                            │
           ┌────────────────┼────────────────┐
           │                │                │
      Pedestrian       Micromobility     Environment
           │                │                │
           └────────────────┼────────────────┘
                            │
                            ▼
              Interaction-Aware Transformer
                            │
                            ▼
              Spatio-Temporal Representation
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
     Pedestrian         Micromobility    Interaction
      Behavior            Behavior         Behavior
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                 Explainability Module (XAI)
                            │
        ┌───────────────────┼──────────────────┐
        ▼                   ▼                  ▼
     Spatial             Temporal           Modality
     Evidence            Evidence           Evidence
        │                   │                  │
        └───────────────────┼──────────────────┘
                            ▼
                     Interaction
                       Evidence
                            │
                            ▼
                  Human-Readable Report
```

---

## 2. Multi-Task Taxonomy (Phase 1)

### Task A — Pedestrian Behavior (8 Classes)
`Walking`, `Standing`, `Stopping`, `Starting`, `Turning`, `Crossing`, `Yielding`, `Avoiding`

### Task B — Micromobility Behavior (9 Classes)
`Moving`, `Stopping`, `Starting`, `Turning`, `Accelerating`, `Decelerating`, `Yielding`, `Avoiding`, `Overtaking`

### Task C — Interaction Behavior (6 Classes)
`Approaching`, `Yielding`, `Avoiding`, `Overtaking`, `Conflict`, `Cooperative movement`

---

## 3. Repository Structure

```
.
├── configs/
│   ├── config.yaml                     # Master hyperparameter and task configuration
│   └── config_loader.py                # Config parser and dictionary merger
├── data/
│   ├── raw/                            # Raw video datasets (JAAD, PIE, TITAN)
│   ├── processed/                      # Preprocessed temporal sequences (.npz / .pt)
│   ├── annotations/                    # Ground-truth behavior labels
│   └── splits/                         # Train / Val / Test split manifests
├── preprocessing/
│   ├── detection.py                    # Multi-agent bounding box extraction
│   ├── tracking.py                     # Multi-object tracking and tracklet linking
│   ├── trajectory.py                   # Kinematics (position, velocity, acceleration, heading)
│   ├── pose_extraction.py              # Skeletal keypoints (J=18, D=3) & velocities
│   ├── segmentation.py                 # Semantic scene context distributions
│   └── build_sequences.py              # Temporal window construction (T=16, 32, 64)
├── datasets/
│   ├── taxonomy.py                     # Class mappings and index lookups
│   ├── synthetic_dataset.py            # Zero-dependency synthetic generator for HPC smoke tests
│   ├── multimodal_dataset.py           # Unified multimodal sequence loader & batch collator
│   ├── pedestrian_dataset.py           # Single-modality adapter
│   └── interaction_dataset.py          # Pairwise interaction adapter
├── models/
│   ├── rgb_encoder.py                  # Visual feature extractor (2D CNN / ResNet)
│   ├── pose_encoder.py                 # Spatio-temporal pose attention (ST-GCN / Transformer)
│   ├── trajectory_encoder.py           # Kinematics 1D-CNN + BiGRU motion encoder
│   ├── scene_encoder.py                # Semantic scene context projection
│   ├── cross_modal_attention.py        # Pairwise cross-attention (Traj <-> RGB, Traj <-> Pose, etc.)
│   ├── modality_fusion.py              # Learned modality gating w_m = softmax(g(F_m))
│   ├── interaction_graph.py            # Dynamic agent graph G_t = (V_t, E_t)
│   ├── interaction_transformer.py      # Relational graph transformer & fusion
│   ├── temporal_transformer.py         # Sequence transformer modeling behavioral transitions
│   ├── baselines.py                    # RGB, Trajectory, Pose baselines & IntentFormer reproduction
│   └── proposed_model.py               # Complete X-MIST / EMIT-HAR architecture
├── heads/
│   ├── pedestrian_behavior.py          # Head 1: 8 classes
│   ├── micromobility_behavior.py       # Head 2: 9 classes
│   └── interaction.py                  # Head 3: 6 classes
├── explainability/
│   ├── temporal_attribution.py         # Temporal importance curve I_t
│   ├── modality_attribution.py         # Modality percentage breakdown (I_RGB, I_Pose, I_Traj, I_Scene)
│   ├── interaction_attribution.py      # Edge importance I_{i,j}
│   ├── faithfulness.py                 # Faithfulness evaluation (Deletion, Insertion, Perturbation)
│   └── explanation_generator.py        # Structured natural language report generator
├── training/
│   ├── losses.py                       # Multi-task loss with alignment & XAI regularization
│   └── trainer.py                      # Multi-GPU DDP trainer with AMP fp16 and checkpointing
├── evaluation/
│   ├── classification_metrics.py       # Accuracy, Precision, Recall, Macro-F1
│   ├── xai_metrics.py                  # Deletion/Insertion AUC & Faithfulness index
│   ├── robustness.py                   # Sensor drop & missing modality stress testing
│   └── ablation.py                     # Automated ablation suite (A1 to A9)
├── visualization/
│   ├── attention_maps.py               # Heatmaps & modality bar charts
│   ├── trajectories.py                 # 2D spatial trajectory paths & headings
│   └── interaction_graph.py            # Dynamic interaction graph visualizer
├── scripts_hpc/
│   ├── slurm_train_single_gpu.sbatch   # Single GPU Slurm batch job
│   ├── slurm_train_ddp_multi_gpu.sbatch# 4-GPU Distributed Data Parallel Slurm job
│   ├── slurm_ablation_matrix.sbatch    # Full ablation benchmark Slurm job
│   └── run_local_test.sh               # 10-second local sanity test script
├── train.py                            # Main training entrypoint CLI
├── evaluate.py                         # Main evaluation & ablation CLI
├── explain.py                          # XAI report & visualization CLI
├── requirements.txt                    # Pip package requirements
├── environment.yml                     # Conda environment definition
└── README.md
```

---

## 4. HPC Installation & Quickstart

### 4.1 Conda Environment Setup on HPC
```bash
# Clone or upload repository to HPC workspace
cd explainable-multimodal-har

# Create conda environment
conda env create -f environment.yml
conda activate emit-har

# Or install via pip
pip install -r requirements.txt
```

### 4.2 Fast Sanity Check (Synthetic Data)
Test the entire pipeline (training, evaluation, XAI, ablations) in under 15 seconds without waiting for large dataset downloads:
```bash
bash scripts_hpc/run_local_test.sh
```

---

## 5. Training Workflows

### 5.1 Single-GPU Training
```bash
# Train proposed X-MIST architecture
python train.py --model proposed --epochs 50 --batch_size 16

# Or submit via Slurm
sbatch scripts_hpc/slurm_train_single_gpu.sbatch
```

### 5.2 Multi-GPU Distributed Data Parallel (DDP) Training
On multi-GPU compute nodes, leverage PyTorch DDP with `torchrun` and automatic mixed precision:
```bash
# Direct launch (4 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --model proposed --distributed

# Or submit via Slurm
sbatch scripts_hpc/slurm_train_ddp_multi_gpu.sbatch
```

### 5.3 Baseline Comparisons & Reproductions (Phases 9 & 10)
```bash
# Baseline 1: Single-Modality RGB
python train.py --model baseline_rgb --epochs 30

# Baseline 2: Single-Modality Trajectory
python train.py --model baseline_traj --epochs 30

# Baseline 3: Single-Modality Pose
python train.py --model baseline_pose --epochs 30

# Baseline 4: IntentFormer Reproduction
python train.py --model intentformer --epochs 40
```

---

## 6. Progressive Research Ablation (Phases 24 & 25)

To generate the primary thesis table comparing all architectural contributions:
```bash
python evaluate.py --ablation
```

This runs the full matrix and exports both CSV and Markdown tables to `experiments/results/ablation_comparison_table.md`:

| Model | RGB | Pose | Trajectory | Scene | Interaction | XAI | Macro-F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RGB baseline** | ✓ | | | | | | ... |
| **Pose baseline** | | ✓ | | | | | ... |
| **Trajectory baseline** | | | ✓ | | | | ... |
| **IntentFormer-style** | ✓ | | ✓ | ✓ | | | ... |
| **Multimodal Transformer** | ✓ | ✓ | ✓ | ✓ | | | ... |
| **+ Interaction** | ✓ | ✓ | ✓ | ✓ | ✓ | | ... |
| **Proposed (X-MIST)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Best** |

---

## 7. Explainability & Faithfulness Suite (Phases 20–22, 26)

Generate human-readable natural language evidence reports with quantitative faithfulness testing:
```bash
python explain.py --sample_idx 0 --faithfulness --visualize
```

### Example Human-Readable Report Output:
```
============================================================
              X-MIST EXPLAINABILITY REPORT
============================================================
Prediction:            Yielding (Interaction: Approaching)
Confidence:            91.4% (Interaction: 88.2%)
Critical Time Window:  1.8s – 2.6s
Interacting Agent:     escooter (ID: 1)
Modality Contribution: Trajectory: 41.2%, Pose: 27.1%, RGB: 20.8%, Scene: 10.9%

Key Behavioral Evidence:
  • Interacting escooter (ID: 1) approached pedestrian path within 3.8m.
  • Pedestrian slowed down significantly (1.35 m/s -> 0.22 m/s).
  • Pedestrian adjusted heading trajectory by 34.2°.
  • Dominant modality guiding decision: Trajectory.

Synthesized Explanation:
  The pedestrian was predicted to execute 'Yielding' influenced by
  the 'Approaching' relation with the escooter.
  Attribution shows this decision was predominantly anchored on the
  Trajectory modality during 1.8s – 2.6s.
============================================================
```

### Faithfulness Verification Tests (Phase 26):
- **Deletion Test**: Verifies target class confidence drops significantly when top-attributed features are masked.
- **Insertion Test**: Measures recovery curve when evidence is inserted into an uninformative baseline.
- **Modality Perturbation**: Tests performance degradation when zeroing each sensor modality.
- **Temporal Perturbation**: Proves masking critical frames hurts prediction significantly more than masking random frames ($Drop_{critical} > Drop_{random}$).
- **Interaction Perturbation**: Tests whether removing the interacting agent changes the predicted yielding/avoiding behavior.

---

## 8. Robustness Stress-Testing (Phase 27)

Evaluate model degradation under realistic adverse sensor and weather conditions:
```bash
python evaluate.py --robustness
```
Evaluates:
1. Clean baseline
2. Missing RGB (sensor failure / low illumination)
3. Missing Pose (occlusion)
4. Noisy Trajectory (GPS/tracker jitter)
5. Missing Interacting Neighbors (isolated context)

---

## 9. Citation & Contact

If using this codebase for your research thesis or publication:
```bibtex
@article{xmist2026,
  title={X-MIST: Explainable Multimodal Interaction-aware Spatio-Temporal Transformer for Multi-Agent Urban Behavior Recognition},
  author={Research Implementation},
  year={2026}
}
```
