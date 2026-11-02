"""Regression tests for the CVMI fixes. Run: python -m pytest -q tests/test_cvmi_fixes.py"""
import math
import torch

from models.proposed_model import ProposedXMISTModel
from preprocessing.augmentation import MultimodalAugmentor
from preprocessing.trajectory import compute_kinematics
from training.losses import MultiTaskBehaviorLoss


def _batch(B=4, T=16, N=4, with_nb=True):
    traj = torch.rand(B, T, 5)
    nb = torch.rand(B, T, N, 5)
    mask = torch.zeros(B, T, N, dtype=torch.bool)
    if with_nb:
        mask[:, :, :2] = True
    return {"trajectory": traj, "neighbor_agents": nb, "neighbor_mask": mask,
            "cross": torch.randint(0, 2, (B,))}


def _model(inter=True):
    torch.manual_seed(0)
    m = ProposedXMISTModel(feature_dim=64, use_rgb=False, use_pose=False, use_scene=False,
                           use_cross_modal=False, use_modality_gating=False, use_interaction=inter, dropout=0.0)
    return m.eval()


def test_traj_only_forward_without_rgb_pose_scene():
    out = _model()(_batch())
    assert out["crossing_logits"].shape == (4, 2)
    assert out["modality_weights"] is None


def test_no_neighbour_output_ignores_padding_content():
    """With all neighbours masked, padded neighbour values must not influence the prediction
    (previously slot 0 was unmasked and its edge features leaked the pedestrian's own kinematics)."""
    m = _model()
    b1 = _batch(with_nb=False)
    b2 = {k: v.clone() for k, v in b1.items()}
    b2["neighbor_agents"] = torch.randn_like(b2["neighbor_agents"]) * 5
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    assert torch.allclose(o1["crossing_logits"], o2["crossing_logits"], atol=1e-5)
    assert o1["interaction_weights"].abs().sum() == 0


def test_masked_neighbour_does_not_matter_but_valid_one_does():
    m = _model()
    b = _batch()
    b_pad = {k: v.clone() for k, v in b.items()}
    b_pad["neighbor_agents"][:, :, 3] += 3.0           # slot 3 is masked
    b_val = {k: v.clone() for k, v in b.items()}
    b_val["neighbor_agents"][:, :, 0] += 3.0           # slot 0 is valid
    with torch.no_grad():
        o, o_pad, o_val = m(b), m(b_pad), m(b_val)
    assert torch.allclose(o["crossing_logits"], o_pad["crossing_logits"], atol=1e-5)
    assert not torch.allclose(o["crossing_logits"], o_val["crossing_logits"], atol=1e-5)


def test_hflip_keeps_heading_consistent_with_velocity():
    pos = torch.tensor([[0.2 + 0.01 * t, 0.5 + 0.005 * t] for t in range(16)]).numpy()
    tr = torch.from_numpy(compute_kinematics(pos, dt=1 / 30))
    f = MultimodalAugmentor.hflip_tracks(tr)
    assert torch.allclose(torch.atan2(f[1:, 3], f[1:, 2]), f[1:, 4], atol=1e-5)
    r = MultimodalAugmentor.rotate_tracks(tr, 0.7)
    assert torch.allclose(torch.atan2(r[1:, 3], r[1:, 2]), r[1:, 4], atol=1e-5)


def test_training_step_runs():
    m = _model().train()
    loss = MultiTaskBehaviorLoss(lambda_cross=1.0, lambda_action=0, lambda_ped=0, lambda_xai=0)
    l = loss(m(_batch()), _batch())["loss_total"]
    l.backward()
    assert math.isfinite(l.item())


def test_augmentor_default_does_not_flip():
    """hflip assumes [0, 1]-normalised x; it must be opt-in so legacy (non-normalised) data is untouched."""
    b = _batch()
    out = MultimodalAugmentor(prob_joint_mask=0.0).augment_batch(b)
    assert torch.equal(out["trajectory"], b["trajectory"])


def test_pie_neighbour_index_survives_id_collision():
    """A vehicle sharing a pedestrian's id must not overwrite that pedestrian in the neighbour index."""
    from datasets.pie_adapter import PIEAdapter
    n = 120

    def ped(x0, crossing):
        return {"bbox": [[x0 + t, 500, 60, 150] for t in range(n)], "frames": list(range(n)),
                "actions": [1] * n, "cross": [1 if (crossing and t >= n - 10) else 0 for t in range(n)]}
    vid = {"ped_annotations": {"a": ped(300, True), "b": ped(340, False)},
           "vehicle_annotations": {"b": {"bbox": [[360, 520, 200, 120]] * n, "frames": list(range(n)), "type": "car"}}}
    seqs = PIEAdapter(pie_root="/nonexistent", window_size=16).extract_tte_sequences({"set01": {"video_0001": vid}})
    types = {t for s in seqs if s["metadata"]["ped_id"] == "a" for t in s["metadata"]["neighbor_types"]}
    assert types == {"pedestrian", "car"}
