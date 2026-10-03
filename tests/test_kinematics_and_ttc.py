"""
Unit Tests for Kinematics, Time-To-Collision (TTC), and Counterfactual Perturbations.
Pre-registered verification suite for X-MIST / EMIT-HAR.
Tests:
1. Closing agents (v_closing > 0, TTC = dist / v_closing)
2. Separating agents (v_closing <= 0, TTC = 10.0s / infinite cap)
3. Stationary and parallel moving agents
4. Backward difference kinematics (zero future frame leakage)
5. Kinematic trajectory co-rescaling continuity at t=T_obs
6. Distance-matched control agent selection logic
"""

import math
import unittest
import numpy as np
import torch

from preprocessing.trajectory import compute_kinematics, compute_edge_features, compute_closing_velocity
from models.interaction_graph import DynamicInteractionGraph


class TestKinematicsAndTTC(unittest.TestCase):

    def test_closing_speed_sign_distinguishes_closing_from_separating(self):
        """
        Verify that line-of-sight closing velocity distinguishes closing from separating agents:
        1. Closing agents (distance decreasing): v_closing > 0.
        2. Separating agents (distance increasing): v_closing < 0.
        3. Stationary/lateral agents: v_closing == 0.
        Essential pre-condition assertion for H4's falsification protocol.
        """
        # Pedestrian at (0, 0) stationary
        p_pos = torch.tensor([[0.0, 0.0]])
        p_vel = torch.tensor([[0.0, 0.0]])

        # 1. Closing: Neighbor at (10, 0) moving towards ped at vx = -2.0 m/s
        n_pos_close = torch.tensor([[10.0, 0.0]])
        n_vel_close = torch.tensor([[-2.0, 0.0]])
        v_close = DynamicInteractionGraph.compute_closing_velocity(p_pos, p_vel, n_pos_close, n_vel_close).item()
        assert v_close > 0.0, f"Approaching agent must have positive closing speed, got {v_close}"
        assert math.isclose(v_close, 2.0, rel_tol=1e-3), f"Expected v_closing=2.0, got {v_close}"

        # 2. Separating: Neighbor at (10, 0) moving away from ped at vx = +3.0 m/s
        n_pos_sep = torch.tensor([[10.0, 0.0]])
        n_vel_sep = torch.tensor([[3.0, 0.0]])
        v_sep = DynamicInteractionGraph.compute_closing_velocity(p_pos, p_vel, n_pos_sep, n_vel_sep).item()
        assert v_sep < 0.0, f"Separating agent must have negative closing speed, got {v_sep}"
        assert math.isclose(v_sep, -3.0, rel_tol=1e-3), f"Expected v_closing=-3.0, got {v_sep}"

        # 3. Purely orthogonal motion: Neighbor at (10, 0) moving in y-direction at vy = +4.0 m/s
        n_vel_ortho = torch.tensor([[0.0, 4.0]])
        v_ortho = DynamicInteractionGraph.compute_closing_velocity(p_pos, p_vel, n_pos_sep, n_vel_ortho).item()
        assert math.isclose(v_ortho, 0.0, abs_tol=1e-4), f"Orthogonal motion must have v_closing=0.0, got {v_ortho}"

        # 4. NumPy parity check in preprocessing.trajectory
        np_close = compute_closing_velocity(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]),
                                            np.array([[10.0, 0.0]]), np.array([[-2.0, 0.0]])).item()
        np_sep = compute_closing_velocity(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]),
                                          np.array([[10.0, 0.0]]), np.array([[3.0, 0.0]])).item()
        assert np_close > 0.0 and math.isclose(np_close, 2.0, rel_tol=1e-3)
        assert np_sep < 0.0 and math.isclose(np_sep, -3.0, rel_tol=1e-3)

    def test_closing_agents_ttc(self):
        """Verify that agents closing in on a collision course yield accurate TTC."""
        # Primary pedestrian at (0, 0) stationary: (B=1, T=1, 5)
        p_traj = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

        # Neighbor at (10.0, 0.0) moving left (-2.0 m/s) toward pedestrian: (B=1, T=1, N=1, 5)
        n_traj = torch.tensor([[[[10.0, 0.0, -2.0, 0.0, math.pi]]]], dtype=torch.float32)

        graph = DynamicInteractionGraph(edge_embed_dim=64, distance_threshold=20.0)
        neighbor_mask = torch.tensor([[[True]]])

        edge_feats, _, adj_mask = graph.compute_relational_edges(
            p_traj, n_traj, neighbor_mask
        )

        dist = edge_feats[0, 0, 0, 0].item()
        delta_v = edge_feats[0, 0, 0, 1].item()
        ttc = edge_feats[0, 0, 0, 3].item()

        assert math.isclose(dist, 10.0, rel_tol=1e-3), f"Expected dist=10.0, got {dist}"
        assert math.isclose(delta_v, 2.0, rel_tol=1e-3), f"Expected delta_v=2.0, got {delta_v}"
        # TTC = dist / closing_speed = 10.0 / 2.0 = 5.0 seconds
        assert math.isclose(ttc, 5.0, rel_tol=1e-3), f"Expected TTC=5.0s, got {ttc}"
        assert adj_mask[0, 0, 0].item() is True, "Agent within distance threshold should be connected"

    def test_separating_agents_ttc_is_capped(self):
        """Verify that agents separating (moving away) yield capped/infinite TTC = 10.0s."""
        # Pedestrian at (0, 0)
        p_traj = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

        # Neighbor at (10.0, 0.0) moving right (+2.0 m/s) AWAY from pedestrian
        n_traj = torch.tensor([[[[10.0, 0.0, 2.0, 0.0, 0.0]]]], dtype=torch.float32)

        graph = DynamicInteractionGraph(edge_embed_dim=64, distance_threshold=20.0)
        neighbor_mask = torch.tensor([[[True]]])

        edge_feats, _, _ = graph.compute_relational_edges(
            p_traj, n_traj, neighbor_mask
        )

        ttc = edge_feats[0, 0, 0, 3].item()
        assert math.isclose(ttc, 10.0, rel_tol=1e-3), f"Separating agents must yield capped TTC=10.0s, got {ttc}"

    def test_stationary_agents_ttc(self):
        """Verify that stationary agents with zero relative speed yield capped TTC = 10.0s."""
        p_traj = torch.tensor([[[5.0, 5.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
        n_traj = torch.tensor([[[[5.0, 15.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)

        graph = DynamicInteractionGraph(edge_embed_dim=64, distance_threshold=20.0)
        neighbor_mask = torch.tensor([[[True]]])

        edge_feats, _, _ = graph.compute_relational_edges(
            p_traj, n_traj, neighbor_mask
        )

        ttc = edge_feats[0, 0, 0, 3].item()
        assert math.isclose(ttc, 10.0, rel_tol=1e-3), f"Stationary agents must yield capped TTC=10.0s, got {ttc}"

    def test_backward_difference_kinematics_no_future_leakage(self):
        """Verify that velocity calculation at time t depends strictly on t and t-1."""
        dt = 0.1
        # Track moving at 1.0 unit/step (10.0 units/sec)
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0]
        ], dtype=np.float32)

        feats = compute_kinematics(positions, dt=dt)
        # Check shapes: (5, 5) -> [x, y, vx, vy, heading]
        assert feats.shape == (5, 5)

        # Check velocities at t=1, 2, 3, 4: should be exactly 10.0
        for t in range(1, 5):
            vx_t = feats[t, 2]
            vy_t = feats[t, 3]
            assert math.isclose(vx_t, 10.0, rel_tol=1e-3), f"Expected vx=10.0 at t={t}, got {vx_t}"
            assert math.isclose(vy_t, 0.0, abs_tol=1e-5), f"Expected vy=0.0 at t={t}, got {vy_t}"

        # Modify future position at t=4 and verify t=2 is completely unchanged
        positions_mod = positions.copy()
        positions_mod[4] = [100.0, 100.0]  # drastic change in future
        feats_mod = compute_kinematics(positions_mod, dt=dt)

        np.testing.assert_allclose(
            feats[:3], feats_mod[:3],
            err_msg="Modifying future frame at t=4 contaminated past kinematics at t<=2!"
        )

    def test_kinematic_co_rescaling_continuity(self):
        """
        Verify that rescaling track history with factor alpha:
        p_new(t) = p(T) - alpha * (p(T) - p(t))
        preserves exact position at observation boundary t=T, and scales velocity by alpha.
        """
        T = 16
        dt = 0.1
        alpha = 1.5

        # Linear track from x=0 to x=15
        t_steps = np.arange(T, dtype=np.float32)
        p_hist = np.stack([t_steps, np.zeros(T, dtype=np.float32)], axis=-1)  # (16, 2)

        p_final = p_hist[-1:, :]  # (1, 2) -> [15.0, 0.0]
        p_rescaled = p_final - alpha * (p_final - p_hist)

        # 1. Check boundary continuity: position at t=T-1 (t=15) must be unchanged
        np.testing.assert_allclose(
            p_rescaled[-1], p_hist[-1],
            err_msg="Position at t=T must be invariant under alpha rescaling"
        )

        # 2. Check velocity scaling: v_new = alpha * v_orig
        v_orig = np.diff(p_hist, axis=0) / dt
        v_rescaled = np.diff(p_rescaled, axis=0) / dt

        np.testing.assert_allclose(
            v_rescaled, alpha * v_orig, rtol=1e-4,
            err_msg=f"Velocity must be scaled by exactly alpha={alpha}"
        )

    def test_distance_matched_control_selection(self):
        """
        Verify that candidate neighbors are matched on distance (+-15%), speed (+-20%),
        and heading (+-30 deg).
        """
        target_dist = 10.0
        target_speed = 2.0
        target_heading = 0.0

        tol_dist = 0.15
        tol_speed = 0.20
        tol_head = math.radians(30.0)

        # Candidate 1: well within tolerances
        c1 = {"dist": 10.5, "speed": 2.1, "heading": math.radians(10.0)}
        # Candidate 2: distance too far (13.0 > 11.5)
        c2 = {"dist": 13.0, "speed": 2.0, "heading": 0.0}
        # Candidate 3: speed too different (3.0 > 2.4)
        c3 = {"dist": 10.0, "speed": 3.0, "heading": 0.0}
        # Candidate 4: heading opposite (pi > 30 deg)
        c4 = {"dist": 10.0, "speed": 2.0, "heading": math.pi}

        def is_match(c):
            d_ok = abs(c["dist"] - target_dist) <= tol_dist * target_dist
            s_ok = abs(c["speed"] - target_speed) <= tol_speed * target_speed
            h_ok = abs(c["heading"] - target_heading) <= tol_head
            return d_ok and s_ok and h_ok

        assert is_match(c1) is True, "Candidate 1 should match"
        assert is_match(c2) is False, "Candidate 2 exceeds distance tolerance"
        assert is_match(c3) is False, "Candidate 3 exceeds speed tolerance"
        assert is_match(c4) is False, "Candidate 4 exceeds heading tolerance"
