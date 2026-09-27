"""
TITAN (Trajectory Inference and Tracking in Agility) Dataset Adapter.
Honda Research Institute Benchmark:
- 700 urban video sequences, 107k frames captured in Tokyo at 10 Hz
- 50 fine-grained action classes (walking, crossing, standing, yielding, waiting, avoiding)
- Explicit multi-agent interaction annotations (pedestrian yielding to vehicle/cyclist)
Provides dense interaction labels that directly populate rare classes (Yielding, Avoiding, Turning).
"""

import os
import glob
import csv
import json
import numpy as np
from typing import Dict, Any, List, Optional

from datasets.taxonomy import (
    PED_TO_IDX,
    MICRO_TO_IDX,
    INTER_TO_IDX
)
from preprocessing.trajectory import compute_kinematics
from preprocessing.build_sequences import build_sample_dict


class TITANAdapter:
    """
    Parser and preprocessor for Honda TITAN dataset.
    Supports official CSV annotations, JSON metadata, and realistic mock fallback.
    """

    def __init__(
        self,
        titan_root: str = "data/raw/TITAN",
        window_size: int = 32,
        stride: int = 8,
        fps: float = 10.0
    ):
        self.titan_root = titan_root
        self.window_size = window_size
        self.stride = stride
        self.fps = fps
        self.dt = 1.0 / fps

    def load_annotations(self, annotation_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads TITAN annotations from CSV or JSON files.
        Falls back to realistic interactive mock data if no local files exist.
        """
        search_dir = annotation_path if annotation_path else self.titan_root

        # 1. Search for real CSV annotation files
        csv_files = []
        if os.path.isdir(search_dir):
            csv_files = glob.glob(os.path.join(search_dir, "**", "*.csv"), recursive=True)
            csv_files = [f for f in csv_files if "annotation" in f.lower() or "clip" in f.lower() or "titan" in f.lower()]
            if not csv_files:
                csv_files = glob.glob(os.path.join(search_dir, "*.csv"))

        if csv_files:
            print(f"[TITANAdapter] Found {len(csv_files)} real TITAN CSV annotation files in {search_dir}.")
            return self._parse_titan_csv_files(csv_files)

        # 2. Search for real JSON annotation files
        json_files = []
        if os.path.isdir(search_dir):
            json_files = glob.glob(os.path.join(search_dir, "**", "*.json"), recursive=True)
            json_files = [f for f in json_files if "annotation" in f.lower() or "clip" in f.lower() or "titan" in f.lower()]
            if not json_files:
                json_files = glob.glob(os.path.join(search_dir, "*.json"))

        if json_files:
            print(f"[TITANAdapter] Found {len(json_files)} real TITAN JSON annotation files in {search_dir}.")
            return self._parse_titan_json_files(json_files)

        # 3. Fallback: Provide informative log and generate high-fidelity simulated sequences
        print(f"[TITANAdapter] No raw TITAN files (.csv / .json) found in '{search_dir}'.")
        print("[TITANAdapter] Generating simulated high-fidelity TITAN multi-agent interaction sequences for dry-run/pipeline integration.")
        return self._generate_mock_titan_structure()

    def _parse_titan_csv_files(self, csv_files: List[str]) -> Dict[str, Any]:
        """Parses Honda TITAN CSV format into structured tracks."""
        titan_data = {}

        for f_idx, cf in enumerate(csv_files):
            clip_id = os.path.splitext(os.path.basename(cf))[0]
            try:
                with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    fieldnames = [fn.lower().strip() for fn in (reader.fieldnames or [])]

                    # Detect column mappings
                    frame_col = next((c for c in fieldnames if "frame" in c), "frame")
                    id_col = next((c for c in fieldnames if "id" in c or "obj" in c or "track" in c), "obj_id")
                    label_col = next((c for c in fieldnames if "label" in c or "class" in c or "type" in c), "label")
                    act_col = next((c for c in fieldnames if "action" in c), None)

                    tracks = {}
                    for row in reader:
                        lower_row = {k.lower().strip(): v for k, v in row.items()}
                        try:
                            f_num = int(float(lower_row.get(frame_col, 0)))
                            t_id = str(lower_row.get(id_col, "0"))
                            obj_label = str(lower_row.get(label_col, "pedestrian")).lower()
                            act_text = str(lower_row.get(act_col, "")) if act_col else ""

                            # Extract box coordinates (supports [x, y, w, h] or [xtl, ytl, xbr, ybr])
                            if "x" in lower_row and "w" in lower_row:
                                x, y = float(lower_row["x"]), float(lower_row["y"])
                                w, h = float(lower_row["w"]), float(lower_row["h"])
                            elif "xtl" in lower_row and "xbr" in lower_row:
                                xtl, ytl = float(lower_row["xtl"]), float(lower_row["ytl"])
                                xbr, ybr = float(lower_row["xbr"]), float(lower_row["ybr"])
                                x, y = xtl, ytl
                                w, h = max(1.0, xbr - xtl), max(1.0, ybr - ytl)
                            elif "xmin" in lower_row:
                                x, y = float(lower_row["xmin"]), float(lower_row["ymin"])
                                w = max(1.0, float(lower_row["xmax"]) - x)
                                h = max(1.0, float(lower_row["ymax"]) - y)
                            else:
                                continue

                            if t_id not in tracks:
                                tracks[t_id] = {
                                    "label": obj_label,
                                    "frames": [],
                                    "boxes": [],
                                    "actions": []
                                }
                            tracks[t_id]["frames"].append(f_num)
                            tracks[t_id]["boxes"].append([x, y, w, h])
                            tracks[t_id]["actions"].append(act_text)
                        except (ValueError, TypeError):
                            continue

                    if tracks:
                        titan_data[clip_id] = {"tracks": tracks}
            except Exception as e:
                continue

        print(f"[TITANAdapter] Successfully loaded {len(titan_data)} clips from TITAN CSV annotations.")
        return titan_data

    def _parse_titan_json_files(self, json_files: List[str]) -> Dict[str, Any]:
        """Parses Honda TITAN JSON annotation structure."""
        titan_data = {}
        for jf in json_files:
            clip_id = os.path.splitext(os.path.basename(jf))[0]
            try:
                with open(jf, "r") as f:
                    content = json.load(f)
                    if isinstance(content, dict) and "tracks" in content:
                        titan_data[clip_id] = content
                    elif isinstance(content, list):
                        # List of track items
                        tracks = {}
                        for item in content:
                            t_id = str(item.get("track_id", item.get("obj_id", len(tracks))))
                            tracks[t_id] = {
                                "label": item.get("label", "pedestrian"),
                                "boxes": item.get("boxes", item.get("bbox", [])),
                                "frames": item.get("frames", list(range(len(item.get("boxes", []))))),
                                "actions": item.get("actions", [])
                            }
                        titan_data[clip_id] = {"tracks": tracks}
            except Exception:
                continue
        return titan_data

    def _generate_mock_titan_structure(self, num_clips: int = 250) -> Dict[str, Any]:
        """
        Generates realistic multi-agent Honda TITAN interaction clips.
        Populates all 8 behaviors with balanced distribution:
        Yielding, Avoiding, Turning, Crossing, Walking, Standing, Stopping, Starting.
        Includes multi-agent interactions (bicycles, vehicles, scooters) with collision conflicts.
        """
        mock_data = {}
        behavior_cycle = [
            "yielding", "avoiding", "crossing", "turning",
            "stopping", "starting", "walking", "standing"
        ]
        rng = np.random.RandomState(42)

        for c_idx in range(1, num_clips + 1):
            clip_id = f"clip_{c_idx:04d}"
            tracks = {}
            assigned_action = behavior_cycle[(c_idx - 1) % len(behavior_cycle)]
            clip_len = int(rng.randint(95, 125))

            # Base coordinates and variations
            x0 = float(rng.uniform(100, 300))
            y0 = float(rng.uniform(180, 260))
            w = float(rng.uniform(45, 55))
            h = float(rng.uniform(130, 150))

            p_boxes = []
            for t in range(clip_len):
                if assigned_action == "yielding":
                    if t < 30:
                        bx = x0 + t * 1.5
                        by = y0
                    elif t < 65:
                        bx = x0 + 30 * 1.5 + rng.normal(0, 0.2)
                        by = y0 + rng.normal(0, 0.2)
                    else:
                        bx = x0 + 45.0 + (t - 65) * 1.8
                        by = y0 + (t - 65) * 0.8
                elif assigned_action == "avoiding":
                    bx = x0 + t * 1.8
                    phase = max(0.0, min(1.0, (t - 25) / 35.0))
                    swerve = 25.0 * np.sin(np.pi * phase) if 25 <= t <= 60 else 0.0
                    by = y0 + swerve
                elif assigned_action == "turning":
                    if t < 45:
                        bx = x0 + t * 1.6
                        by = y0
                    else:
                        bx = x0 + 45 * 1.6 + rng.normal(0, 0.2)
                        by = y0 + (t - 45) * 1.8
                elif assigned_action == "crossing":
                    bx = x0 + t * 1.8
                    by = y0 + t * 0.6
                elif assigned_action == "stopping":
                    if t < 35:
                        bx = x0 + t * 1.8
                    elif t < 60:
                        decay = max(0.0, 1.0 - (t - 35) / 25.0)
                        bx = x0 + 35 * 1.8 + (t - 35) * 1.8 * decay
                    else:
                        bx = x0 + 35 * 1.8 + 25 * 0.9 + rng.normal(0, 0.2)
                    by = y0
                elif assigned_action == "starting":
                    if t < 35:
                        bx = x0 + rng.normal(0, 0.2)
                    elif t < 60:
                        accel = min(1.0, (t - 35) / 25.0)
                        bx = x0 + (t - 35) * 1.8 * accel
                    else:
                        bx = x0 + 25 * 0.9 + (t - 60) * 2.0
                    by = y0
                elif assigned_action == "standing":
                    bx = x0 + rng.normal(0, 0.2)
                    by = y0 + rng.normal(0, 0.2)
                else:  # walking
                    bx = x0 + t * 1.7 + rng.normal(0, 0.2)
                    by = y0 + rng.normal(0, 0.2)

                p_boxes.append([float(bx), float(by), float(w), float(h)])

            tracks["ped_1"] = {
                "label": "pedestrian",
                "boxes": p_boxes,
                "frames": list(range(clip_len)),
                "actions": [assigned_action] * clip_len
            }

            # Interacting Agent: Oncoming vehicle, bicycle, or e-scooter
            v_type_cycle = ["vehicle", "bicycle", "escooter"]
            v_label = v_type_cycle[c_idx % 3]
            v_w = 140 if v_label == "vehicle" else (65 if v_label == "bicycle" else 45)
            v_h = 110 if v_label == "vehicle" else (120 if v_label == "bicycle" else 130)
            vx0 = float(rng.uniform(350, 500))
            vy0 = y0 + float(rng.uniform(-20, 20))
            v_speed = 3.0 if v_label == "vehicle" else (2.4 if v_label == "bicycle" else 2.2)

            v_boxes = []
            for t in range(clip_len):
                v_bx = vx0 - t * v_speed
                v_boxes.append([float(v_bx), float(vy0), float(v_w), float(v_h)])

            tracks["inter_1"] = {
                "label": v_label,
                "boxes": v_boxes,
                "frames": list(range(clip_len)),
                "actions": ["moving" if t < 40 else ("yielding" if assigned_action == "crossing" else "moving") for t in range(clip_len)]
            }

            # Dense multi-agent interaction in 50% of clips
            if c_idx % 2 == 0:
                p2_x0 = x0 + float(rng.uniform(60, 100))
                p2_boxes = []
                for t in range(clip_len):
                    p2_bx = p2_x0 + t * 1.2
                    p2_boxes.append([float(p2_bx), float(y0 - 40), 50.0, 140.0])
                tracks["ped_2"] = {
                    "label": "pedestrian",
                    "boxes": p2_boxes,
                    "frames": list(range(clip_len)),
                    "actions": ["walking"] * clip_len
                }

            mock_data[clip_id] = {"tracks": tracks}
        print(f"[TITANAdapter] Generated {len(mock_data)} multi-agent interaction clips with all 8 behavioral classes.")
        return mock_data

    def extract_sequences(
        self,
        data: Dict[str, Any],
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts synchronized temporal sequence windows (T=32) with interaction neighbors.
        Applies memory-compact 1-element RGB placeholder to avoid RAM exhaustion.
        """
        sequences = []
        total_clips = len(data)

        for c_idx, (clip_id, clip_content) in enumerate(data.items()):
            tracks = clip_content.get("tracks", {})
            ped_tracks = {tid: tdata for tid, tdata in tracks.items() if "ped" in tdata.get("label", "").lower() or tdata.get("label") == "pedestrian"}
            other_tracks = {tid: tdata for tid, tdata in tracks.items() if tid not in ped_tracks}

            for ped_id, ped_data in ped_tracks.items():
                boxes = ped_data.get("boxes", [])
                actions = ped_data.get("actions", ["walking"] * len(boxes))
                frames = ped_data.get("frames", list(range(len(boxes))))
                total_len = len(boxes)

                if total_len < self.window_size:
                    continue

                for start in range(0, total_len - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    win_boxes = boxes[start:end]
                    win_actions = [str(a).lower() for a in actions[start:end]]
                    win_frames = set(frames[start:end])

                    centers = np.array([[b[0] + b[2]/2.0, b[1] + b[3]/2.0] for b in win_boxes], dtype=np.float32)
                    kinematics = compute_kinematics(centers, dt=self.dt)
                    speeds = np.linalg.norm(kinematics[:, 2:4], axis=1)

                    # Map to Task A (Pedestrian Behavior) with balanced support
                    act_str = " ".join(win_actions)
                    if "yield" in act_str or "wait" in act_str:
                        ped_label = PED_TO_IDX["Yielding"]
                    elif "avoid" in act_str or "swerve" in act_str:
                        ped_label = PED_TO_IDX["Avoiding"]
                    elif "turn" in act_str:
                        ped_label = PED_TO_IDX["Turning"]
                    elif "cross" in act_str:
                        ped_label = PED_TO_IDX["Crossing"]
                    elif "stop" in act_str:
                        ped_label = PED_TO_IDX["Stopping"]
                    elif "start" in act_str:
                        ped_label = PED_TO_IDX["Starting"]
                    elif "stand" in act_str:
                        ped_label = PED_TO_IDX["Standing"]
                    elif speeds[-1] < 0.2 and speeds[0] > 0.5:
                        ped_label = PED_TO_IDX["Stopping"]
                    elif speeds[-1] > 0.6 and speeds[0] < 0.2:
                        ped_label = PED_TO_IDX["Starting"]
                    elif np.mean(speeds) > 0.6:
                        ped_label = PED_TO_IDX["Walking"]
                    else:
                        ped_label = PED_TO_IDX["Standing"]

                    # Extract up to K=4 interacting neighbors present in the same frames
                    neighbor_agents = np.zeros((self.window_size, 4, 5), dtype=np.float32)
                    neighbor_mask = np.zeros((self.window_size, 4), dtype=bool)

                    micro_label = MICRO_TO_IDX["Moving"]
                    inter_label = INTER_TO_IDX["Cooperative"]

                    n_slot = 0
                    for oth_id, oth_data in other_tracks.items():
                        if n_slot >= 4:
                            break
                        oth_boxes = oth_data.get("boxes", [])
                        oth_frames = oth_data.get("frames", list(range(len(oth_boxes))))

                        # Extract overlapping time slice
                        frame_to_box = {f: b for f, b in zip(oth_frames, oth_boxes)}
                        matched_boxes = [frame_to_box.get(f) for f in frames[start:end]]

                        if all(b is not None for b in matched_boxes):
                            oth_centers = np.array([[b[0] + b[2]/2.0, b[1] + b[3]/2.0] for b in matched_boxes], dtype=np.float32)
                            oth_kinematics = compute_kinematics(oth_centers, dt=self.dt)
                            neighbor_agents[:, n_slot, :] = oth_kinematics
                            neighbor_mask[:, n_slot] = True

                            # Check relative distance and conflict
                            rel_dists = np.linalg.norm(oth_centers - centers, axis=1)
                            if np.min(rel_dists) < 50.0:
                                inter_label = INTER_TO_IDX["Yielding"] if ped_label == PED_TO_IDX["Yielding"] else INTER_TO_IDX["Conflict"]
                            n_slot += 1

                    # Memory-Safe Sample Dict: 1-element RGB array
                    sample = build_sample_dict(
                        rgb_frames=np.zeros((1,), dtype=np.float32),
                        pose_seq=np.zeros((self.window_size, 18, 3), dtype=np.float32),
                        trajectory=kinematics,
                        scene_context=np.zeros((self.window_size, 10), dtype=np.float32),
                        neighbor_agents=neighbor_agents,
                        neighbor_mask=neighbor_mask,
                        ped_label=ped_label,
                        micro_label=micro_label,
                        inter_label=inter_label,
                        agent_type="pedestrian",
                        metadata={"dataset": "TITAN", "clip_id": clip_id, "ped_id": ped_id}
                    )
                    sequences.append(sample)

                    if max_samples and len(sequences) >= max_samples:
                        print(f"[TITANAdapter] Reached requested sample limit: {max_samples}")
                        return sequences

        print(f"[TITANAdapter] Successfully extracted {len(sequences)} sequences from TITAN.")
        return sequences
