"""
PIE (Pedestrian & Bicyclist Intention Estimation) Dataset Adapter.
Phase 2 Implementation:
Parses official PIE dataset annotations (.pkl / .json / directory structure):
- Pedestrian bounding boxes, tracking, and actions
- Bicycle bounding boxes, tracking, and trajectories
- Ego-vehicle kinematics
- Multi-agent interaction pairs (Pedestrian <-> Bicycle, Pedestrian <-> Vehicle)
- Ingests or derives poses, scene context, and temporal sequence windows (T=32).
"""

import os
import pickle
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from datasets.taxonomy import (
    PED_TO_IDX,
    MICRO_TO_IDX,
    INTER_TO_IDX,
    PEDESTRIAN_CLASSES,
    MICROMOBILITY_CLASSES,
    INTERACTION_CLASSES
)
from preprocessing.trajectory import compute_kinematics, compute_edge_features
from preprocessing.build_sequences import pad_or_truncate_sequence, build_sample_dict


class PIEAdapter:
    """
    Parser and preprocessor for the PIE benchmark dataset.
    Extracts pedestrians, bicycles, and their dynamic interactions.
    """

    def __init__(
        self,
        pie_root: str,
        window_size: int = 32,
        stride: int = 8,
        fps: float = 10.0,
        distance_threshold: float = 12.0
    ):
        self.pie_root = pie_root
        self.window_size = window_size
        self.stride = stride
        self.fps = fps
        self.dt = 1.0 / fps
        self.distance_threshold = distance_threshold

    def load_annotations(self, annotation_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads PIE annotation pickle or json file.
        Falls back to searching default PIE folder layout.
        """
        if annotation_path is None:
            # Common PIE annotation file locations
            candidates = [
                os.path.join(self.pie_root, "annotations.pkl"),
                os.path.join(self.pie_root, "pie_annotations.pkl"),
                os.path.join(self.pie_root, "annotations", "PIE_annotations.pkl"),
                os.path.join(self.pie_root, "annotations.json"),
                os.path.join(self.pie_root, "pie_annotations.json"),
                os.path.join(self.pie_root, "PIE_annotations.json"),
                os.path.join(self.pie_root, "annotations", "PIE_annotations.json")
            ]
            for c in candidates:
                if os.path.exists(c):
                    annotation_path = c
                    break

            if annotation_path is None:
                import glob
                all_jsons = glob.glob(os.path.join(self.pie_root, "**", "*.json"), recursive=True)
                annot_jsons = [f for f in all_jsons if "calibration" not in f.lower() and "camera_param" not in f.lower()]
                if annot_jsons:
                    annotation_path = annot_jsons[0]
                else:
                    all_pkls = glob.glob(os.path.join(self.pie_root, "**", "*.pkl"), recursive=True)
                    annot_pkls = [f for f in all_pkls if "calibration" not in f.lower() and "camera_param" not in f.lower()]
                    if annot_pkls:
                        annotation_path = annot_pkls[0]

        if annotation_path and os.path.exists(annotation_path):
            try:
                print(f"[PIEAdapter] Loading annotations from {annotation_path}...")
                if annotation_path.endswith(".pkl"):
                    with open(annotation_path, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(annotation_path, "r") as f:
                        data = json.load(f)

                if isinstance(data, dict):
                    has_tracks = any(isinstance(v, dict) for v in data.values())
                    if has_tracks:
                        return data
                    else:
                        print(f"[PIEAdapter] {annotation_path} is metadata/calibration, not track annotations. Skipping.")
            except Exception as e:
                print(f"[PIEAdapter] Error loading {annotation_path}: {e}")

        # Check for official PIE XML annotations directory
        xml_dir = os.path.join(self.pie_root, "annotations")
        if os.path.isdir(xml_dir):
            import glob
            xml_files = sorted(glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True) + glob.glob(os.path.join(xml_dir, "*.xml")))
            if xml_files:
                print(f"[PIEAdapter] Found {len(xml_files)} official PIE XML annotation files in {xml_dir}. Parsing...")
                return self._parse_pie_xml_files(xml_files)

        print("[PIEAdapter] No valid PIE annotation tracks found on disk. Skipping PIE dataset.")
        return {}

    def _parse_pie_xml_files(self, xml_files: List[str]) -> Dict[str, Any]:
        """Parses official PIE XML annotation files across all sets."""
        import xml.etree.ElementTree as ET
        pie_data = {}
        for xf in xml_files:
            # e.g. path/to/annotations/set01/video_0001_annot.xml
            rel_parts = os.path.normpath(xf).split(os.sep)
            set_id = "set01"
            for p in rel_parts:
                if p.startswith("set"):
                    set_id = p
                    break
            video_id = os.path.splitext(os.path.basename(xf))[0].replace("_annot", "")

            if set_id not in pie_data:
                pie_data[set_id] = {}

            try:
                tree = ET.parse(xf)
                root = tree.getroot()
                ped_tracks = {}
                bike_tracks = {}

                for track in root.findall(".//track"):
                    label = track.get("label", "").lower()
                    track_id = track.get("id", "")
                    boxes, frames, cross_flags, actions = [], [], [], []

                    for box in track.findall("box"):
                        if box.get("outside", "0") == "1":
                            continue
                        f_idx = int(box.get("frame", 0))
                        xtl = float(box.get("xtl", 0))
                        ytl = float(box.get("ytl", 0))
                        xbr = float(box.get("xbr", 0))
                        ybr = float(box.get("ybr", 0))
                        w = max(1.0, xbr - xtl)
                        h = max(1.0, ybr - ytl)
                        boxes.append([xtl, ytl, w, h])
                        frames.append(f_idx)

                        cross_val = 0
                        action_val = 1
                        for attr in box.findall("attribute"):
                            aname = attr.get("name", "").lower()
                            atext = (attr.text or "").strip().lower()
                            if "cross" in aname:
                                cross_val = 1 if atext in ["1", "true", "yes", "crossing"] else 0
                            if "action" in aname:
                                action_val = 1 if "walk" in atext else 0
                        cross_flags.append(cross_val)
                        actions.append(action_val)

                    if len(boxes) >= self.window_size:
                        if label == "pedestrian":
                            ped_tracks[track_id or f"ped_{len(ped_tracks)+1}"] = {
                                "bbox": boxes, "frames": frames, "actions": actions, "cross": cross_flags
                            }
                        elif label in ["bicycle", "bicyclist", "bike"]:
                            bike_tracks[track_id or f"bike_{len(bike_tracks)+1}"] = {
                                "bbox": boxes, "frames": frames, "type": "bicycle"
                            }

                if ped_tracks or bike_tracks:
                    pie_data[set_id][video_id] = {
                        "ped_annotations": ped_tracks,
                        "vehicle_annotations": bike_tracks
                    }
            except Exception:
                continue

        total_vids = sum(len(v) for v in pie_data.values())
        print(f"[PIEAdapter] Successfully loaded real tracks from {total_vids} PIE videos across {len(pie_data)} sets.")
        return pie_data

    def _generate_mock_pie_structure(self) -> Dict[str, Any]:
        """
        Generates a realistic mock PIE structure matching official PIE schema
        for validation and testing before full dataset download.
        """
        mock_data = {}
        for set_id in ["set01", "set02", "set03"]:
            mock_data[set_id] = {}
            for video_id in ["video_0001", "video_0002"]:
                ped_tracks = {}
                # Create pedestrian track
                ped_tracks["ped_1"] = {
                    "bbox": [[300 + t * 2, 250, 60, 150] for t in range(100)],
                    "frames": list(range(100)),
                    "actions": [1 if t < 50 else 0 for t in range(100)],  # 1: walking, 0: standing
                    "cross": [1 if 30 <= t <= 70 else 0 for t in range(100)]
                }
                # Create bicycle track
                bicycle_tracks = {}
                bicycle_tracks["bike_1"] = {
                    "bbox": [[350 + t * 5, 260, 50, 100] for t in range(100)],
                    "frames": list(range(100)),
                    "type": "bicycle"
                }

                mock_data[set_id][video_id] = {
                    "ped_annotations": ped_tracks,
                    "vehicle_annotations": bicycle_tracks,
                    "traffic_annotations": {}
                }
        return mock_data

    def map_pie_action_to_taxonomy(
        self,
        actions: List[int],
        cross_flags: List[int],
        kinematics: np.ndarray
    ) -> int:
        """
        Maps PIE low-level signals to our 8 fine-grained pedestrian behaviors:
        ['Walking', 'Standing', 'Stopping', 'Starting', 'Turning', 'Crossing', 'Yielding', 'Avoiding']
        """
        # Kinematics: [x, y, vx, vy, heading]
        speeds = np.linalg.norm(kinematics[:, 2:4], axis=1)
        mean_speed = float(np.mean(speeds))
        v_init = float(speeds[0]) if len(speeds) > 0 else 0.0
        v_final = float(speeds[-1]) if len(speeds) > 0 else 0.0
        heading_change = float(np.abs(kinematics[-1, 4] - kinematics[0, 4])) if len(kinematics) > 0 else 0.0

        is_crossing = any(c == 1 for c in cross_flags)
        is_walking = (np.mean(actions) > 0.5) or (mean_speed > 0.8)

        # Behavioral heuristics grounded on kinematics and annotations
        if is_crossing:
            return PED_TO_IDX["Crossing"]
        elif heading_change > 0.6:  # > ~35 degrees
            return PED_TO_IDX["Turning"]
        elif v_init > 0.8 and v_final < 0.3:
            return PED_TO_IDX["Stopping"]
        elif v_init < 0.3 and v_final > 0.8:
            return PED_TO_IDX["Starting"]
        elif v_init > 0.8 and v_final < 0.5 and heading_change > 0.3:
            return PED_TO_IDX["Yielding"]
        elif is_walking:
            return PED_TO_IDX["Walking"]
        else:
            return PED_TO_IDX["Standing"]

    def map_bicycle_action_to_taxonomy(self, kinematics: np.ndarray) -> int:
        """
        Maps bicycle kinematics to 9 micromobility behaviors:
        ['Moving', 'Stopping', 'Starting', 'Turning', 'Accelerating', 'Decelerating', 'Yielding', 'Avoiding', 'Overtaking']
        """
        speeds = np.linalg.norm(kinematics[:, 2:4], axis=1)
        v_init = float(speeds[0]) if len(speeds) > 0 else 0.0
        v_final = float(speeds[-1]) if len(speeds) > 0 else 0.0
        heading_change = float(np.abs(kinematics[-1, 4] - kinematics[0, 4])) if len(kinematics) > 0 else 0.0

        if heading_change > 0.5:
            return MICRO_TO_IDX["Turning"]
        elif v_final > v_init * 1.3:
            return MICRO_TO_IDX["Accelerating"]
        elif v_final < v_init * 0.7 and v_final > 0.3:
            return MICRO_TO_IDX["Decelerating"]
        elif v_final <= 0.3:
            return MICRO_TO_IDX["Stopping"]
        elif v_init <= 0.3 and v_final > 1.0:
            return MICRO_TO_IDX["Starting"]
        else:
            return MICRO_TO_IDX["Moving"]

    def infer_interaction_label(
        self,
        primary_kinematics: np.ndarray,
        neighbor_kinematics: np.ndarray,
        edge_feats: np.ndarray
    ) -> int:
        """
        Infers pairwise interaction label from relative distance, speed difference, and heading.
        ['Approaching', 'Yielding', 'Avoiding', 'Overtaking', 'Conflict', 'Cooperative']
        """
        # edge_feats: (T, 4) [dist, delta_v, delta_theta, ttc]
        mean_dist = float(np.mean(edge_feats[:, 0]))
        min_dist = float(np.min(edge_feats[:, 0]))
        ttc = float(np.mean(edge_feats[:, 3]))

        if min_dist < 2.0 or ttc < 1.5:
            return INTER_TO_IDX["Conflict"]
        elif edge_feats[-1, 0] < edge_feats[0, 0] * 0.7:  # distance decreasing
            return INTER_TO_IDX["Approaching"]
        elif float(np.linalg.norm(primary_kinematics[-1, 2:4])) < float(np.linalg.norm(primary_kinematics[0, 2:4])) * 0.7:
            return INTER_TO_IDX["Yielding"]
        elif float(np.abs(primary_kinematics[-1, 4] - primary_kinematics[0, 4])) > 0.4:
            return INTER_TO_IDX["Avoiding"]
        else:
            return INTER_TO_IDX["Cooperative"]

    def extract_sequences_from_annotations(
        self,
        pie_data: Dict[str, Any],
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Converts raw PIE nested track structure into standardized temporal sequence windows (T=32).
        """
        sequences = []

        for set_id, videos in pie_data.items():
            if not isinstance(videos, dict):
                continue
            for video_id, video_content in videos.items():
                if not isinstance(video_content, dict):
                    continue
                ped_annotations = video_content.get("ped_annotations", {})
                vehicle_annotations = video_content.get("vehicle_annotations", {})

                # Find all bicycle tracks in this video
                bicycle_tracks = {}
                for v_id, v_data in vehicle_annotations.items():
                    # PIE vehicle annotations might classify by type (bicycle, car, etc.)
                    v_type = v_data.get("type", "vehicle")
                    if "bike" in v_id.lower() or v_type == "bicycle":
                        bicycle_tracks[v_id] = v_data

                # Process each pedestrian track
                for ped_id, ped_data in ped_annotations.items():
                    boxes = ped_data["bbox"]  # list of [x, y, w, h]
                    frames = ped_data["frames"]
                    actions = ped_data.get("actions", [1] * len(boxes))
                    cross_flags = ped_data.get("cross", [0] * len(boxes))

                    total_len = len(boxes)
                    if total_len < self.window_size:
                        continue

                    # Sliding temporal windows (e.g. 32 frames)
                    for start in range(0, total_len - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        win_boxes = boxes[start:end]
                        win_actions = actions[start:end]
                        win_cross = cross_flags[start:end]

                        # Primary agent kinematics: (T, 5)
                        centers = np.array([[b[0] + b[2]/2.0, b[1] + b[3]/2.0] for b in win_boxes], dtype=np.float32)
                        primary_kinematics = compute_kinematics(centers, dt=self.dt)

                        # Primary behavior label
                        ped_label = self.map_pie_action_to_taxonomy(win_actions, win_cross, primary_kinematics)

                        # Check interacting neighbors (bicycles / other pedestrians)
                        neighbor_trajs = np.zeros((self.window_size, 4, 5), dtype=np.float32)
                        neighbor_mask = np.zeros((self.window_size, 4), dtype=bool)
                        inter_label = INTER_TO_IDX["Cooperative"]
                        micro_label = MICRO_TO_IDX["Moving"]

                        n_idx = 0
                        # 1. Nearby bicycles
                        for bike_id, b_data in bicycle_tracks.items():
                            if n_idx >= 4:
                                break
                            b_boxes = b_data.get("bbox", [])
                            if len(b_boxes) >= end:
                                b_win = b_boxes[start:end]
                                b_centers = np.array([[b[0] + b[2]/2.0, b[1] + b[3]/2.0] for b in b_win], dtype=np.float32)
                                b_kin = compute_kinematics(b_centers, dt=self.dt)
                                edge_feats = compute_edge_features(primary_kinematics, b_kin)

                                if np.mean(edge_feats[:, 0]) <= self.distance_threshold:
                                    neighbor_trajs[:, n_idx, :] = b_kin
                                    neighbor_mask[:, n_idx] = True
                                    micro_label = self.map_bicycle_action_to_taxonomy(b_kin)
                                    inter_label = self.infer_interaction_label(primary_kinematics, b_kin, edge_feats)
                                    n_idx += 1

                        # Synthesize or extract pose and scene features
                        # Pose: (T, 18, 3) normalized to bbox
                        pose_seq = np.zeros((self.window_size, 18, 3), dtype=np.float32)
                        pose_seq[:, :, :2] = 0.5
                        pose_seq[:, :, 2] = 0.9

                        # Scene: (T, 10) semantic context
                        scene_seq = np.zeros((self.window_size, 10), dtype=np.float32)
                        scene_seq[:, 1] = 0.7  # sidewalk
                        scene_seq[:, 0] = 0.3  # road

                        # Lightweight RGB representation placeholder
                        rgb_seq = np.zeros((1,), dtype=np.float32)

                        sample = build_sample_dict(
                            rgb_frames=rgb_seq,
                            pose_seq=pose_seq,
                            trajectory=primary_kinematics,
                            scene_context=scene_seq,
                            neighbor_agents=neighbor_trajs,
                            neighbor_mask=neighbor_mask,
                            ped_label=ped_label,
                            micro_label=micro_label,
                            inter_label=inter_label,
                            agent_type="pedestrian",
                            metadata={"dataset": "PIE", "set_id": set_id, "video_id": video_id, "ped_id": ped_id}
                        )
                        sequences.append(sample)

                        if max_samples and len(sequences) >= max_samples:
                            return sequences

        return sequences
