"""
Taxonomy and Label Specifications for X-MIST / EMIT-HAR Research Pipeline.
Matches Phase 1 of the implementation plan:
- Task A: Pedestrian Behavior (8 classes)
- Task B: Micromobility Behavior (9 classes)
- Task C: Interaction Behavior (6 classes)
- Scene Context (10 classes)
- Agent Types (4 classes)
"""

from typing import List, Dict

AGENT_TYPES: List[str] = [
    "pedestrian",
    "bicycle",
    "escooter",
    "vehicle"
]

PEDESTRIAN_CLASSES: List[str] = [
    "Walking",
    "Standing",
    "Stopping",
    "Starting",
    "Turning",
    "Crossing",
    "Yielding",
    "Avoiding"
]

MICROMOBILITY_CLASSES: List[str] = [
    "Moving",
    "Stopping",
    "Starting",
    "Turning",
    "Accelerating",
    "Decelerating",
    "Yielding",
    "Avoiding",
    "Overtaking"
]

INTERACTION_CLASSES: List[str] = [
    "Approaching",
    "Yielding",
    "Avoiding",
    "Overtaking",
    "Conflict",
    "Cooperative"
]

SCENE_CLASSES: List[str] = [
    "road",
    "sidewalk",
    "crosswalk",
    "traffic_signal",
    "buildings",
    "obstacles",
    "vehicles",
    "cycling_lane",
    "pedestrian_area",
    "background"
]

# Mapping helpers
PED_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(PEDESTRIAN_CLASSES)}
IDX_TO_PED: Dict[int, str] = {i: name for i, name in enumerate(PEDESTRIAN_CLASSES)}

MICRO_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(MICROMOBILITY_CLASSES)}
IDX_TO_MICRO: Dict[int, str] = {i: name for i, name in enumerate(MICROMOBILITY_CLASSES)}

INTER_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(INTERACTION_CLASSES)}
IDX_TO_INTER: Dict[int, str] = {i: name for i, name in enumerate(INTERACTION_CLASSES)}

AGENT_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(AGENT_TYPES)}
IDX_TO_AGENT: Dict[int, str] = {i: name for i, name in enumerate(AGENT_TYPES)}

SCENE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(SCENE_CLASSES)}
IDX_TO_SCENE: Dict[int, str] = {i: name for i, name in enumerate(SCENE_CLASSES)}
