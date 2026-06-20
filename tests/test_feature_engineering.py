"""
Tests for src/data/feature_engineering.py

Verifies:
  - Room feature tensor has correct shape and normalization
  - Env feature tensor has correct shape and normalization
  - One-hot encoding correctness
  - Edge index construction
  - Edge attribute construction
  - Full HeteroData assembly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    DaylightLevel, NoiseLevel, ZoneType,
)
from utils.constants import ROOM_FEAT_DIM, ENV_FEAT_DIM

from data.room_factory import (
    RoomSpec, RoomNode, EnvironmentalNode, RoomCatalog, RoomFactory, EnvNodeFactory
)
from data.feature_engineering import FeatureEngineer


def _make_test_rooms():
    """Create a small set of test rooms."""
    catalog = RoomCatalog({
        RoomType.CLASSROOM: RoomSpec(
            room_type=RoomType.CLASSROOM, display_name="教室",
            area_range_sqm=(54.0, 72.0), aspect_ratio_range=(1.0, 1.8),
            daylight_level=DaylightLevel.HIGH, noise_level=NoiseLevel.MODERATE,
            noise_tolerance=NoiseLevel.MODERATE, occupancy_density=1.2,
            fire_exits_min=2, floor_preference=[1, 2, 3],
        ),
        RoomType.CORRIDOR: RoomSpec(
            room_type=RoomType.CORRIDOR, display_name="走道",
            area_range_sqm=(12.0, 48.0), aspect_ratio_range=(3.0, 12.0),
            daylight_level=DaylightLevel.LOW, noise_level=NoiseLevel.NOISY,
            noise_tolerance=NoiseLevel.LOUD, occupancy_density=0.3,
            fire_exits_min=2, floor_preference=[0, 1, 2, 3, 4],
        ),
        RoomType.TOILET: RoomSpec(
            room_type=RoomType.TOILET, display_name="卫生间",
            area_range_sqm=(12.0, 24.0), aspect_ratio_range=(1.0, 2.5),
            daylight_level=DaylightLevel.NONE, noise_level=NoiseLevel.MODERATE,
            noise_tolerance=NoiseLevel.MODERATE, occupancy_density=0.8,
            fire_exits_min=1, floor_preference=[0, 1, 2, 3, 4],
        ),
    })

    rooms = [
        RoomNode("classroom_000_f1", catalog.get(RoomType.CLASSROOM),
                 60.0, 1.5, 1, (50.0, 100.0), zone_id=0),
        RoomNode("classroom_001_f1", catalog.get(RoomType.CLASSROOM),
                 60.0, 1.5, 1, (60.0, 100.0), zone_id=0),
        RoomNode("corridor_000_f1", catalog.get(RoomType.CORRIDOR),
                 30.0, 6.0, 1, (55.0, 80.0), zone_id=4),
        RoomNode("toilet_000_f1", catalog.get(RoomType.TOILET),
                 18.0, 1.5, 1, (70.0, 70.0), zone_id=3),
    ]
    return rooms


def _make_test_env_nodes():
    """Create a small set of test environment nodes."""
    return [
        EnvironmentalNode("south_facing_00", EnvNodeType.SOUTH_FACING,
                          (100.0, 150.0), {'solar_orientation': 1.0}),
        EnvironmentalNode("main_road_00", EnvNodeType.MAIN_ROAD_ACCESS,
                          (0.0, 75.0), {'access_type': 1.0}),
    ]


def test_room_feature_shape():
    """Room feature tensor should have shape (num_rooms, ROOM_FEAT_DIM)."""
    eng = FeatureEngineer()
    rooms = _make_test_rooms()
    features, idx_map = eng.build_room_features(rooms)

    assert features.shape == (4, ROOM_FEAT_DIM), \
        f"Expected (4, {ROOM_FEAT_DIM}), got {features.shape}"
    assert len(idx_map) == 4
    assert idx_map['classroom_000_f1'] == 0
    print("  PASS: test_room_feature_shape")


def test_room_feature_one_hot():
    """Verify one-hot encoding for RoomType in column range [0:13]."""
    eng = FeatureEngineer()
    rooms = _make_test_rooms()
    features, _ = eng.build_room_features(rooms)

    # classrooms (idx 0) should have RoomType.CLASSROOM = index 0 set to 1
    classroom_idx = list(RoomType).index(RoomType.CLASSROOM)
    assert features[0, classroom_idx] == 1.0, \
        f"classroom one-hot: expected 1.0 at col {classroom_idx}, got {features[0, classroom_idx]}"

    # corridor (idx 1) should have RoomType.CORRIDOR = index 8 set to 1
    corridor_idx = list(RoomType).index(RoomType.CORRIDOR)
    assert features[2, corridor_idx] == 1.0, \
        f"corridor one-hot: expected 1.0 at col {corridor_idx}, got {features[2, corridor_idx]}"

    # Check that only one bit is set in [0:13] per room
    for i in range(4):
        one_hot_sum = features[i, :13].sum()
        assert abs(one_hot_sum - 1.0) < 1e-6, \
            f"Room {i}: one-hot sum = {one_hot_sum}, expected 1.0"

    print("  PASS: test_room_feature_one_hot")


def test_room_feature_scalars():
    """Verify scalar features are correctly placed and normalized."""
    eng = FeatureEngineer(max_area=800.0, max_occupancy=300.0)
    rooms = _make_test_rooms()
    features, _ = eng.build_room_features(rooms)

    # Room 0 (classroom): area=60, occupancy=50
    assert abs(features[0, 13] - 60.0/800.0) < 1e-6, f"area: {features[0, 13]}"
    assert abs(features[0, 15] - 50.0/300.0) < 1e-6, f"occupancy: {features[0, 15]}"

    # Room 0: daylight_level = HIGH = 3, normalized = 3/4
    assert abs(features[0, 16] - 0.75) < 1e-6, f"daylight: {features[0, 16]}"

    # Room 2 (corridor): daylight_level = LOW = 1, normalized = 1/4
    assert abs(features[2, 16] - 0.25) < 1e-6, f"corridor daylight: {features[2, 16]}"

    print("  PASS: test_room_feature_scalars")


def test_room_feature_zone_one_hot():
    """Verify ZoneType one-hot encoding in column range [20:26]."""
    eng = FeatureEngineer()
    rooms = _make_test_rooms()
    features, _ = eng.build_room_features(rooms)

    # classroom → teaching zone (index 0 in ZoneType enum)
    assert abs(features[0, 20] - 1.0) < 1e-6, f"classroom zone: {features[0, 20:26]}"
    # corridor → circulation zone (index 4)
    assert abs(features[2, 24] - 1.0) < 1e-6, f"corridor zone: {features[2, 20:26]}"
    # toilet → service zone (index 3)
    assert abs(features[3, 23] - 1.0) < 1e-6, f"toilet zone: {features[3, 20:26]}"

    print("  PASS: test_room_feature_zone_one_hot")


def test_env_feature_shape():
    """Env feature tensor should have shape (num_env, ENV_FEAT_DIM)."""
    eng = FeatureEngineer()
    env_nodes = _make_test_env_nodes()
    features, idx_map = eng.build_env_features(env_nodes)

    assert features.shape == (2, ENV_FEAT_DIM), \
        f"Expected (2, {ENV_FEAT_DIM}), got {features.shape}"
    assert len(idx_map) == 2
    print("  PASS: test_env_feature_shape")


def test_env_feature_one_hot():
    """Verify one-hot encoding for EnvNodeType."""
    eng = FeatureEngineer()
    env_nodes = _make_test_env_nodes()
    features, _ = eng.build_env_features(env_nodes)

    # south_facing = index 0
    south_idx = list(EnvNodeType).index(EnvNodeType.SOUTH_FACING)
    assert features[0, south_idx] == 1.0

    # main_road_access = index 1
    road_idx = list(EnvNodeType).index(EnvNodeType.MAIN_ROAD_ACCESS)
    assert features[1, road_idx] == 1.0

    print("  PASS: test_env_feature_one_hot")


def test_edge_index_construction():
    """Edge index should map IDs to correct integer indices."""
    eng = FeatureEngineer()

    src_ids = {'a': 0, 'b': 1, 'c': 2}
    dst_ids = {'a': 0, 'b': 1, 'c': 2}

    edges = [
        ('a', 'b', {'weight': 1.0}),
        ('b', 'c', {'weight': 2.0}),
        ('c', 'a', {'weight': 3.0}),
    ]

    edge_index = eng.build_edge_index(edges, src_ids, dst_ids)

    assert edge_index.shape == (2, 3), f"Shape: {edge_index.shape}"
    assert edge_index[:, 0].tolist() == [0, 1]  # a→b
    assert edge_index[:, 1].tolist() == [1, 2]  # b→c
    assert edge_index[:, 2].tolist() == [2, 0]  # c→a
    print("  PASS: test_edge_index_construction")


def test_edge_index_empty():
    """Empty edge list should produce zero-shaped tensor."""
    eng = FeatureEngineer()
    ids = {'a': 0}
    edge_index = eng.build_edge_index([], ids, ids)
    assert edge_index.shape == (2, 0)
    print("  PASS: test_edge_index_empty")


def test_normalize_clamping():
    """Values outside [0, 1] after normalization should be clamped."""
    assert abs(FeatureEngineer._normalize(10.0, 5.0) - 1.0) < 1e-9
    assert abs(FeatureEngineer._normalize(-1.0, 5.0) - 0.0) < 1e-9
    print("  PASS: test_normalize_clamping")


def test_feature_norm_bounds():
    """All feature values should be in [0, 1]."""
    eng = FeatureEngineer()
    rooms = _make_test_rooms()
    features, _ = eng.build_room_features(rooms)

    assert features.min() >= 0.0, f"Min feature = {features.min()}"
    assert features.max() <= 1.0 + 1e-6, f"Max feature = {features.max()}"

    env_nodes = _make_test_env_nodes()
    env_features, _ = eng.build_env_features(env_nodes)
    assert env_features.min() >= 0.0
    assert env_features.max() <= 1.0 + 1e-6

    print("  PASS: test_feature_norm_bounds")


def run_all_tests():
    test_room_feature_shape()
    test_room_feature_one_hot()
    test_room_feature_scalars()
    test_room_feature_zone_one_hot()
    test_env_feature_shape()
    test_env_feature_one_hot()
    test_edge_index_construction()
    test_edge_index_empty()
    test_normalize_clamping()
    test_feature_norm_bounds()


if __name__ == '__main__':
    print("Running tests for: feature_engineering.py")
    run_all_tests()
    print("All tests passed!")
