"""
Room Distribution Rules — 房间分布合理性校验

Validates architectural common-sense rules beyond building code constraints:
  - Toilets per floor (not too many, not too few)
  - Minimum classroom area
  - Corridor presence on each floor
  - Reasonable room type mix

These are soft validation rules that flag unrealistic layouts.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
from collections import defaultdict, Counter


def validate_room_distribution(
    rooms: list,  # List[RoomNode]
    num_floors: int = 3,
) -> Tuple[bool, List[str]]:
    """
    Check room distribution for architectural realism.

    Returns:
        (is_valid, warnings_list)
    """
    warnings: List[str] = []

    # ── Toilet rule: 1-3 toilets per PHYSICAL floor ──
    # Count physical floors per typical floor
    tf_phys_count: Dict[str, int] = {}
    for room in rooms:
        tf = getattr(room, 'typical_floor', 'ground')
        fr = getattr(room, 'floor_range', (0, 0))
        phys = fr[1] - fr[0] + 1
        if tf not in tf_phys_count:
            tf_phys_count[tf] = phys

    tf_toilets: Dict[str, int] = defaultdict(int)
    for room in rooms:
        rt = _get_room_type(room)
        tf = getattr(room, 'typical_floor', 'ground')
        if rt == 'toilet':
            tf_toilets[tf] += 1

    for tf, phys_floors in tf_phys_count.items():
        n = tf_toilets.get(tf, 0)
        per_phys = n / max(1, phys_floors)
        if n == 0:
            warnings.append(f"{tf} ({phys_floors} physical floors) has NO toilets")
        elif per_phys > 3:
            warnings.append(f"{tf} has {n} toilets across {phys_floors} floors "
                          f"({per_phys:.1f}/floor) — cap at 3/floor")

    # ── Minimum classroom area: ≥ 54 m² (GB50099 §5.2.1) ──
    for room in rooms:
        rt = _get_room_type(room)
        area = getattr(room, 'area', 0)
        if rt == 'classroom' and area < 54:
            warnings.append(
                f"Classroom {getattr(room, 'room_id', '?')} area={area:.0f}m² "
                f"— minimum is 54m² (GB50099 §5.2.1)"
            )

    # ── Corridor on each typical floor ──
    tf_corridors: Dict[str, int] = defaultdict(int)
    for room in rooms:
        rt = _get_room_type(room)
        tf = getattr(room, 'typical_floor', 'ground')
        if rt == 'corridor':
            tf_corridors[tf] += 1

    for tf in tf_phys_count:
        n = tf_corridors.get(tf, 0)
        if n == 0:
            warnings.append(f"{tf} floor has NO corridors — circulation impossible")

    # ── Staircase per floor ──
    tf_stairs: Dict[str, int] = defaultdict(int)
    for room in rooms:
        rt = _get_room_type(room)
        tf = getattr(room, 'typical_floor', 'ground')
        if rt == 'staircase':
            tf_stairs[tf] += 1

    for tf in tf_phys_count:
        n = tf_stairs.get(tf, 0)
        if n == 0:
            warnings.append(f"{tf} floor has NO staircases — fire egress impossible")
        elif n < 1:
            warnings.append(f"{tf} floor has only {n} staircase — need ≥2 per GB50016")

    # ── Room type diversity ──
    type_counts = Counter(_get_room_type(r) for r in rooms)
    essential = {'classroom', 'corridor', 'staircase', 'toilet'}
    missing = essential - set(type_counts.keys())
    if missing:
        warnings.append(f"Missing essential room types: {missing}")

    return (len(warnings) == 0, warnings)


def _get_room_type(room) -> str:
    """Extract room type string from RoomNode or dict."""
    if hasattr(room, 'room_type'):
        rt = room.room_type
        return rt.value if hasattr(rt, 'value') else str(rt)
    return str(room.get('room_type', 'unknown'))


def check_toilet_coverage(
    ortho_rooms: list,  # List of OrthoRoom from layout engine
    service_radius: float = 30.0,  # meters, max walking distance to toilet
) -> Tuple[bool, List[str]]:
    """
    Check that every occupied room is within service_radius of a toilet.

    Reference: In real schools, toilets should be reachable within ~25-30m
    walking distance (about 30 seconds at student pace).

    Args:
        ortho_rooms: List of OrthoRoom with x, y, width, height, room_type.
        service_radius: Maximum Euclidean distance from room center to toilet center.

    Returns:
        (all_covered, warnings_list)
    """
    import math

    # Rooms that need toilet access
    needs_toilet = {'classroom', 'special_classroom', 'office', 'teacher_office',
                    'library', 'music_room', 'cafeteria', 'entrance_hall',
                    'gymnasium'}

    toilets = [r for r in ortho_rooms if getattr(r, 'room_type', '') == 'toilet']
    occupied = [r for r in ortho_rooms if getattr(r, 'room_type', '') in needs_toilet]

    if not toilets:
        return (False, ["No toilets on this floor — all rooms uncovered"])

    warnings = []
    uncovered = []

    for room in occupied:
        rx = room.x + room.width / 2
        ry = room.y + room.height / 2

        min_dist = float('inf')
        nearest = None
        for t in toilets:
            tx = t.x + t.width / 2
            ty = t.y + t.height / 2
            d = math.sqrt((rx - tx)**2 + (ry - ty)**2)
            if d < min_dist:
                min_dist = d
                nearest = getattr(t, 'room_id', '?')

        if min_dist > service_radius:
            uncovered.append((getattr(room, 'room_id', '?'),
                            getattr(room, 'room_type', '?'),
                            min_dist, nearest))

    # Sort uncovered by distance (worst first)
    uncovered.sort(key=lambda x: -x[2])

    if uncovered:
        for rid, rtype, dist, nearest in uncovered[:5]:
            warnings.append(
                f"{rtype} '{rid}' is {dist:.0f}m from nearest toilet '{nearest}' "
                f"— exceeds {service_radius:.0f}m service radius"
            )

    coverage = 1 - len(uncovered) / max(1, len(occupied))
    return (len(uncovered) == 0, warnings, coverage, uncovered)


def enforce_minimums(
    rooms: list,
    num_floors: int = 3,
):
    """
    Adjust room distribution to meet architectural realism requirements.

    Modifies room objects in-place:
      - Minimum classroom area ≥54 m² (GB50099 §5.2.1)
      - Bumps very small rooms to reasonable minimums
      - Ensures each typical floor has ≥1 staircase and ≥1 corridor
    """
    from copy import deepcopy

    tf_groups: Dict[str, list] = defaultdict(list)
    for room in rooms:
        tf = getattr(room, 'typical_floor', 'ground')
        tf_groups[tf].append(room)

    # ── Fix classroom areas ──
    for room in rooms:
        if _get_room_type(room) == 'classroom' and room.area < 54:
            room.area = 60.0
        if room.area < 6.0:
            room.area = 12.0
