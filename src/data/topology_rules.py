"""
Topology Rule Engine — 拓扑规则引擎

Generates all edge types for a school building graph according to the
multi-modal edge definitions in task.md §3.2:

  - 物理连通边 (physical_connects): doors, passages, circulation
  - 声学阻断边 (acoustic_blocks): sound-isolating walls
  - 视线/采光边 (sight_lines): visual/light connections

=== Design Principle (§4 — 拒绝黑盒 API) ===
Every rule is a standalone, pure function with the signature:
    rule(rooms, env_nodes, params) -> List[(src_id, dst_id, edge_attrs)]

Each rule is:
  - Independently testable (no cross-rule side effects)
  - Explicitly debuggable (clear for-loop logic, not black-box tensor ops)
  - Documented with GB50099-2011 / GB50016-2014 references
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    DaylightLevel, NoiseLevel,
)

# Type alias for edge representation
# (src_node_id, dst_node_id, {attr_name: value})
Edge = Tuple[str, str, dict]


class TopologyRuleEngine:
    """
    Generates all edge types for a school building graph.

    The engine applies rules in a deterministic order:
      1. Physical connection rules (circulation spine, corridor network)
      2. Acoustic blocking rules (noisy↔quiet separation)
      3. Sight/lighting rules (daylight connections)

    Rules reference the building_rules.yaml parameters for tunable thresholds.
    """

    def __init__(self, rule_params: Optional[Dict] = None, seed: int = None):
        """
        Args:
            rule_params: Optional dict of parameters from building_rules.yaml.
            seed: Random seed for stochastic edge creation (daylight, redundant connections).
        """
        import random as _random
        params = rule_params or {}
        self._rng = _random.Random(seed) if seed is not None else _random.Random()

        # Acoustic parameters
        acoustic = params.get('acoustic', {})
        self.noise_gap_threshold: int = acoustic.get('noise_gap_threshold', 2)
        self.min_path_distance: int = acoustic.get('min_path_distance', 2)
        self.proximity_threshold: float = acoustic.get('proximity_threshold', 15.0)
        self.default_attenuation_db: float = acoustic.get('default_attenuation_db', 45.0)
        self.music_classroom_attenuation_db: float = acoustic.get(
            'music_classroom_attenuation_db', 55.0
        )

        # Daylight parameters
        daylight = params.get('daylight', {})
        self.high_requirement_min_edges: int = daylight.get('high_requirement_min_edges', 1)

        # Topology parameters
        topology = params.get('topology', {})
        self.corridor_segment_length: float = topology.get('corridor_segment_length', 10.0)

    # ========================================================================
    # Physical Connection Rules (物理连通边)
    # ========================================================================

    def connect_rooms_to_corridors(
        self,
        rooms: list,    # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P1: Every non-corridor room connects to corridors on overlapping floors.

        Reference: GB50099-2011 §8.2 — all teaching rooms must have direct
        access to a corridor for circulation and fire evacuation.

        GB50016-2014 §5.5 — high-occupancy rooms (≥50 persons) require
        at least TWO independent evacuation routes.

        Formula:
            For each room r where r.room_type != CORRIDOR:
                Find corridors on overlapping floor(s)
                If occupancy >= 50: connect to 2 nearest corridors
                Else: connect to 1 nearest corridor
        """
        import heapq

        edges: List[Edge] = []
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]
        non_corridors = [r for r in rooms if r.room_type != RoomType.CORRIDOR]

        for room in non_corridors:
            # Find corridors on overlapping floor(s)
            overlapping = [
                c for c in corridors if c.overlaps_floor(room)
            ]

            if not overlapping:
                # Fallback: any corridor
                overlapping = corridors

            if not overlapping:
                continue

            # High-occupancy rooms: connect to TWO nearest corridors
            num_needed = 2 if room.occupancy >= 50 else 1
            num_needed = min(num_needed, len(overlapping))

            top_n = heapq.nsmallest(
                num_needed, overlapping,
                key=lambda c: room.euclidean_distance_to(c),
            )

            for nearest in top_n:
                dist = room.euclidean_distance_to(nearest)
                edges.append((
                    room.room_id,
                    nearest.room_id,
                    {
                        'distance_weight': dist,
                        'is_stair_connection': 0.0,
                    },
                ))

        return edges

    def connect_corridor_network(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P2: Corridors on the same floor connect to form a circulation
        network. The topology is randomized per graph to create diversity:

          - 'spine' (60%): linear chain sorted by X (traditional spine)
          - 'loop' (25%): spine with tail-to-head closure (ring corridor)
          - 'branch' (15%): Y-shaped branch from a central hub corridor

        Each topology creates different global circulation patterns that
        the GNN can learn to distinguish.
        """
        edges: List[Edge] = []
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]

        by_floor: Dict[int, list] = defaultdict(list)
        for c in corridors:
            by_floor[c.floor].append(c)

        for _floor_num, floor_corridors in by_floor.items():
            if len(floor_corridors) < 2:
                continue

            sorted_c = sorted(floor_corridors, key=lambda r: r.centroid[0])
            n = len(sorted_c)
            topo_type = self._rng.choices(
                ['spine', 'spine', 'spine', 'loop', 'loop', 'branch'],
                k=1
            )[0]

            if topo_type == 'spine':
                # Linear chain: c0−c1−c2−...−cn
                for i in range(n - 1):
                    edges.append((sorted_c[i].room_id, sorted_c[i+1].room_id,
                                  {'distance_weight': sorted_c[i].euclidean_distance_to(sorted_c[i+1]),
                                   'is_stair_connection': 0.0}))

            elif topo_type == 'loop':
                # Ring: spine + tail→head closure
                for i in range(n - 1):
                    edges.append((sorted_c[i].room_id, sorted_c[i+1].room_id,
                                  {'distance_weight': sorted_c[i].euclidean_distance_to(sorted_c[i+1]),
                                   'is_stair_connection': 0.0}))
                # Close the loop
                edges.append((sorted_c[-1].room_id, sorted_c[0].room_id,
                              {'distance_weight': sorted_c[-1].euclidean_distance_to(sorted_c[0]),
                               'is_stair_connection': 0.0}))

            elif topo_type == 'branch':
                # Y-branch: hub at n//2, all others connect to hub
                hub = sorted_c[n // 2]
                for i in range(n):
                    if i != n // 2:
                        edges.append((sorted_c[i].room_id, hub.room_id,
                                      {'distance_weight': sorted_c[i].euclidean_distance_to(hub),
                                       'is_stair_connection': 0.0}))

        return edges

    def connect_corridor_cross_links(
        self,
        rooms: list,
        env_nodes: list,
    ) -> List[Edge]:
        """
        Rule P2b: Add random cross-links between non-adjacent corridor
        segments on the same floor. Creates redundant paths → mesh-like
        circulation network with higher algebraic connectivity (lambda_2).

        40% probability per eligible pair.
        """
        edges: List[Edge] = []
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]
        by_floor: Dict[int, list] = defaultdict(list)
        for c in corridors:
            by_floor[c.floor].append(c)

        for _floor_num, floor_corridors in by_floor.items():
            if len(floor_corridors) < 3:
                continue
            sorted_c = sorted(floor_corridors, key=lambda r: r.centroid[0])
            # Add cross-links between every other corridor (skip neighbors)
            for i in range(len(sorted_c) - 2):
                if self._rng.random() < 0.4:
                    j = i + 2 + self._rng.randint(0, min(2, len(sorted_c) - i - 3))
                    dist = sorted_c[i].euclidean_distance_to(sorted_c[j])
                    edges.append((sorted_c[i].room_id, sorted_c[j].room_id,
                                  {'distance_weight': dist, 'is_stair_connection': 0.0}))

        return edges

    def connect_redundant_room_links(
        self,
        rooms: list,
        env_nodes: list,
    ) -> List[Edge]:
        """
        Rule P1b: Add redundant room→corridor connections.
        Some rooms (30% chance) get a second physical connection to another
        corridor beyond their nearest one. This creates mesh-like local
        circulation, providing alternative evacuation routes.

        Only applies when multiple corridors exist on the same floor.
        """
        edges: List[Edge] = []
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]
        non_corridors = [r for r in rooms if r.room_type != RoomType.CORRIDOR]

        for room in non_corridors:
            if self._rng.random() > 0.3:
                continue

            overlapping = [c for c in corridors if c.overlaps_floor(room)]
            if len(overlapping) < 2:
                continue

            # Find the SECOND-nearest corridor
            sorted_c = sorted(overlapping, key=lambda c: room.euclidean_distance_to(c))
            second = sorted_c[1] if len(sorted_c) > 1 else None
            if second is not None:
                dist = room.euclidean_distance_to(second)
                edges.append((room.room_id, second.room_id,
                              {'distance_weight': dist, 'is_stair_connection': 0.0}))

        return edges

    def connect_staircases_to_corridors(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P3: Staircases connect to corridors on EVERY floor they span
        (based on floor_range), ensuring cross-floor physical connectivity.

        Reference: GB50016-2014 §5.5 — staircases must be accessible from
        all served floors via corridors. The physical graph must remain
        connected across all floors.
        """
        edges: List[Edge] = []
        staircases = [r for r in rooms if r.room_type == RoomType.STAIRCASE]
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]

        for stair in staircases:
            # Connect to corridors on every floor the stair spans
            for phys_floor in range(stair.floor_range[0], stair.floor_range[1] + 1):
                # Find corridors that overlap this physical floor
                floor_corridors = [
                    c for c in corridors
                    if c.floor_range[0] <= phys_floor <= c.floor_range[1]
                ]
                nearest = min(
                    floor_corridors,
                    key=lambda c: stair.euclidean_distance_to(c),
                    default=None,
                )
                if nearest is not None:
                    dist = stair.euclidean_distance_to(nearest)
                    edges.append((
                        stair.room_id,
                        nearest.room_id,
                        {
                            'distance_weight': dist,
                            'is_stair_connection': 1.0,  # Marked for fire safety
                        },
                    ))

        return edges

    def connect_staircase_vertical_chain(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P3b: Connect staircases of adjacent typical floors to form
        a vertical circulation chain, ensuring the physical graph is
        connected across floors.

        Connects: ground_stair ↔ teaching_stair ↔ top_stair
        """
        edges: List[Edge] = []
        staircases = [r for r in rooms if r.room_type == RoomType.STAIRCASE]

        # Group by typical floor
        ground_stairs = [s for s in staircases if s.typical_floor == 'ground']
        teaching_stairs = [s for s in staircases if s.typical_floor == 'teaching']
        top_stairs = [s for s in staircases if s.typical_floor == 'top']

        # Connect ground → teaching
        for gs in ground_stairs:
            for ts in teaching_stairs:
                dist = gs.euclidean_distance_to(ts)
                edges.append((
                    gs.room_id, ts.room_id,
                    {'distance_weight': dist, 'is_stair_connection': 1.0},
                ))

        # Connect teaching → top
        for ts in teaching_stairs:
            for ps in top_stairs:
                dist = ts.euclidean_distance_to(ps)
                edges.append((
                    ts.room_id, ps.room_id,
                    {'distance_weight': dist, 'is_stair_connection': 1.0},
                ))

        return edges

    def connect_entrance_to_corridor_and_road(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P4: Entrance hall connects to nearest ground-floor corridor
        AND to the main_road_access environmental node.

        Reference: GB50016-2014 §5.5 — main entrance must connect directly
        to the external evacuation route.
        """
        edges: List[Edge] = []
        entrances = [r for r in rooms if r.room_type == RoomType.ENTRANCE_HALL]
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]
        road_nodes = [e for e in env_nodes if e.env_type == EnvNodeType.MAIN_ROAD_ACCESS]

        for entrance in entrances:
            # Connect to nearest corridor on same floor (ground floor)
            same_floor = [c for c in corridors if c.floor == entrance.floor]
            nearest_c = min(
                same_floor,
                key=lambda c: entrance.euclidean_distance_to(c),
                default=None,
            )
            if nearest_c is not None:
                dist = entrance.euclidean_distance_to(nearest_c)
                edges.append((
                    entrance.room_id,
                    nearest_c.room_id,
                    {
                        'distance_weight': dist,
                        'is_stair_connection': 0.0,
                    },
                ))

            # Connect to main road access env node
            for road in road_nodes:
                edges.append((
                    entrance.room_id,
                    road.env_id,
                    {'access_type': 1.0},  # 1 = main entrance connection
                ))

        return edges

    def connect_ground_special_rooms(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P5 & P6: Special ground-floor rooms (cafeteria, gymnasium)
        connect to corridors and (for gymnasium) to main road access.
        """
        edges: List[Edge] = []
        ground_rooms = [
            r for r in rooms
            if r.room_type in (RoomType.CAFETERIA, RoomType.GYMNASIUM)
        ]
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]
        road_nodes = [e for e in env_nodes if e.env_type == EnvNodeType.MAIN_ROAD_ACCESS]

        for room in ground_rooms:
            # Connect to nearest corridor on same floor
            same_floor = [c for c in corridors if c.floor == room.floor]
            nearest_c = min(
                same_floor,
                key=lambda c: room.euclidean_distance_to(c),
                default=None,
            )
            if nearest_c is not None:
                dist = room.euclidean_distance_to(nearest_c)
                edges.append((
                    room.room_id,
                    nearest_c.room_id,
                    {
                        'distance_weight': dist,
                        'is_stair_connection': 0.0,
                    },
                ))

            # Gymnasium and cafeteria also connect to main road (emergency egress)
            if room.room_type in (RoomType.GYMNASIUM, RoomType.CAFETERIA):
                for road in road_nodes:
                    edges.append((
                        room.room_id,
                        road.env_id,
                        {'access_type': 0.5 if room.room_type == RoomType.GYMNASIUM
                         else 0.7},  # cafeteria gets higher access priority
                    ))

        return edges

    def connect_service_rooms(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule P7: Toilets and storage rooms connect to nearest corridor
        on the same floor.
        """
        edges: List[Edge] = []
        service_rooms = [
            r for r in rooms
            if r.room_type in (RoomType.TOILET, RoomType.STORAGE)
        ]
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]

        for room in service_rooms:
            same_floor = [c for c in corridors if c.floor == room.floor]
            nearest = min(
                same_floor,
                key=lambda c: room.euclidean_distance_to(c),
                default=None,
            )
            if nearest is not None:
                dist = room.euclidean_distance_to(nearest)
                edges.append((
                    room.room_id,
                    nearest.room_id,
                    {
                        'distance_weight': dist,
                        'is_stair_connection': 0.0,
                    },
                ))

        return edges

    # ========================================================================
    # Acoustic Blocking Rules (声学阻断边)
    # ========================================================================

    def acoustic_separation_noisy_quiet(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule A1: For every pair (noisy_room, quiet_room) where the noise gap
        exceeds the threshold, add an acoustic_blocks edge.

        Formula (§4 硬约束, 声学):
            For all pairs (r1, r2) on the same floor:
                if noise_level(r1) - noise_tolerance(r2) >= noise_gap_threshold
                AND Euclidean distance < proximity_threshold:
                    Add acoustic_blocks edge with default_attenuation_db

        Reference: GB50099-2011 §7.3 — music rooms, gymnasiums must be
        acoustically isolated from teaching areas.
        """
        edges: List[Edge] = []

        for i, r1 in enumerate(rooms):
            for r2 in rooms[i + 1:]:
                # Only check same-floor neighbors within proximity
                if not r1.same_floor(r2):
                    continue
                dist = r1.euclidean_distance_to(r2)
                if dist > self.proximity_threshold:
                    continue

                # Check noise gap in both directions
                gap_12 = r1.noise_level.value - r2.noise_tolerance.value
                gap_21 = r2.noise_level.value - r1.noise_tolerance.value

                if gap_12 >= self.noise_gap_threshold or gap_21 >= self.noise_gap_threshold:
                    # Determine attenuation based on room types
                    atten = self.default_attenuation_db
                    if (r1.room_type == RoomType.MUSIC_ROOM and
                            r2.room_type == RoomType.CLASSROOM):
                        atten = self.music_classroom_attenuation_db
                    elif (r2.room_type == RoomType.MUSIC_ROOM and
                          r1.room_type == RoomType.CLASSROOM):
                        atten = self.music_classroom_attenuation_db

                    edges.append((
                        r1.room_id,
                        r2.room_id,
                        {'attenuation_db': atten},
                    ))

        return edges

    # ========================================================================
    # Sight / Lighting Rules (视线/采光边)
    # ========================================================================

    def daylight_connection_high_requirement(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule S1: Every room with daylight_level >= HIGH gets a sight_lines
        edge to the south_facing environmental node.

        Formula (§4 软约束, 采光):
            For each room r where r.requires_daylight():
                Connect r to south_facing env node
                orientation_preference = cosine of angle between
                    (r→south) vector and true south (positive Y)

        Reference: GB50099-2011 §5.1 — classrooms must have good natural
        lighting, preferably facing south.
        """
        edges: List[Edge] = []
        south_nodes = [e for e in env_nodes if e.env_type == EnvNodeType.SOUTH_FACING]

        if not south_nodes:
            return edges

        south = south_nodes[0]  # There should be exactly one south node

        for room in rooms:
            if not room.requires_daylight:
                continue

            # Compute orientation preference: cosine of angle to south (Y+)
            sx, sy = south.position
            rx, ry = room.centroid
            dx, dy = sx - rx, sy - ry
            mag = math.sqrt(dx * dx + dy * dy)
            if mag < 1e-9:
                orientation = 0.0
            else:
                # Normalize and compute cos(angle from Y+):
                # dot product with (0, 1) = dy / mag
                orientation = dy / mag  # Range [-1, 1], 1 = perfect south

            dist = mag  # Euclidean distance to south node

            edges.append((
                room.room_id,
                south.env_id,
                {
                    'orientation_preference': orientation,
                    'distance': dist,
                },
            ))

        return edges

    def daylight_connection_medium_requirement(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule S2: Rooms with daylight_level >= MEDIUM get sight_lines
        to the nearest corridor (light well proxy).

        HIGH rooms: 60% probability of second sight path via corridor.
        MEDIUM rooms: always get corridor connection (their only daylight).
        Randomization creates meaningful variance in daylight_quality scores.
        """
        edges: List[Edge] = []
        corridors = [r for r in rooms if r.room_type == RoomType.CORRIDOR]

        for room in rooms:
            if room.spec.daylight_level < DaylightLevel.MEDIUM:
                continue

            if room.spec.daylight_level >= DaylightLevel.HIGH:
                # 60% chance of additional corridor connection
                if self._rng.random() > 0.6:
                    continue

            same_floor = [c for c in corridors if c.floor == room.floor]
            nearest = min(
                same_floor,
                key=lambda c: room.euclidean_distance_to(c),
                default=None,
            )
            if nearest is not None:
                dist = room.euclidean_distance_to(nearest)
                transparency = 0.4 if room.spec.daylight_level >= DaylightLevel.HIGH else 0.5
                edges.append((
                    room.room_id,
                    nearest.room_id,
                    {
                        'transparency': transparency,
                        'sight_distance': dist,
                    },
                ))

        return edges

    def library_green_space_connection(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> List[Edge]:
        """
        Rule S4: Library reading areas connect to green_space env nodes
        with sight_lines for view quality.

        This is a soft constraint that enhances the architectural quality.
        """
        edges: List[Edge] = []
        libraries = [r for r in rooms if r.room_type == RoomType.LIBRARY]
        green_spaces = [e for e in env_nodes if e.env_type == EnvNodeType.GREEN_SPACE]

        for library in libraries:
            for green in green_spaces:
                gx, gy = green.position
                lx, ly = library.centroid
                dx, dy = gx - lx, gy - ly
                dist = math.sqrt(dx * dx + dy * dy)

                edges.append((
                    library.room_id,
                    green.env_id,
                    {
                        'orientation_preference': 0.5,  # neutral for green space
                        'distance': dist,
                    },
                ))

        return edges

    # ========================================================================
    # Master rule application
    # ========================================================================

    def apply_all_rules(
        self,
        rooms: list,     # List[RoomNode]
        env_nodes: list,  # List[EnvironmentalNode]
    ) -> Dict[EdgeCategory, List[Edge]]:
        """
        Apply all topology rules in the correct order and return edges
        grouped by category.

        Rule application order:
          1. PHYSICAL_CONNECTS:
             - connect_rooms_to_corridors
             - connect_corridor_network
             - connect_staircases_to_corridors
             - connect_entrance_to_corridor_and_road
             - connect_ground_special_rooms
             - connect_service_rooms

          2. ACOUSTIC_BLOCKS:
             - acoustic_separation_noisy_quiet

          3. SIGHT_LINES:
             - daylight_connection_high_requirement
             - daylight_connection_medium_requirement
             - library_green_space_connection

        Args:
            rooms: List of RoomNode instances.
            env_nodes: List of EnvironmentalNode instances.

        Returns:
            Dict mapping each EdgeCategory to its edge list.
        """
        # --- Physical connections ---
        phys_edges: List[Edge] = []
        phys_edges.extend(self.connect_rooms_to_corridors(rooms, env_nodes))
        phys_edges.extend(self.connect_corridor_network(rooms, env_nodes))
        phys_edges.extend(self.connect_corridor_cross_links(rooms, env_nodes))
        phys_edges.extend(self.connect_redundant_room_links(rooms, env_nodes))
        phys_edges.extend(self.connect_staircases_to_corridors(rooms, env_nodes))
        phys_edges.extend(self.connect_staircase_vertical_chain(rooms, env_nodes))
        phys_edges.extend(self.connect_entrance_to_corridor_and_road(rooms, env_nodes))
        phys_edges.extend(self.connect_ground_special_rooms(rooms, env_nodes))
        phys_edges.extend(self.connect_service_rooms(rooms, env_nodes))

        # --- Acoustic blocks ---
        acous_edges = self.acoustic_separation_noisy_quiet(rooms, env_nodes)

        # --- Sight lines ---
        sight_edges: List[Edge] = []
        sight_edges.extend(self.daylight_connection_high_requirement(rooms, env_nodes))
        sight_edges.extend(self.daylight_connection_medium_requirement(rooms, env_nodes))
        sight_edges.extend(self.library_green_space_connection(rooms, env_nodes))

        return {
            EdgeCategory.PHYSICAL_CONNECTS: phys_edges,
            EdgeCategory.ACOUSTIC_BLOCKS: acous_edges,
            EdgeCategory.SIGHT_LINES: sight_edges,
        }
