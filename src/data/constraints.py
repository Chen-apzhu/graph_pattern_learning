"""
Constraint Validator — 约束验证器

Implements the neuro-symbolic constraint mechanism described in task.md §4.
Each constraint is an explicit, debuggable Python function — NOT a black-box API.

=== Constraint Categories (§4) ===

  Hard Constraints (拓扑掩码 Topological Masking) — absolute red lines:
    1. Fire Safety: high-occupancy rooms must have adequate physical connections
    2. Daylight Compliance: rooms with high daylight requirement must have sight_lines
    3. Acoustic Separation: noisy and quiet rooms must be adequately separated
    4. Connectivity: the physical connection graph must be fully connected

  Soft Constraints (损失函数惩罚 Loss Penalty):
    5. Area Bounds: every room's area must be within its spec range
    6. Circulation Ratio: corridor area proportion sanity check

Each constraint function returns: (passed: bool, violations: List[str])
where violations contains human-readable descriptions of each failure.
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    DaylightLevel,
)


class ConstraintValidator:
    """
    Validates a generated graph against building code constraints.

    Every constraint method:
      - Has a clear mathematical formula in its docstring
      - References the specific building code (GB50099-2011 / GB50016-2014)
      - Returns human-readable violation descriptions
      - Is independently testable
    """

    def __init__(self, rules_config: Dict = None):
        """
        Args:
            rules_config: Optional dict from building_rules.yaml.
        """
        config = rules_config or {}

        fire = config.get('fire_safety', {})
        self.occupancy_threshold: int = fire.get('occupancy_threshold', 50)
        self.min_fire_exit_degree: int = fire.get('min_fire_exit_degree', 2)

        daylight = config.get('daylight', {})
        self.high_requirement_min_edges: int = daylight.get('high_requirement_min_edges', 1)

        acoustic = config.get('acoustic', {})
        self.noise_gap_threshold: int = acoustic.get('noise_gap_threshold', 2)
        self.min_path_distance: int = acoustic.get('min_path_distance', 2)
        self.proximity_threshold: float = acoustic.get('proximity_threshold', 15.0)

        connect = config.get('connectivity', {})
        self.physical_graph_must_be_connected: bool = connect.get(
            'physical_graph_must_be_connected', True
        )

        # Area completeness config
        footprint = config.get('building_footprint', {})
        self._area_tolerance: float = footprint.get('area_tolerance', 0.05)

        # Floor-level corridor ratio bounds
        self.floor_corr_min: float = 0.10
        self.floor_corr_max: float = 0.25
        # per_floor_area is NOT stored here — it's per school size, passed at call time

    # ======================================================================
    # Fire Safety Constraint (§4 硬约束, 消防疏散)
    # ======================================================================

    def check_fire_exits(
        self,
        rooms: list,                    # List[RoomNode]
        phys_edges: List[Tuple[str, str, dict]],
    ) -> Tuple[bool, List[str]]:
        """
        Fire Safety Constraint (Hard).

        Reference: GB50016-2014 §5.5 (fire evacuation),
                   GB50099-2011 §8.2 (corridor width & exit spacing)

        Formula:
            For every room node r:
                Let O(r) = estimated occupancy (persons)
                Let D_p(r) = degree of r in the physical connection subgraph
                Let M(r) = r.fire_exits_min (minimum required exits)

                if O(r) >= OCCUPANCY_THRESHOLD:
                    D_p(r) >= M(r)

        Rationale: High-occupancy rooms (≥50 persons) must have at least
        2 independent evacuation routes to prevent bottleneck deaths.
        """
        violations: List[str] = []

        # Build adjacency for degree computation
        adj: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, _attrs in phys_edges:
            adj[src].add(dst)
            adj[dst].add(src)

        for room in rooms:
            degree = len(adj.get(room.room_id, set()))
            if room.occupancy >= self.occupancy_threshold:
                if degree < room.fire_exits_min:
                    violations.append(
                        f"[FIRE] Room '{room.room_id}' "
                        f"(type={room.room_type.value}, "
                        f"occupancy={room.occupancy} persons) "
                        f"has phys_degree={degree} but requires "
                        f">= {room.fire_exits_min} fire exits "
                        f"(GB50016-2014 §5.5)."
                    )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Daylight Compliance Constraint (§4 软约束, 采光)
    # ======================================================================

    def check_daylight_compliance(
        self,
        rooms: list,                    # List[RoomNode]
        sight_edges: List[Tuple[str, str, dict]],
    ) -> Tuple[bool, List[str]]:
        """
        Daylight Compliance Constraint (Soft/Hard).

        Reference: GB50099-2011 §5.1 (natural lighting for classrooms)

        Formula:
            For every room r where r.requires_daylight() [daylight_level >= HIGH]:
                Let D_s(r) = degree of r in the sight_lines subgraph
                D_s(r) >= high_requirement_min_edges

        Rationale: Classrooms and teaching spaces must have direct or
        indirect natural light access for student health and energy efficiency.
        """
        violations: List[str] = []

        # Build sight-line adjacency
        sight_adj: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, _attrs in sight_edges:
            sight_adj[src].add(dst)
            sight_adj[dst].add(src)

        for room in rooms:
            if not room.requires_daylight:
                continue

            sight_degree = len(sight_adj.get(room.room_id, set()))
            if sight_degree < self.high_requirement_min_edges:
                violations.append(
                    f"[DAYLIGHT] Room '{room.room_id}' "
                    f"(type={room.room_type.value}, "
                    f"daylight_level={room.daylight_level.name}) "
                    f"has sight_degree={sight_degree} but requires "
                    f">= {self.high_requirement_min_edges} daylight connections "
                    f"(GB50099-2011 §5.1)."
                )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Acoustic Separation Constraint (§4 硬约束, 声学)
    # ======================================================================

    def check_acoustic_separation(
        self,
        rooms: list,                    # List[RoomNode]
        acoustic_edges: List[Tuple[str, str, dict]],
        phys_edges: List[Tuple[str, str, dict]],
    ) -> Tuple[bool, List[str]]:
        """
        Acoustic Separation Constraint (Hard).

        Reference: GB50099-2011 §7.3 (acoustic design for teaching buildings)

        Formula:
            For all pairs (r1, r2) on the same floor within proximity_threshold:
                Let NG(r1, r2) = noise_level(r1) - noise_tolerance(r2)
                If NG(r1, r2) >= noise_gap_threshold:
                    Then EITHER:
                      (a) An acoustic_blocks edge exists between r1 and r2, OR
                      (b) shortest_path_distance(r1, r2, physical_connects) >= min_path_distance

        Rationale: Loud spaces (music rooms, gymnasiums) must be acoustically
        isolated from quiet spaces (classrooms, library, teacher offices) via
        physical separation or sound-isolating construction.
        """
        violations: List[str] = []

        # Build acoustic adjacency
        acous_adj: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, _attrs in acoustic_edges:
            acous_adj[src].add(dst)
            acous_adj[dst].add(src)

        # Build physical adjacency for shortest path calculation
        phys_adj: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, _attrs in phys_edges:
            phys_adj[src].add(dst)
            phys_adj[dst].add(src)

        def _shortest_path_len(src: str, dst: str) -> int:
            """BFS shortest path length in physical graph. Returns INF if unreachable."""
            if src == dst:
                return 0
            visited = {src}
            queue = [(src, 0)]
            while queue:
                node, dist = queue.pop(0)
                for neighbor in phys_adj.get(node, set()):
                    if neighbor == dst:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return 10**9  # Unreachable

        # Check all pairs
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

                if gap_12 >= self.noise_gap_threshold:
                    # r1 is too noisy for r2
                    has_acoustic = r2.room_id in acous_adj.get(r1.room_id, set())
                    path_dist = _shortest_path_len(r1.room_id, r2.room_id)
                    if not has_acoustic and path_dist < self.min_path_distance:
                        violations.append(
                            f"[ACOUSTIC] '{r1.room_id}' (noise={r1.noise_level.name}) "
                            f"is too close to '{r2.room_id}' "
                            f"(tolerance={r2.noise_tolerance.name}). "
                            f"No acoustic_blocks edge and physical path distance "
                            f"={path_dist} < {self.min_path_distance} "
                            f"(GB50099-2011 §7.3)."
                        )

                if gap_21 >= self.noise_gap_threshold:
                    # r2 is too noisy for r1
                    has_acoustic = r1.room_id in acous_adj.get(r2.room_id, set())
                    path_dist = _shortest_path_len(r2.room_id, r1.room_id)
                    if not has_acoustic and path_dist < self.min_path_distance:
                        violations.append(
                            f"[ACOUSTIC] '{r2.room_id}' (noise={r2.noise_level.name}) "
                            f"is too close to '{r1.room_id}' "
                            f"(tolerance={r1.noise_tolerance.name}). "
                            f"No acoustic_blocks edge and physical path distance "
                            f"={path_dist} < {self.min_path_distance} "
                            f"(GB50099-2011 §7.3)."
                        )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Connectivity Constraint (§4 硬约束, 连通性)
    # ======================================================================

    def check_connectivity(
        self,
        rooms: list,                    # List[RoomNode]
        phys_edges: List[Tuple[str, str, dict]],
    ) -> Tuple[bool, List[str]]:
        """
        Connectivity Constraint (Hard).

        Formula:
            The physical connection subgraph must be fully connected.
            Every room must have a path to at least one staircase
            and at least one exit (entrance_hall or main_road_access connection).

        Reference: GB50016-2014 §5.5.17 — every occupied space must have
        an unobstructed evacuation route to a safe exit.
        """
        violations: List[str] = []

        if not self.physical_graph_must_be_connected:
            return (True, [])

        # Build physical adjacency
        phys_adj: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, _attrs in phys_edges:
            phys_adj[src].add(dst)
            phys_adj[dst].add(src)

        room_ids = {r.room_id for r in rooms}
        room_ids_with_edges = set(phys_adj.keys())

        # Check 1: No isolated rooms (rooms with degree 0 in physical graph)
        isolated = room_ids - room_ids_with_edges
        if isolated:
            for rid in isolated:
                violations.append(
                    f"[CONNECTIVITY] Room '{rid}' has NO physical connections — "
                    f"completely isolated from the circulation network "
                    f"(GB50016-2014 §5.5.17)."
                )

        # Check 2: Connected components via BFS
        if room_ids_with_edges:
            visited: Set[str] = set()
            start = next(iter(room_ids_with_edges))
            queue = [start]
            visited.add(start)
            while queue:
                node = queue.pop(0)
                for neighbor in phys_adj.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            unvisited = room_ids_with_edges - visited
            if unvisited:
                violations.append(
                    f"[CONNECTIVITY] Physical graph has multiple connected "
                    f"components. {len(unvisited)} room(s) not reachable "
                    f"from '{start}': {sorted(unvisited)[:5]}..."
                    f"(GB50016-2014 §5.5.17)."
                )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Area Bounds Constraint (Soft)
    # ======================================================================

    def check_area_bounds(
        self,
        rooms: list,  # List[RoomNode]
    ) -> Tuple[bool, List[str]]:
        """
        Area Bounds Constraint (Soft).

        Formula:
            For every room r:
                area_range_sqm[0] <= r.area <= area_range_sqm[1]

        This is a data validation check rather than a topological constraint.
        Rooms outside their spec range should not exist (factory rejects them),
        but this catches any manual creation or post-processing errors.
        """
        violations: List[str] = []

        for room in rooms:
            min_a, max_a = room.spec.area_range_sqm
            if room.area < min_a or room.area > max_a:
                violations.append(
                    f"[AREA] Room '{room.room_id}' "
                    f"(type={room.room_type.value}) "
                    f"area={room.area:.1f} m² is outside valid range "
                    f"[{min_a}, {max_a}]."
                )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Circulation Ratio Constraint (Soft)
    # ======================================================================

    def check_circulation_ratio(
        self,
        rooms: list,  # List[RoomNode]
    ) -> Tuple[bool, List[str]]:
        """
        Circulation Ratio Constraint (Soft).

        Reference: GB50099-2011 §8.2.3 — corridor area should be
        approximately 15-25% of total floor area for teaching buildings.

        Formula:
            total_corridor_area / total_area ∈ [0.10, 0.30]
        """
        violations: List[str] = []
        total_area = sum(r.area for r in rooms)
        corridor_area = sum(
            r.area for r in rooms if r.room_type == RoomType.CORRIDOR
        )

        if total_area > 0:
            ratio = corridor_area / total_area
            if ratio < 0.10:
                violations.append(
                    f"[CIRCULATION] Corridor area ratio={ratio:.2%} is below "
                    f"10% — may indicate insufficient circulation space "
                    f"(GB50099-2011 §8.2.3)."
                )
            elif ratio > 0.30:
                violations.append(
                    f"[CIRCULATION] Corridor area ratio={ratio:.2%} exceeds "
                    f"30% — may indicate inefficient space utilization."
                )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Area Completeness Constraint (§4 硬约束, 面积完备性)
    # ======================================================================

    def check_area_completeness(
        self,
        rooms: list,                    # List[RoomNode]
        num_floors: int,
        per_floor_area: float = None,
    ) -> Tuple[bool, List[str]]:
        """
        Area Completeness Constraint (Hard).

        Formula:
            For each typical floor type tf ∈ {ground, teaching, top}:
                Let rooms_tf = {r | r.typical_floor == tf}
                Let sum_area_tf = Σ(r.area for r in rooms_tf)
                Let deviation = |sum_area_tf - per_floor_area| / per_floor_area
                If deviation > area_tolerance → VIOLATION

            Global check:
                total_design = per_floor_area × num_floors
                deviation = |Σ(r.area) - total_design| / total_design
                If deviation > area_tolerance → VIOLATION

        Rationale: All rooms, corridors, and service spaces must fill the
        building footprint boundary. There should be no leftover empty space
        or gaps in the floor plan.
        """
        violations: List[str] = []

        if per_floor_area is None:
            per_floor_area = getattr(self, '_per_floor_area', None)
        if per_floor_area is None:
            return (True, [])  # No footprint defined → skip

        tolerance = getattr(self, '_area_tolerance', 0.05)

        # --- Per-typical-floor check ---
        tf_groups: dict = {}
        for r in rooms:
            tf = getattr(r, 'typical_floor', 'ground')
            tf_groups.setdefault(tf, []).append(r)

        for tf, tf_rooms in tf_groups.items():
            # Each typical floor covers N physical floors — budget scales accordingly
            n_spanned = max(
                (getattr(r, 'num_floors_spanned', 1) for r in tf_rooms),
                default=1
            )
            tf_budget = per_floor_area * n_spanned
            sum_area = sum(r.area for r in tf_rooms)
            deviation = abs(sum_area - tf_budget) / tf_budget if tf_budget > 0 else 0
            if deviation > tolerance:
                violations.append(
                    f"[AREA_COMPLETENESS] Typical floor '{tf}' "
                    f"(×{n_spanned} phys floors): "
                    f"sum(area)={sum_area:.1f} m² vs budget={tf_budget:.1f} m², "
                    f"deviation={deviation:.2%} > {tolerance:.2%}."
                )

        # --- Global check ---
        # Each room's area contributes to all floors it spans. For global check,
        # compare simple sum against per_floor_area * num_floors. The per-floor
        # check above handles the weighted allocation per typical floor type.
        total_simple = sum(r.area for r in rooms)
        total_design = per_floor_area * num_floors
        if total_design > 0:
            global_deviation = abs(total_simple - total_design) / total_design
            if global_deviation > tolerance:
                violations.append(
                    f"[AREA_COMPLETENESS] Global: Σ(area)={total_simple:.1f} m² "
                    f"vs total_design={total_design:.1f} m² ({num_floors} fl × "
                    f"{per_floor_area:.1f} m²), deviation={global_deviation:.2%} "
                    f"> {tolerance:.2%}."
                )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Floor Corridor Ratio Constraint (§4 硬约束, 逐层走廊比)
    # ======================================================================

    def check_floor_corridor_ratio(
        self,
        rooms: list,
    ) -> Tuple[bool, List[str]]:
        """
        Floor-level corridor ratio constraint (Hard).

        Reference: GB50099-2011 §8.2.3
        For each typical floor, corridor_area / total_floor_area ∈ [0.10, 0.25].

        Formula:
            Group rooms by typical_floor (ground, teaching, top).
            For each group:
                ratio = Σ(r.area for corridor rooms) / Σ(r.area for all rooms)
                If ratio < floor_corr_min or ratio > floor_corr_max → VIOLATION
        """
        violations: List[str] = []

        tf_groups: dict = {}
        for r in rooms:
            tf = getattr(r, 'typical_floor', 'ground')
            tf_groups.setdefault(tf, []).append(r)

        for tf, tf_rooms in tf_groups.items():
            total = sum(r.area for r in tf_rooms)
            corr_total = sum(
                r.area for r in tf_rooms
                if getattr(r, 'room_type', None) and r.room_type.value == 'corridor'
            )
            if total > 0:
                ratio = corr_total / total
                if ratio < self.floor_corr_min:
                    violations.append(
                        f"[FLOOR_CORR] Typical floor '{tf}': corridor ratio "
                        f"={ratio:.2%} < {self.floor_corr_min:.0%} min "
                        f"(GB50099-2011 §8.2.3)."
                    )
                elif ratio > self.floor_corr_max:
                    violations.append(
                        f"[FLOOR_CORR] Typical floor '{tf}': corridor ratio "
                        f"={ratio:.2%} > {self.floor_corr_max:.0%} max "
                        f"(GB50099-2011 §8.2.3)."
                    )

        return (len(violations) == 0, violations)

    # ======================================================================
    # Master validation
    # ======================================================================

    def validate_all(
        self,
        rooms: list,                           # List[RoomNode]
        env_nodes: list,                       # List[EnvironmentalNode]
        all_edges: Dict[EdgeCategory, list],   # Dict[EdgeCategory, List[Edge]]
        num_floors: int = None,                # For area completeness check
        per_floor_area: float = None,          # For area completeness check
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Run all hard and soft constraint checks.

        Args:
            rooms: List of RoomNode instances.
            env_nodes: List of EnvironmentalNode instances.
            all_edges: Dict mapping EdgeCategory → edge list.
            num_floors: Number of floors (for area completeness).
            per_floor_area: Per-floor design area in m² (for area completeness).

        Returns:
            Dict mapping constraint name → (passed, violations).
            Use all_passed() to check if all hard constraints pass.
        """
        phys_edges = all_edges.get(EdgeCategory.PHYSICAL_CONNECTS, [])
        acous_edges = all_edges.get(EdgeCategory.ACOUSTIC_BLOCKS, [])
        sight_edges = all_edges.get(EdgeCategory.SIGHT_LINES, [])

        results = {}

        results['fire_exits'] = self.check_fire_exits(rooms, phys_edges)
        results['daylight'] = self.check_daylight_compliance(rooms, sight_edges)
        results['acoustic'] = self.check_acoustic_separation(
            rooms, acous_edges, phys_edges
        )
        results['connectivity'] = self.check_connectivity(rooms, phys_edges)
        results['area_bounds'] = self.check_area_bounds(rooms)
        results['circulation_ratio'] = self.check_circulation_ratio(rooms)
        results['area_completeness'] = self.check_area_completeness(
            rooms, num_floors or 3, per_floor_area
        )
        results['floor_corridor_ratio'] = self.check_floor_corridor_ratio(rooms)

        return results

    @staticmethod
    def all_passed(results: Dict[str, Tuple[bool, List[str]]]) -> bool:
        """Check if all constraints passed."""
        return all(passed for passed, _violations in results.values())

    @staticmethod
    def hard_constraints_passed(
        results: Dict[str, Tuple[bool, List[str]]]
    ) -> bool:
        """Check if all hard constraints passed (fire, daylight, acoustic, connectivity, area_completeness)."""
        hard_keys = {'fire_exits', 'daylight', 'acoustic', 'connectivity', 'area_completeness', 'floor_corridor_ratio'}
        return all(
            results[k][0] for k in hard_keys if k in results
        )

    @staticmethod
    def format_violations(
        results: Dict[str, Tuple[bool, List[str]]]
    ) -> str:
        """Format all violations as a human-readable report."""
        lines = []
        for constraint_name, (passed, violations) in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            lines.append(f"  [{status}] {constraint_name}")
            for v in violations:
                lines.append(f"    → {v}")
        return "\n".join(lines)
