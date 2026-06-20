"""
School Building Graph Generator — 学校建筑图生成器主编排器

Master orchestrator for synthetic school building graph generation.
Simulates the parametric logic of Rhino/Grasshopper to produce valid,
constraint-compliant school building graphs.

=== Generation Pipeline (task.md §5 Phase 1) ===

1. School Program    — Determine room counts from school size template
2. Room Generation   — Instantiate RoomNodes via RoomFactory
3. Environment Nodes — Generate EnvironmentalNodes via EnvNodeFactory
4. Spatial Layout    — Assign zones, floors, and approximate positions
5. Topology Rules    — Apply physical/acoustic/sight edge rules
6. Validation        — Check all hard constraints
7. Output            — Convert to HeteroData or return raw objects
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory, ZoneType, ROOM_TO_ZONE,
)
from utils.constants import DEFAULT_SEED, SCHOOL_SIZES

from data.room_factory import (
    RoomSpec, RoomNode, EnvironmentalNode,
    RoomCatalog, RoomFactory, EnvNodeFactory,
)
from data.topology_rules import TopologyRuleEngine, Edge
from data.constraints import ConstraintValidator
from data.feature_engineering import FeatureEngineer
from data.room_rules import enforce_minimums, validate_room_distribution

# Try importing PyG
try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class SchoolBuildingGenerator:
    """
    Master orchestrator for school building graph generation.

    Typical workflow:
        gen = SchoolBuildingGenerator(config_path='src/config')
        result = gen.generate(num_floors=3, school_size='medium')
        result.hetero_data  # PyG HeteroData
        result.summary()    # Human-readable summary

    If PyG is not installed, generate() returns the raw (rooms, env_nodes, edges)
    objects without creating the HeteroData.
    """

    # Maximum retries for constraint validation
    MAX_RETRIES = 5

    def __init__(
        self,
        config_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            config_dir: Directory containing room_catalog.yaml and building_rules.yaml.
                        Defaults to 'src/config/' relative to project root.
            seed: Random seed for reproducibility. Defaults to DEFAULT_SEED (42).
        """
        self.seed = seed if seed is not None else DEFAULT_SEED
        self.rng = np.random.default_rng(self.seed)

        # Resolve config directory
        if config_dir is None:
            # Default: look for src/config/ relative to this file
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'config'
            )
        self.config_dir = config_dir

        # Load configurations
        self.catalog = self._load_catalog()
        self.rule_params = self._load_rule_params()

        # Initialize sub-components
        self.room_factory = RoomFactory(self.catalog, self.rng)
        self.env_factory = EnvNodeFactory(
            self._get_site_bounds(), self.rng
        )
        self.rule_engine = TopologyRuleEngine(self.rule_params)
        self.validator = ConstraintValidator(self.rule_params)
        self.feature_engineer = FeatureEngineer()

        # Track generation state
        self._last_rooms: List[RoomNode] = []
        self._last_env_nodes: List[EnvironmentalNode] = []
        self._last_edges: Dict[EdgeCategory, list] = {}
        self._last_school_size: str = "medium"
        self._last_num_floors: int = 3

    # ========================================================================
    # Configuration loading
    # ========================================================================

    def _load_catalog(self) -> RoomCatalog:
        """Load room_catalog.yaml from config directory."""
        yaml_path = os.path.join(self.config_dir, 'room_catalog.yaml')
        if os.path.exists(yaml_path):
            return RoomCatalog.from_yaml(yaml_path)
        else:
            # Fallback: build catalog from built-in defaults
            return self._default_catalog()

    def _load_rule_params(self) -> Dict:
        """Load building_rules.yaml from config directory."""
        yaml_path = os.path.join(self.config_dir, 'building_rules.yaml')
        if os.path.exists(yaml_path):
            try:
                import yaml
            except ImportError:
                return {}
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _get_site_bounds(self) -> Tuple[float, float, float, float]:
        """Extract site bounds from rule params."""
        site = self.rule_params.get('site', {})
        bounds = site.get('bounds', [0.0, 0.0, 200.0, 150.0])
        return tuple(bounds)

    @staticmethod
    def _default_catalog() -> RoomCatalog:
        """
        Build a minimal RoomCatalog when no YAML file is available.
        Uses the same specs as room_catalog.yaml but hardcoded as fallback.
        """
        from data.room_factory import RoomSpec
        from utils.enums import DaylightLevel, NoiseLevel

        defaults = {
            RoomType.CLASSROOM: RoomSpec(
                RoomType.CLASSROOM, "普通教室", (54.0, 72.0), (1.0, 1.8),
                DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
                1.2, 2, [1, 2, 3],
            ),
            RoomType.SPECIAL_CLASSROOM: RoomSpec(
                RoomType.SPECIAL_CLASSROOM, "专用教室", (72.0, 96.0), (1.0, 1.8),
                DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
                1.5, 2, [1, 2, 3],
            ),
            RoomType.MUSIC_ROOM: RoomSpec(
                RoomType.MUSIC_ROOM, "音乐教室", (72.0, 90.0), (1.0, 1.6),
                DaylightLevel.MEDIUM, NoiseLevel.LOUD, NoiseLevel.QUIET,
                1.5, 2, [0, 1],
            ),
            RoomType.GYMNASIUM: RoomSpec(
                RoomType.GYMNASIUM, "体育馆", (400.0, 800.0), (1.4, 2.0),
                DaylightLevel.MEDIUM, NoiseLevel.VERY_LOUD, NoiseLevel.QUIET,
                3.0, 3, [0],
            ),
            RoomType.LIBRARY: RoomSpec(
                RoomType.LIBRARY, "图书馆", (100.0, 200.0), (1.0, 2.5),
                DaylightLevel.HIGH, NoiseLevel.QUIET, NoiseLevel.QUIET,
                2.0, 2, [0, 1],
            ),
            RoomType.OFFICE: RoomSpec(
                RoomType.OFFICE, "行政办公室", (15.0, 30.0), (1.0, 2.0),
                DaylightLevel.MEDIUM, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
                4.0, 1, [0, 1],
            ),
            RoomType.TEACHER_OFFICE: RoomSpec(
                RoomType.TEACHER_OFFICE, "教师办公室", (30.0, 60.0), (1.0, 2.0),
                DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.QUIET,
                3.0, 1, [1, 2, 3],
            ),
            RoomType.CORRIDOR: RoomSpec(
                RoomType.CORRIDOR, "走道", (12.0, 48.0), (3.0, 12.0),
                DaylightLevel.LOW, NoiseLevel.NOISY, NoiseLevel.LOUD,
                0.3, 2, [0, 1, 2, 3, 4],
            ),
            RoomType.STAIRCASE: RoomSpec(
                RoomType.STAIRCASE, "楼梯间", (18.0, 30.0), (1.0, 2.0),
                DaylightLevel.NONE, NoiseLevel.MODERATE, NoiseLevel.LOUD,
                0.5, 2, [0, 1, 2, 3, 4],
            ),
            RoomType.TOILET: RoomSpec(
                RoomType.TOILET, "卫生间", (12.0, 24.0), (1.0, 2.5),
                DaylightLevel.NONE, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
                0.8, 1, [0, 1, 2, 3, 4],
            ),
            RoomType.STORAGE: RoomSpec(
                RoomType.STORAGE, "储藏室", (6.0, 15.0), (1.0, 3.0),
                DaylightLevel.NONE, NoiseLevel.QUIET, NoiseLevel.LOUD,
                0.1, 1, [0, 1, 2, 3, 4],
            ),
            RoomType.CAFETERIA: RoomSpec(
                RoomType.CAFETERIA, "食堂", (200.0, 400.0), (1.0, 2.5),
                DaylightLevel.MEDIUM, NoiseLevel.NOISY, NoiseLevel.NOISY,
                1.0, 3, [0],
            ),
            RoomType.ENTRANCE_HALL: RoomSpec(
                RoomType.ENTRANCE_HALL, "门厅", (40.0, 80.0), (1.0, 3.0),
                DaylightLevel.MEDIUM, NoiseLevel.NOISY, NoiseLevel.LOUD,
                0.5, 2, [0],
            ),
        }
        return RoomCatalog(defaults)

    # ========================================================================
    # School program generation
    # ========================================================================

    def _generate_school_program(
        self, school_size: str
    ) -> Dict[RoomType, int]:
        """
        Determine room type counts from school size template.

        Templates defined in building_rules.yaml → school_sizes.

        Args:
            school_size: 'small', 'medium', or 'large'.

        Returns:
            Dict mapping RoomType → count.
        """
        size_configs = self.rule_params.get('school_sizes', {})
        template = size_configs.get(school_size, {})

        if not template:
            # Hardcoded fallbacks
            fallbacks = {
                'small': {
                    RoomType.CLASSROOM: 12, RoomType.SPECIAL_CLASSROOM: 2,
                    RoomType.MUSIC_ROOM: 1, RoomType.GYMNASIUM: 1,
                    RoomType.LIBRARY: 1, RoomType.OFFICE: 4,
                    RoomType.TEACHER_OFFICE: 2, RoomType.CORRIDOR: 8,
                    RoomType.STAIRCASE: 3, RoomType.TOILET: 6,
                    RoomType.STORAGE: 2, RoomType.CAFETERIA: 1,
                    RoomType.ENTRANCE_HALL: 1,
                },
                'medium': {
                    RoomType.CLASSROOM: 24, RoomType.SPECIAL_CLASSROOM: 3,
                    RoomType.MUSIC_ROOM: 2, RoomType.GYMNASIUM: 1,
                    RoomType.LIBRARY: 1, RoomType.OFFICE: 8,
                    RoomType.TEACHER_OFFICE: 4, RoomType.CORRIDOR: 14,
                    RoomType.STAIRCASE: 4, RoomType.TOILET: 12,
                    RoomType.STORAGE: 4, RoomType.CAFETERIA: 1,
                    RoomType.ENTRANCE_HALL: 1,
                },
                'large': {
                    RoomType.CLASSROOM: 36, RoomType.SPECIAL_CLASSROOM: 5,
                    RoomType.MUSIC_ROOM: 3, RoomType.GYMNASIUM: 2,
                    RoomType.LIBRARY: 2, RoomType.OFFICE: 12,
                    RoomType.TEACHER_OFFICE: 6, RoomType.CORRIDOR: 20,
                    RoomType.STAIRCASE: 6, RoomType.TOILET: 18,
                    RoomType.STORAGE: 6, RoomType.CAFETERIA: 1,
                    RoomType.ENTRANCE_HALL: 1,
                },
            }
            program = fallbacks.get(school_size, fallbacks['medium'])
        else:
            program = {}
            for type_str, count in template.items():
                try:
                    program[RoomType(type_str)] = count
                except ValueError:
                    continue  # Unknown type — skip

        return program

    # ========================================================================
    # Standard Floor Resolution (标准层)
    # ========================================================================

    def _resolve_typical_floors(
        self, num_floors: int
    ) -> List[Tuple[str, int, int]]:
        """
        Determine which typical floors exist based on num_floors.

        Returns:
            List of (typical_floor_type, min_floor, max_floor) tuples.
            Example for 4 floors: [('ground', 0, 0), ('teaching', 1, 3)]
            Example for 6 floors: [('ground', 0, 0), ('teaching', 1, 4), ('top', 5, 5)]
        """
        tf_config = self.rule_params.get('typical_floors', {})
        top_min = (tf_config.get('top') or {}).get('min_floors_for_top', 5)

        floors = []
        # Ground floor always exists
        floors.append(('ground', 0, 0))

        if num_floors >= top_min:
            # Ground + teaching*(N-2) + top
            floors.append(('teaching', 1, num_floors - 2))
            floors.append(('top', num_floors - 1, num_floors - 1))
        elif num_floors >= 2:
            # Ground + teaching*(N-1)
            floors.append(('teaching', 1, num_floors - 1))

        return floors

    def _split_program_to_floors(
        self,
        program: Dict[RoomType, int],
        typical_floors: List[Tuple[str, int, int]],
    ) -> Dict[str, Dict[RoomType, int]]:
        """
        Split total room counts across standard floors.

        Logic:
          - ground-only rooms (cafeteria, entrance_hall, gymnasium): 100% → ground
          - teaching-only rooms (classroom, teacher_office, music_room): split across
            teaching floors proportionally
          - shared rooms (corridor, toilet, staircase, storage): split proportionally

        Returns:
            Dict[tf_type, Dict[RoomType, count]]
        """
        tf_config = self.rule_params.get('typical_floors', {})

        # Build set of which room types go to which typical floor
        tf_rooms: Dict[str, set] = {}
        for tf_type, cfg in tf_config.items():
            room_names = cfg.get('rooms', [])
            tf_rooms[tf_type] = {RoomType(n) for n in room_names if n in [rt.value for rt in RoomType]}

        # Count teaching and top PHYSICAL floors (not typical floor types)
        n_teaching_phys = sum(hi - lo + 1 for tf, lo, hi in typical_floors if tf == 'teaching')
        n_top_phys = sum(hi - lo + 1 for tf, lo, hi in typical_floors if tf == 'top')
        n_teaching_types = sum(1 for tf, _, _ in typical_floors if tf == 'teaching')
        n_top_types = sum(1 for tf, _, _ in typical_floors if tf == 'top')
        total_tf_count = max(1, n_teaching_types + n_top_types)

        result: Dict[str, Dict[RoomType, int]] = {'ground': {}, 'teaching': {}, 'top': {}}

        for rt, total_count in program.items():
            # Determine which floors this room type belongs to
            in_ground = rt in tf_rooms.get('ground', set())
            in_teaching = rt in tf_rooms.get('teaching', set())
            in_top = rt in tf_rooms.get('top', set())

            # Rooms exclusive to ground
            if in_ground and not in_teaching and not in_top:
                result['ground'][rt] = total_count
            # Rooms exclusive to teaching floors
            elif in_teaching and not in_ground:
                if n_top_types > 0 and in_top:
                    # Split between teaching and top
                    teach_phys = n_teaching_phys
                    top_phys = n_top_phys
                    teach_count = int(total_count * teach_phys / (teach_phys + top_phys))
                    top_count = total_count - teach_count
                    result['teaching'][rt] = teach_count
                    result['top'][rt] = top_count
                else:
                    result['teaching'][rt] = total_count
            # Rooms shared across all floors
            else:
                # Split proportionally by physical floor count
                n_physical = sum(hi - lo + 1 for _, lo, hi in typical_floors)
                n_teaching_share = n_teaching_phys
                ground_share = max(1, total_count * 1 // n_physical) if total_count > 0 else 0
                remainder = total_count - ground_share

                result['ground'][rt] = ground_share
                if n_top_types > 0:
                    teach_share = int(remainder * n_teaching_phys / (n_teaching_phys + n_top_phys))
                    result['teaching'][rt] = teach_share
                    result['top'][rt] = remainder - teach_share
                else:
                    result['teaching'][rt] = remainder

            # Ensure every floor has sufficient stairs and corridors per PHYSICAL floor
            if rt == RoomType.STAIRCASE:
                result['teaching'][rt] = max(result['teaching'].get(rt, 0), n_teaching_phys)
                result['ground'][rt] = max(result['ground'].get(rt, 0), 1)
            if rt == RoomType.CORRIDOR:
                # Corridors must cover 10-30% of total area (GB50099-2011 §8.2.3)
                # Estimate: each corridor segment averages 30 sqm.
                # For a building with total_area sqm, we need ~ total_area * 0.12 / 30 corridors.
                # Approximate total_area from program: sum(area_per_type * count)
                total_est_area = sum(
                    (self.catalog.get(t).area_range_sqm[0] + self.catalog.get(t).area_range_sqm[1]) / 2 * c
                    for t, c in program.items() if t != RoomType.CORRIDOR and c > 0
                )
                target_corr_area = total_est_area * 0.13  # target 13% corridor ratio
                avg_corr_area = 30.0
                total_corr_needed = max(8, int(target_corr_area / avg_corr_area))
                # Distribute: 30% ground, 60% teaching, 10% top
                result['ground'][rt] = max(result['ground'].get(rt, 0), max(4, total_corr_needed * 30 // 100))
                result['teaching'][rt] = max(result['teaching'].get(rt, 0), max(4, total_corr_needed * 60 // 100))
                if n_top_types > 0:
                    result['top'][rt] = max(result['top'].get(rt, 0), max(3, total_corr_needed * 10 // 100))

        return result

    # ========================================================================
    # Position assignment (standard floor layout)
    # ========================================================================

    def _assign_positions(
        self,
        rooms: List[RoomNode],
    ):
        """
        Assign (x, y) positions based on typical floor type.

        Standard floor layout:
          - Ground: admin west, special/noisy east, entrance center-west,
                    service/circulation center, corridor spine east-west
          - Teaching: classrooms south, special north, corridor center,
                      staircases at ends
          - Top: similar to teaching, with slight variation

        Each typical floor gets a vertical slice of the site.
        """
        min_x, min_y, max_x, max_y = self._get_site_bounds()
        mid_x = (min_x + max_x) / 2.0

        # Group rooms by typical floor type
        tf_groups: Dict[str, list] = {'ground': [], 'teaching': [], 'top': []}
        for room in rooms:
            tf_groups.get(room.typical_floor, 'ground').append(room)

        # Assign vertical slice: ground=bottom, teaching=middle, top=top
        n_types = sum(1 for g in tf_groups.values() if g)
        slice_height = (max_y - min_y) / max(1, n_types)
        tf_y_offsets = {
            'ground': min_y,
            'teaching': min_y + slice_height,
            'top': min_y + 2 * slice_height,
        }
        tf_centers = {k: v + slice_height / 2 for k, v in tf_y_offsets.items()}

        for tf_type, tf_rooms in tf_groups.items():
            if not tf_rooms:
                continue
            base_y = tf_centers[tf_type]

            for room in tf_rooms:
                zone = ROOM_TO_ZONE.get(room.room_type, ZoneType.MIXED)

                if tf_type == 'ground':
                    if zone == ZoneType.ADMIN:
                        x = self.rng.uniform(min_x + 10, min_x + 50)
                        y = base_y + self.rng.uniform(-10, 10)
                    elif zone == ZoneType.SPECIAL or room.room_type == RoomType.CAFETERIA:
                        x = self.rng.uniform(max_x - 50, max_x - 10)
                        y = base_y + self.rng.uniform(-10, 10)
                    elif room.room_type == RoomType.ENTRANCE_HALL:
                        x = min_x + 10
                        y = base_y
                    else:
                        x = self.rng.uniform(mid_x - 40, mid_x + 40)
                        y = base_y + self.rng.uniform(-5, 5)
                else:
                    # Teaching or Top floor
                    if zone == ZoneType.TEACHING:
                        x = self.rng.uniform(min_x + 10, max_x - 10)
                        y = base_y + self.rng.uniform(slice_height * 0.1, slice_height * 0.4)
                    elif zone == ZoneType.SPECIAL:
                        x = self.rng.uniform(min_x + 10, max_x - 10)
                        y = base_y - self.rng.uniform(slice_height * 0.05, slice_height * 0.35)
                    elif room.room_type == RoomType.CORRIDOR:
                        x = self.rng.uniform(mid_x - 30, mid_x + 30)
                        y = base_y
                    elif room.room_type == RoomType.STAIRCASE:
                        stair_pos = self.rng.choice(['west', 'center', 'east'])
                        if stair_pos == 'west':
                            x = min_x + 10
                        elif stair_pos == 'east':
                            x = max_x - 10
                        else:
                            x = mid_x
                        y = base_y
                    else:
                        x = self.rng.uniform(min_x + 10, max_x - 10)
                        y = base_y + self.rng.uniform(-10, 10)

                room.centroid = (x, y)

    # ========================================================================
    # Main generation pipeline
    # ========================================================================

    def generate(
        self,
        num_floors: int = 3,
        school_size: str = 'medium',
        validate: bool = True,
    ) -> 'GenerationResult':
        """
        Generate a complete school building graph.

        Args:
            num_floors: Number of floors (2-5 recommended).
            school_size: 'small', 'medium', or 'large'.
            validate: If True, retry on constraint failure (up to MAX_RETRIES).

        Returns:
            GenerationResult containing rooms, env_nodes, edges, and metadata.

        Raises:
            RuntimeError: If generation fails after MAX_RETRIES attempts.
        """
        if school_size not in SCHOOL_SIZES:
            raise ValueError(
                f"Invalid school_size '{school_size}'. Must be one of {SCHOOL_SIZES}"
            )

        self.room_factory.reset_counters()

        best_result = None
        best_violations = float('inf')

        for attempt in range(1, self.MAX_RETRIES + 1):
            # Step 1: Resolve typical floors & generate school program
            typical_floors = self._resolve_typical_floors(num_floors)
            program = self._generate_school_program(school_size)
            tf_program = self._split_program_to_floors(program, typical_floors)

            # Step 2: Generate room nodes per typical floor
            rooms: List[RoomNode] = []
            for tf_type, lo_floor, hi_floor in typical_floors:
                tf_room_counts = tf_program.get(tf_type, {})
                for room_type, count in tf_room_counts.items():
                    if count <= 0:
                        continue
                    zone_idx = list(ZoneType).index(ROOM_TO_ZONE[room_type])
                    generated = self.room_factory.generate_batch(
                        room_type, count,
                        floor=lo_floor,
                        zone_id=zone_idx,
                        floor_range=(lo_floor, hi_floor),
                        typical_floor=tf_type,
                    )
                    rooms.extend(generated)

            # Step 2b: Enforce minimum room distribution rules
            enforce_minimums(rooms, num_floors)

            # Step 3: Generate environmental nodes
            env_nodes = self.env_factory.generate_all(school_size)

            # Step 4: Assign positions based on typical floor
            self._assign_positions(rooms)

            # Step 5: Apply topology rules
            edges_by_category = self.rule_engine.apply_all_rules(rooms, env_nodes)

            # Step 6: Validate
            if validate:
                results = self.validator.validate_all(rooms, env_nodes, edges_by_category)
                all_passed = ConstraintValidator.all_passed(results)
                num_violations = sum(
                    len(vs) for _p, vs in results.values()
                )

                if all_passed:
                    # Perfect — return immediately
                    self._last_rooms = rooms
                    self._last_env_nodes = env_nodes
                    self._last_edges = edges_by_category
                    self._last_school_size = school_size
                    self._last_num_floors = num_floors

                    return GenerationResult(
                        rooms=rooms,
                        env_nodes=env_nodes,
                        edges_by_category=edges_by_category,
                        school_size=school_size,
                        num_floors=num_floors,
                        seed=self.seed,
                        attempt=attempt,
                        validation_results=results,
                    )
                elif num_violations < best_violations:
                    best_violations = num_violations
                    best_result = GenerationResult(
                        rooms=rooms,
                        env_nodes=env_nodes,
                        edges_by_category=edges_by_category,
                        school_size=school_size,
                        num_floors=num_floors,
                        seed=self.seed,
                        attempt=attempt,
                        validation_results=results,
                    )

            else:
                # No validation — return immediately
                results = self.validator.validate_all(rooms, env_nodes, edges_by_category)
                self._last_rooms = rooms
                self._last_env_nodes = env_nodes
                self._last_edges = edges_by_category

                return GenerationResult(
                    rooms=rooms,
                    env_nodes=env_nodes,
                    edges_by_category=edges_by_category,
                    school_size=school_size,
                    num_floors=num_floors,
                    seed=self.seed,
                    attempt=1,
                    validation_results=results,
                )

            # Reseed for next attempt
            self.rng = np.random.default_rng(self.seed + attempt * 1000)
            self.room_factory._rng = self.rng
            self.env_factory._rng = self.rng

        # All retries exhausted — return best candidate with warnings
        if best_result is not None:
            self._last_rooms = best_result.rooms
            self._last_env_nodes = best_result.env_nodes
            self._last_edges = best_result.edges_by_category
            return best_result

        raise RuntimeError(
            f"Failed to generate valid school graph after {self.MAX_RETRIES} attempts."
        )

    # ========================================================================
    # HeteroData conversion
    # ========================================================================

    def to_hetero_data(self, result: 'GenerationResult') -> "HeteroData":
        """
        Convert a GenerationResult to a PyG HeteroData object.

        Args:
            result: GenerationResult from generate().

        Returns:
            PyG HeteroData with all node and edge types populated.

        Raises:
            ImportError: If PyG is not installed.
        """
        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric is required for HeteroData conversion. "
                "Install with: pip install torch-geometric"
            )

        return self.feature_engineer.build_hetero_data(
            result.rooms,
            result.env_nodes,
            result.edges_by_category,
        )

    def save(
        self,
        result: 'GenerationResult',
        output_dir: str = 'outputs',
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate HeteroData and save to disk.

        Args:
            result: GenerationResult from generate().
            output_dir: Directory to save to.
            filename: Optional filename. Auto-generated if None.

        Returns:
            Path to the saved file.

        Raises:
            ImportError: If PyG or serialization unavailable.
        """
        from utils.serialization import save_hetero_data

        hetero_data = self.to_hetero_data(result)

        if filename is None:
            filename = (
                f"school_{result.school_size}_"
                f"f{result.num_floors}_"
                f"n{len(result.rooms)}_"
                f"seed{result.seed}.pt"
            )

        filepath = os.path.join(output_dir, filename)
        metadata = {
            'school_size': result.school_size,
            'num_floors': result.num_floors,
            'num_rooms': len(result.rooms),
            'num_env_nodes': len(result.env_nodes),
            'seed': result.seed,
            'attempt': result.attempt,
            'all_passed': ConstraintValidator.all_passed(
                result.validation_results
            ) if result.validation_results else None,
        }

        return save_hetero_data(hetero_data, filepath, metadata)


# ============================================================================
# Generation Result
# ============================================================================

class GenerationResult:
    """
    Container for a single generation run's output.

    Attributes:
        rooms: List of generated RoomNode instances.
        env_nodes: List of generated EnvironmentalNode instances.
        edges_by_category: Dict[EdgeCategory, List[Edge]].
        school_size: Size label ('small'/'medium'/'large').
        num_floors: Number of floors.
        seed: Random seed used.
        attempt: Which retry attempt produced this result.
        validation_results: Output of ConstraintValidator.validate_all().
        hetero_data: Optional HeteroData (populated via to_hetero_data()).
    """
    def __init__(
        self,
        rooms: List[RoomNode],
        env_nodes: List[EnvironmentalNode],
        edges_by_category: Dict[EdgeCategory, list],
        school_size: str,
        num_floors: int,
        seed: int,
        attempt: int,
        validation_results: Dict = None,
    ):
        self.rooms = rooms
        self.env_nodes = env_nodes
        self.edges_by_category = edges_by_category
        self.school_size = school_size
        self.num_floors = num_floors
        self.seed = seed
        self.attempt = attempt
        self.validation_results = validation_results or {}
        self._hetero_data = None

    @property
    def hetero_data(self) -> "HeteroData":
        """Lazy HeteroData conversion."""
        if self._hetero_data is None:
            engineer = FeatureEngineer()
            self._hetero_data = engineer.build_hetero_data(
                self.rooms, self.env_nodes, self.edges_by_category
            )
        return self._hetero_data

    def summary(self) -> str:
        """Human-readable generation summary."""
        lines = [
            "=" * 60,
            f"  School Building Graph — {self.school_size.upper()} ({self.num_floors} floors)",
            f"  Seed: {self.seed} | Attempt: {self.attempt}",
            "-" * 60,
            f"  Rooms:     {len(self.rooms)}",
            f"  Env Nodes: {len(self.env_nodes)}",
        ]

        for ec in EdgeCategory:
            count = len(self.edges_by_category.get(ec, []))
            lines.append(f"  {ec.value:25s}: {count:4d}")

        lines.append("-" * 60)

        if self.validation_results:
            if ConstraintValidator.all_passed(self.validation_results):
                lines.append("  ✓ All constraints PASSED")
            else:
                lines.append("  ✗ Some constraints FAILED:")
                for name, (passed, violations) in self.validation_results.items():
                    status = "✓" if passed else "✗"
                    lines.append(f"    [{status}] {name}")
                    for v in violations[:3]:  # Show first 3
                        lines.append(f"      → {v}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(school={self.school_size}, "
            f"floors={self.num_floors}, rooms={len(self.rooms)}, "
            f"seed={self.seed})"
        )
