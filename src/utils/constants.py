"""
Project-wide constants and reference tables.

All constants are derived from GB50099-2011 (中小学设计规范) and GB50016-2014 (建筑设计防火规范).
Values defined here are the DEFAULT fallbacks — the YAML config files take precedence.
"""

from utils.enums import RoomType, EnvNodeType, EdgeCategory, ZoneType, ROOM_TO_ZONE

# --- Enum cardinalities (for one-hot encoding dimensions) ---
NUM_ROOM_TYPES: int = len(RoomType)
NUM_ENV_TYPES: int = len(EnvNodeType)
NUM_ZONE_TYPES: int = len(ZoneType)

# --- Feature dimension constants ---
# room features: one-hot(13) + area + aspect_ratio + occupancy + daylight + noise_lvl + noise_tol + floor + one-hot(6) + fire_exits
ROOM_FEAT_DIM: int = NUM_ROOM_TYPES + 7 + NUM_ZONE_TYPES + 1
#   13 (one-hot RoomType)
#  + 1 (area, normalized)
#  + 1 (aspect_ratio)
#  + 1 (occupancy, normalized)
#  + 1 (daylight_level, ordinal normalized)
#  + 1 (noise_level, ordinal normalized)
#  + 1 (noise_tolerance, ordinal normalized)
#  + 1 (floor, normalized)
#  = 7 scalar features between one-hot blocks
#  + 6 (one-hot ZoneType)
#  + 1 (fire_exits_min, normalized)
#  = 27 total

# env features: one-hot(4) + pos_x + pos_y
ENV_FEAT_DIM: int = NUM_ENV_TYPES + 2
#   4 (one-hot EnvNodeType)
#  + 1 (position_x, normalized)
#  + 1 (position_y, normalized)
#  = 6

# --- Edge attribute dimensions ---
PHYSICAL_EDGE_ATTR_DIM: int = 2      # [distance_weight, is_stair_connection]
ACOUSTIC_EDGE_ATTR_DIM: int = 1      # [attenuation_db]
SIGHT_ROOM_EDGE_ATTR_DIM: int = 2    # [transparency, sight_distance]
SIGHT_ENV_EDGE_ATTR_DIM: int = 2     # [orientation_preference, distance]
PHYS_ENV_EDGE_ATTR_DIM: int = 1      # [access_type]

# --- Default normalization bounds (for MinMax scaling to [0,1]) ---
DEFAULT_MAX_AREA: float = 800.0       # Max room area (gymnasium ~800 m²)
DEFAULT_MAX_OCCUPANCY: float = 300.0  # Max estimated occupancy
DEFAULT_MAX_FIRE_EXITS: int = 4       # Max required fire exits
DEFAULT_MAX_FLOORS: int = 4           # Max floor count (0-indexed)
DEFAULT_SITE_MAX_X: float = 200.0     # Site boundary X (m)
DEFAULT_SITE_MAX_Y: float = 150.0     # Site boundary Y (m)
DEFAULT_MAX_ATTENUATION: float = 60.0 # Max sound attenuation (dB)

# --- GB50099-2011 reference values ---
# Classroom occupancy: 1.0 - 1.5 m²/person (§5.2.1)
# Corridor width: >= 2.4m for teaching buildings (§8.2.1)
# Fire exit spacing: <= 30m between staircases (§8.6.2)
# Max travel distance to exit: 35m for classrooms (§8.6.3)

# --- School program size aliases ---
SCHOOL_SIZES: tuple[str, ...] = ("small", "medium", "large")

# --- Default random seed ---
DEFAULT_SEED: int = 42
