"""
Project-wide enumerations for the multi-modal heterogeneous attributed graph.

All enum values correspond to the data structures defined in task.md §3 (Nodes & Edges).
Each enum is independently importable and used across data/, graph/, and tests/.
"""

from enum import Enum, IntEnum


class RoomType(str, Enum):
    """
    Spatial node type classification (§3.1 节点定义).

    Each type carries semantic meaning for the building layout:
    - Teaching types have high daylight requirements
    - Noise-generating types need acoustic separation
    - Service types have low daylight/quiet requirements
    """
    CLASSROOM = "classroom"                # 普通教室 — primary teaching unit
    SPECIAL_CLASSROOM = "special_classroom" # 专用教室 — lab, art, computer room
    MUSIC_ROOM = "music_room"              # 音乐教室 — strong acoustic isolation needed
    GYMNASIUM = "gymnasium"                # 体育馆 — large span, very loud
    LIBRARY = "library"                    # 图书馆 — quiet zone, high daylight
    OFFICE = "office"                      # 行政办公室 — moderate requirements
    TEACHER_OFFICE = "teacher_office"      # 教师办公室 — quiet workspace
    CORRIDOR = "corridor"                  # 走道 — circulation spine
    STAIRCASE = "staircase"                # 楼梯间 — vertical circulation + fire egress
    TOILET = "toilet"                      # 卫生间 — sanitary
    STORAGE = "storage"                    # 储藏室 — minimal requirements
    CAFETERIA = "cafeteria"                # 食堂 — noisy, ground floor
    ENTRANCE_HALL = "entrance_hall"        # 门厅 — main lobby, connects to exterior


class EnvNodeType(str, Enum):
    """Virtual global environment nodes (§3.1 — 环境节点)."""
    SOUTH_FACING = "south_facing"          # 正南向节点 — daylight source
    MAIN_ROAD_ACCESS = "main_road_access"  # 主干道接驳点 — site access
    PLAYGROUND = "playground"              # 操场 — outdoor activity
    GREEN_SPACE = "green_space"            # 绿化 — view quality


class EdgeCategory(str, Enum):
    """
    Edge type classification (§3.2 边定义).

    Each edge type represents a distinct modality of spatial relationship.
    """
    PHYSICAL_CONNECTS = "physical_connects"    # 物理连通边 — doors/passages
    ACOUSTIC_BLOCKS = "acoustic_blocks"        # 声学阻断边 — sound-isolating walls
    SIGHT_LINES = "sight_lines"                # 视线/采光边 — visual/light connections


class DaylightLevel(IntEnum):
    """
    Daylight requirement intensity (§3.1 — 采光代理).

    Ordinal scale: higher = stronger daylight mandate.
    Used to determine sight_line edge creation and daylight constraint checking.
    """
    NONE = 0       # No daylight needed (storage, toilets)
    LOW = 1        # Some daylight acceptable (corridors)
    MEDIUM = 2     # Daylight desirable (offices, cafeteria)
    HIGH = 3       # Daylight mandatory (classrooms, library, art rooms)
    CRITICAL = 4   # South-facing required (art studios, special labs)


class NoiseLevel(IntEnum):
    """
    Noise emission level of a room (§3.2 — 声学阻断边).

    Ordinal scale: higher = louder.
    Used for acoustic separation rule calculation.
    """
    QUIET = 0      # Library, study room
    MODERATE = 1   # Classroom, office
    NOISY = 2      # Corridor, cafeteria, entrance hall
    LOUD = 3       # Music room
    VERY_LOUD = 4  # Gymnasium


class ZoneType(str, Enum):
    """Functional zone classification for spatial grouping."""
    TEACHING = "teaching"         # 教学区: classrooms, special classrooms, library, teacher offices
    SPECIAL = "special"           # 特殊区: music rooms, gymnasium (noise generators)
    ADMIN = "admin"               # 行政办公区: offices, entrance hall
    SERVICE = "service"           # 服务区: cafeteria, toilets, storage, staircases
    CIRCULATION = "circulation"   # 交通区: corridors
    MIXED = "mixed"               # Mixed/undefined zone


class FloorType(str, Enum):
    """Standard floor (标准层) classification."""
    GROUND = "ground"       # 首层: entrance, admin, cafeteria, gymnasium
    TEACHING = "teaching"   # 教学标准层: classrooms, corridors, toilets, stairs (repeated N times)
    TOP = "top"             # 顶层 (optional): similar to teaching, ≥5 floors only


# Mapping from RoomType to its functional ZoneType
ROOM_TO_ZONE: dict[RoomType, ZoneType] = {
    RoomType.CLASSROOM: ZoneType.TEACHING,
    RoomType.SPECIAL_CLASSROOM: ZoneType.TEACHING,
    RoomType.LIBRARY: ZoneType.TEACHING,
    RoomType.TEACHER_OFFICE: ZoneType.TEACHING,
    RoomType.MUSIC_ROOM: ZoneType.SPECIAL,
    RoomType.GYMNASIUM: ZoneType.SPECIAL,
    RoomType.OFFICE: ZoneType.ADMIN,
    RoomType.ENTRANCE_HALL: ZoneType.ADMIN,
    RoomType.CAFETERIA: ZoneType.SERVICE,
    RoomType.TOILET: ZoneType.SERVICE,
    RoomType.STORAGE: ZoneType.SERVICE,
    RoomType.STAIRCASE: ZoneType.SERVICE,
    RoomType.CORRIDOR: ZoneType.CIRCULATION,
}
