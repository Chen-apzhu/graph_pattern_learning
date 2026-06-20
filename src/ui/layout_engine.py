"""
Orthogonal Layout Engine — 教学楼紧凑平面布局

Single teaching building with:
  - South-facing classrooms (≥54 m²)
  - Central corridor spine (width adapts to room count)
  - Staircases at both ends
  - Toilets at ends + middle for coverage
  - Compact: no wasted space, corridor only as long as needed

Reference: GB50099-2011
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


ROOM_COLORS = {
    'classroom': '#4C72B0', 'special_classroom': '#55A868',
    'music_room': '#C44E52', 'library': '#937860',
    'office': '#8CA5C8', 'teacher_office': '#7FB8D0',
    'corridor': '#CCCCCC', 'staircase': '#8172B2',
    'toilet': '#A0A0A0', 'storage': '#D0D0D0',
    'entrance_hall': '#64B5CD',
}

# Room types that belong in a teaching building
TEACHING_TYPES = {'classroom', 'special_classroom', 'teacher_office',
                  'music_room', 'library', 'office',
                  'corridor', 'staircase', 'toilet', 'storage'}

# Room types that need toilet access
NEEDS_TOILET = {'classroom', 'special_classroom', 'office',
                'teacher_office', 'library', 'music_room'}


@dataclass
class OrthoRoom:
    room_id: str; room_type: str
    x: float; y: float; width: float; height: float
    floor: int = 0; area: float = 0.0
    color: str = '#CCCCCC'

    def to_dict(self) -> dict:
        return {
            'room_id': self.room_id, 'room_type': self.room_type,
            'x': round(self.x, 1), 'y': round(self.y, 1),
            'width': round(self.width, 1), 'height': round(self.height, 1),
            'floor': self.floor, 'area': round(self.area, 1),
        }


class OrthogonalLayoutEngine:
    """Compact single teaching building layout."""

    MARGIN = 3.0
    CORRIDOR_H = 3.0          # ≥2.4m per GB50099 §8.2
    BUILDING_DEPTH = 20.0     # classroom 9m + corridor 3m + north rooms 8m
    MIN_ROOM_W = 7.0          # Minimum classroom width (~60m² / 9m depth)
    STAIR_W = 4.0
    TOILET_W = 4.0
    MAX_ROW_WIDTH = 80.0      # Max building width before splitting into wings

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ══════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════

    def layout(
        self, rooms: list, edges: Dict = None, num_floors: int = 3
    ) -> 'OrthoLayout':
        if not rooms:
            return OrthoLayout(rooms=[], boundary=(60, 40), num_floors=0)

        # Filter to teaching building rooms only
        teaching_rooms = [r for r in rooms if self._rt(r) in TEACHING_TYPES]

        # Group by typical floor, split teaching by physical sub-floors
        tf_groups = self._group(teaching_rooms)

        all_rooms: List[OrthoRoom] = []
        floor_results = []

        for tf_type, tf_rooms in tf_groups.items():
            if not tf_rooms:
                continue

            # Determine physical floor count for this typical floor
            phys_count = 1
            for r in tf_rooms:
                fr = getattr(r, 'floor_range', (0, 0))
                phys_count = max(phys_count, fr[1] - fr[0] + 1)

            if phys_count > 1 and len(tf_rooms) > 15:
                # Split rooms evenly across physical sub-floors
                sub_floors = self._split_across_physical(tf_rooms, phys_count)
                for sub_idx, sub_rooms in enumerate(sub_floors):
                    placed, w = self._layout_floor(sub_rooms, f'{tf_type}_p{sub_idx}',
                                                   len(floor_results))
                    floor_results.append((len(floor_results), placed, w))
                    all_rooms.extend(placed)
            else:
                placed, w = self._layout_floor(tf_rooms, tf_type, len(floor_results))
                floor_results.append((len(floor_results), placed, w))
                all_rooms.extend(placed)

        # Stack floors vertically
        final_rooms = []
        floor_h = self.BUILDING_DEPTH + self.MARGIN
        max_width = self.MARGIN * 2
        for i, placed, w_used in floor_results:
            y_offset = i * floor_h + self.MARGIN
            for r in placed:
                r.y += y_offset
            final_rooms.extend(placed)
            max_width = max(max_width, w_used + 2 * self.MARGIN)

        total_h = len(floor_results) * floor_h + self.MARGIN
        total_w = max_width

        return OrthoLayout(
            rooms=final_rooms,
            boundary=(total_w, total_h),
            num_floors=len(floor_results),
        )

    def _group(self, rooms: list) -> Dict[str, list]:
        groups: Dict[str, list] = defaultdict(list)
        for room in rooms:
            tf = getattr(room, 'typical_floor', 'ground')
            groups[tf].append(room)
        return {k: groups[k] for k in ['ground', 'teaching', 'top'] if k in groups}

    # ══════════════════════════════════════════════════════════════
    # Single floor layout — compact, no wasted space
    # ══════════════════════════════════════════════════════════════

    def _layout_floor(
        self, rooms: list, tf_type: str, floor_idx: int,
    ) -> Tuple[List[OrthoRoom], float]:
        """
        Layout one typical floor compactly.
        If rooms exceed MAX_ROW_WIDTH, split into parallel wings
        connected by the same corridor.
        """
        corridors = self._pick(rooms, 'corridor')
        stairs = self._pick(rooms, 'staircase')
        toilets = self._pick(rooms, 'toilet')
        storage = self._pick(rooms, 'storage')

        south = [r for r in rooms if self._rt(r) in NEEDS_TOILET
                 and self._rt(r) not in ('staircase', 'toilet', 'storage', 'corridor')]
        north = [r for r in rooms if self._rt(r) not in NEEDS_TOILET
                 and self._rt(r) not in ('staircase', 'toilet', 'storage', 'corridor')]
        used_ids = {id(r) for r in corridors + stairs + toilets + storage + south + north}
        south += [r for r in rooms if id(r) not in used_ids]

        # Estimate width needed for south and north rows
        south_est = self._estimate_row_width(south, stairs[:2])
        north_est = self._estimate_row_width(toilets + storage, stairs[2:])
        needed_w = max(south_est, north_est)

        # If too wide, split south rooms into two halves → two parallel rows
        if needed_w > self.MAX_ROW_WIDTH and len(south) >= 6:
            mid = len(south) // 2
            south_a = south[:mid]
            south_b = south[mid:]
            # Each half gets its own row width
            w_a = self._estimate_row_width(south_a, stairs[:1])
            w_b = self._estimate_row_width(south_b, stairs[1:2])
            bld_w = max(w_a + w_b + self.STAIR_W, north_est)
            bld_w = min(bld_w, self.MAX_ROW_WIDTH * 1.5)
        else:
            bld_w = min(needed_w, self.MAX_ROW_WIDTH * 1.2)

        bld_w = max(30.0, bld_w)

        result = []
        bx = self.MARGIN
        by = 0.0
        bh = self.BUILDING_DEPTH

        # ── Corridor spine (only as wide as needed) ──
        corr_y = by + (bh - self.CORRIDOR_H) / 2
        result.append(OrthoRoom(
            room_id=f'corridor_{tf_type}', room_type='corridor',
            x=bx + 1, y=corr_y, width=bld_w - 2, height=self.CORRIDOR_H,
            floor=floor_idx, area=bld_w * self.CORRIDOR_H,
            color=ROOM_COLORS['corridor'],
        ))

        # ── South row: classrooms ──
        south_y = corr_y + self.CORRIDOR_H + 0.5
        south_h = (by + bh) - south_y - 0.5
        south_row = stairs[:1] + south + stairs[1:2]
        result.extend(self._pack_row(south_row, bx, south_y, bld_w, south_h, floor_idx))

        # ── North row: toilets evenly spaced for optimal coverage ──
        north_y = by + 0.5
        north_h = corr_y - north_y - 0.5
        optimal_n = max(2, int(bld_w / 35) + 1)
        usable_toilets = toilets[:min(optimal_n, len(toilets))]
        result.extend(self._place_toilets_evenly(
            usable_toilets, storage, stairs[2:],
            bx, north_y, bld_w, north_h, floor_idx))

        return (result, bld_w)

    def _split_across_physical(self, rooms: list, phys_count: int) -> List[list]:
        """Split teaching floor rooms evenly across physical sub-floors."""
        if phys_count <= 1:
            return [rooms]

        # Separate essential per-floor rooms
        classrooms = self._pick(rooms, 'classroom')
        specials = self._pick(rooms, 'special_classroom')
        teachers = self._pick(rooms, 'teacher_office')
        music = self._pick(rooms, 'music_room')
        library = self._pick(rooms, 'library')
        offices = self._pick(rooms, 'office')
        corridors = self._pick(rooms, 'corridor')
        stairs = self._pick(rooms, 'staircase')
        toilets = self._pick(rooms, 'toilet')
        storage = self._pick(rooms, 'storage')

        # Flex rooms (classrooms, specials, teachers) → split evenly
        flex = classrooms + specials + teachers + music
        other_flex = library + offices

        sub_floors = [[] for _ in range(phys_count)]

        # Distribute flex rooms
        for i, room in enumerate(flex):
            sub_floors[i % phys_count].append(room)

        # Distribute other rooms
        for i, room in enumerate(other_flex):
            sub_floors[i % phys_count].append(room)

        # Each sub-floor needs at least 1 corridor, 1 staircase, 1-2 toilets
        for i in range(phys_count):
            if i < len(corridors):
                sub_floors[i].append(corridors[i])
            if i < len(stairs):
                sub_floors[i].append(stairs[i])
            # Distribute toilets evenly
            t_start = i * len(toilets) // phys_count
            t_end = (i + 1) * len(toilets) // phys_count
            sub_floors[i].extend(toilets[t_start:t_end])
            # Distribute storage
            s_start = i * len(storage) // phys_count
            s_end = (i + 1) * len(storage) // phys_count
            sub_floors[i].extend(storage[s_start:s_end])

        # Ensure no empty sub-floor
        return [sf for sf in sub_floors if sf]

    def _estimate_row_width(self, main_rooms: list, end_rooms: list) -> float:
        """Estimate total width needed for a row."""
        w = 0
        for r in end_rooms:
            rt = self._rt(r)
            w += (self.STAIR_W if rt == 'staircase' else self.TOILET_W) + 0.3
        for r in main_rooms:
            area = getattr(r, 'area', 60)
            w += max(self.MIN_ROOM_W, area / 9.0) + 0.3  # area/depth ≈ width
        return w

    def _place_toilets_evenly(
        self, toilets: list, storage: list, extra_stairs: list,
        x0: float, y0: float, total_w: float, h: float, floor_idx: int,
    ) -> List[OrthoRoom]:
        """
        Place toilets at evenly-spaced positions along the row for optimal coverage.
        Then fill gaps with storage and stairs.
        """
        n = len(toilets)
        if n == 0:
            return self._pack_row(storage + extra_stairs, x0, y0, total_w, h, floor_idx)

        result = []
        gap = 0.2

        if n == 1:
            # Center
            tx = x0 + total_w / 2 - self.TOILET_W / 2
            result.append(self._make_ortho(toilets[0], tx, y0, self.TOILET_W, h, floor_idx))
            # Fill left and right
            result.extend(self._pack_row(storage[:len(storage)//2] + extra_stairs[:len(extra_stairs)//2],
                                        x0, y0, tx - x0 - gap, h, floor_idx))
            result.extend(self._pack_row(storage[len(storage)//2:] + extra_stairs[len(extra_stairs)//2:],
                                        tx + self.TOILET_W + gap, y0,
                                        total_w - (tx + self.TOILET_W - x0) - gap, h, floor_idx))

        elif n == 2:
            # Left + Right ends
            result.append(self._make_ortho(toilets[0], x0, y0, self.TOILET_W, h, floor_idx))
            result.append(self._make_ortho(toilets[1], x0 + total_w - self.TOILET_W, y0,
                                          self.TOILET_W, h, floor_idx))
            mid_w = total_w - 2 * (self.TOILET_W + gap)
            result.extend(self._pack_row(storage + extra_stairs,
                                        x0 + self.TOILET_W + gap, y0, mid_w, h, floor_idx))

        else:
            # Evenly spaced: divide total_w into n segments, toilet at center of each
            seg_w = total_w / n
            other_items = storage + extra_stairs
            other_idx = 0
            for i in range(n):
                tx = x0 + i * seg_w + seg_w / 2 - self.TOILET_W / 2
                result.append(self._make_ortho(toilets[i], tx, y0, self.TOILET_W, h, floor_idx))
                # Fill gap after this toilet (except last)
                if i < n - 1:
                    next_tx = x0 + (i + 1) * seg_w + seg_w / 2 - self.TOILET_W / 2
                    gap_w = next_tx - (tx + self.TOILET_W)
                    gap_items = []
                    while other_idx < len(other_items) and len(gap_items) < 3:
                        gap_items.append(other_items[other_idx])
                        other_idx += 1
                    if gap_items:
                        result.extend(self._pack_row(gap_items, tx + self.TOILET_W + gap,
                                                    y0, gap_w - gap, h, floor_idx))
            # Remaining items after last toilet
            if other_idx < len(other_items) and n > 0:
                last_tx = x0 + (n-1) * seg_w + seg_w / 2 + self.TOILET_W / 2
                result.extend(self._pack_row(other_items[other_idx:],
                                            last_tx + gap, y0,
                                            total_w - (last_tx - x0) - gap, h, floor_idx))

        return result

    def _pack_row(
        self, rooms: list,
        x0: float, y0: float, total_w: float, h: float, floor_idx: int,
    ) -> List[OrthoRoom]:
        """Pack rooms into a row with minimum widths, filling available space."""
        if not rooms or h <= 0:
            return []

        result = []
        gap = 0.3

        # Fixed-width rooms (stairs, toilets)
        fixed_w = 0
        flex_rooms = []
        for room in rooms:
            rt = self._rt(room)
            if rt == 'staircase':
                fixed_w += self.STAIR_W + gap
            elif rt == 'toilet':
                fixed_w += self.TOILET_W + gap
            else:
                flex_rooms.append(room)

        available = total_w - fixed_w - gap
        if available < self.MIN_ROOM_W * max(1, len(flex_rooms)):
            available = self.MIN_ROOM_W * max(1, len(flex_rooms))

        # Proportionally distribute remaining width, preserving input order
        total_area = sum(getattr(r, 'area', 50) for r in flex_rooms)
        if total_area <= 0:
            total_area = len(flex_rooms)

        flex_index = 0
        cx = x0
        for room in rooms:
            rt = self._rt(room)
            if rt == 'staircase':
                w = self.STAIR_W
            elif rt == 'toilet':
                w = self.TOILET_W
            else:
                area = getattr(room, 'area', 50)
                w = max(self.MIN_ROOM_W, (area / total_area) * available) if total_area > 0 else 5.0

            w = min(w, (x0 + total_w) - cx - gap)
            if w < 2.0:
                continue

            result.append(OrthoRoom(
                room_id=getattr(room, 'room_id', self._rt(room)),
                room_type=self._rt(room),
                x=cx, y=y0, width=w - gap, height=h,
                floor=floor_idx,
                area=getattr(room, 'area', w * h),
                color=ROOM_COLORS.get(self._rt(room), '#CCCCCC'),
            ))
            cx += w

        return result

    def _compute_width(self, rooms: list) -> float:
        """Estimate the minimum width needed for a list of rooms."""
        if not rooms:
            return self.MIN_ROOM_W * 2
        w = 0
        for room in rooms:
            rt = self._rt(room)
            if rt == 'staircase':
                w += self.STAIR_W + 0.3
            elif rt == 'toilet':
                w += self.TOILET_W + 0.3
            else:
                area = getattr(room, 'area', 50)
                w += max(self.MIN_ROOM_W, math.sqrt(area) * 0.9) + 0.3
        return max(self.MIN_ROOM_W * 2, w)

    def _make_ortho(self, room, x, y, w, h, floor_idx) -> OrthoRoom:
        return OrthoRoom(
            room_id=getattr(room, 'room_id', self._rt(room)),
            room_type=self._rt(room),
            x=x, y=y, width=w, height=h,
            floor=floor_idx,
            area=getattr(room, 'area', w * h),
            color=ROOM_COLORS.get(self._rt(room), '#CCCCCC'),
        )

    @staticmethod
    def _rt(room) -> str:
        rt = getattr(room, 'room_type', None)
        if rt is None:
            return 'unknown'
        return rt.value if hasattr(rt, 'value') else str(rt)

    @staticmethod
    def _pick(rooms: list, type_name: str) -> list:
        return [r for r in rooms if OrthogonalLayoutEngine._rt(r) == type_name]


@dataclass
class OrthoLayout:
    rooms: List[OrthoRoom]
    boundary: Tuple[float, float]
    num_floors: int

    @property
    def width(self) -> float:
        return self.boundary[0]

    @property
    def height(self) -> float:
        return self.boundary[1]

    def to_dict(self) -> dict:
        return {
            'rooms': [r.to_dict() for r in self.rooms],
            'boundary': list(self.boundary),
            'num_floors': self.num_floors,
        }
