"""
Pattern-Driven Interactive Layout Engine — 模式驱动交互式布局引擎

Lets users select architectural patterns (motifs) from the learned dictionary,
place them on a canvas, auto-complete with corridors/stairs/toilets, and
get real-time GNN quality scores.

Workflow:
  1. Load motifs from outputs/explainer/motif_dictionary.json
  2. User selects + places patterns on canvas
  3. Auto-complete fills in missing essentials
  4. Canvas → HeteroData → GNN scorer → quality score
  5. Constraint validator checks compliance
"""

from __future__ import annotations

import json, math, io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from data.generator import SchoolBuildingGenerator
from data.constraints import ConstraintValidator
from utils.enums import RoomType, EdgeCategory


ROOM_COLORS = {
    'classroom': '#4C72B0', 'special_classroom': '#55A868',
    'music_room': '#C44E52', 'library': '#937860',
    'office': '#8CA5C8', 'teacher_office': '#7FB8D0',
    'corridor': '#CCCCCC', 'staircase': '#8172B2',
    'toilet': '#A0A0A0', 'storage': '#D0D0D0',
    'entrance_hall': '#64B5CD', 'gymnasium': '#DD8452',
    'cafeteria': '#E8B44F',
}

ROOM_CN = {
    'classroom': '教室', 'special_classroom': '专用教室',
    'music_room': '音乐教室', 'library': '图书馆', 'office': '办公室',
    'teacher_office': '教师办公', 'corridor': '走道', 'staircase': '楼梯间',
    'toilet': '卫生间', 'storage': '储藏室', 'entrance_hall': '门厅',
}

# Default room sizes (m²)
DEFAULT_SIZES = {
    'classroom': (8, 8), 'special_classroom': (8, 10),
    'music_room': (9, 9), 'library': (10, 10), 'office': (4, 6),
    'teacher_office': (5, 8), 'corridor': (3, 30), 'staircase': (4, 8),
    'toilet': (4, 5), 'storage': (3, 4), 'entrance_hall': (6, 8),
}


@dataclass
class PlacedRoom:
    """A room placed on the interactive canvas."""
    room_type: str      # e.g., 'classroom'
    x: float; y: float
    width: float; height: float
    label: str = ''     # e.g., '教室_01'
    color: str = '#CCC'
    is_corridor: bool = False

    def to_dict(self) -> dict:
        return {
            'room_type': self.room_type, 'label': self.label,
            'x': round(self.x, 1), 'y': round(self.y, 1),
            'w': round(self.width, 1), 'h': round(self.height, 1),
        }


@dataclass
class Pattern:
    """A learned architectural pattern from the motif dictionary."""
    motif_id: str
    name: str
    room_counts: Dict[str, float]   # room_type → avg count
    edge_counts: Dict[str, float]   # edge_type → avg count
    frequency: int
    percentage: float
    centroid_graph: Optional[dict] = None
    layout: List[Dict] = field(default_factory=list)  # pre-computed layout template

    def summary(self) -> str:
        items = ', '.join(
            f'{ROOM_CN.get(k,k)}×{v:.0f}'
            for k, v in sorted(self.room_counts.items(), key=lambda x: -x[1])
            if v >= 0.5
        )
        return f"[{self.motif_id}] {self.name}: {items}"


class PatternEngine:
    """
    Interactive pattern-driven layout engine.

    Usage:
        engine = PatternEngine()
        engine.load_motifs('outputs/explainer/motif_dictionary.json')
        engine.place_pattern(0, x=10, y=10)
        engine.auto_complete()
        score = engine.score_with_gnn(model)
        engine.render_canvas() → matplotlib Figure
    """

    CANVAS_W = 120.0
    CANVAS_H = 80.0
    CORRIDOR_H = 3.0

    def __init__(self):
        self.patterns: List[Pattern] = []
        self.placed_rooms: List[PlacedRoom] = []
        self.selected_pattern_idx: int = 0
        self.placement_x: float = 5.0
        self.placement_y: float = 5.0
        self.gnn_score: Optional[float] = None
        self.validation: dict = {}
        self.next_label_counters: Dict[str, int] = defaultdict(int)

    # ══════════════════════════════════════════════════════════════
    # Motif loading
    # ══════════════════════════════════════════════════════════════

    def load_motifs(self, json_path: str = 'outputs/explainer/motif_dictionary.json'):
        """Load learned motifs and extract sub-patterns for interactive placement."""
        path = Path(json_path)
        if not path.exists():
            self._make_default_patterns()
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.patterns = []
        for m in data.get('motifs', []):
            room_comp = m.get('room_composition', {})
            # Extract sub-patterns from the large motif
            sub_patterns = self._extract_sub_patterns(room_comp, m)
            self.patterns.extend(sub_patterns)

        # Also add default patterns as fallback
        if not self.patterns:
            self._make_default_patterns()

    def _extract_sub_patterns(self, room_comp: Dict, motif_meta: dict) -> List[Pattern]:
        """
        Extract smaller, placeable sub-patterns from a large motif.
        A motif with 'classroom: 16, corridor: 14' becomes patterns like:
          - '教学单元 4教室+走道' (classroom×4, corridor×1)
          - '交通核 楼梯+卫生间' (staircase×1, toilet×1)
        """
        results = []
        valid = set(ROOM_COLORS.keys())
        comp = {k: v for k, v in room_comp.items() if k in valid and v >= 0.5}

        # ── Teaching unit: classroom cluster ──
        n_class = int(comp.get('classroom', 0))
        if n_class >= 4:
            unit_size = min(6, n_class)
            results.append(Pattern(
                motif_id=f'{motif_meta.get("motif_id","?")}_U1',
                name=f'教学单元 ({unit_size}教室+走道)',
                room_counts={'classroom': unit_size, 'corridor': 1, 'toilet': 1},
                edge_counts={},
                frequency=motif_meta.get('frequency', 0),
                percentage=motif_meta.get('percentage', 0),
                layout=self._make_grid(unit_size, 1, 'classroom', 8, 8),
            ))
            # If many classrooms, offer half-size too
            if n_class >= 8:
                half = n_class // 2
                results.append(Pattern(
                    motif_id=f'{motif_meta.get("motif_id","?")}_U2',
                    name=f'教学单元 ({half}教室)',
                    room_counts={'classroom': half, 'corridor': 1},
                    edge_counts={},
                    frequency=motif_meta.get('frequency', 0),
                    percentage=motif_meta.get('percentage', 0),
                    layout=self._make_grid(half, 1, 'classroom', 8, 8),
                ))

        # ── Circulation core: staircase + toilet ──
        n_stair = int(comp.get('staircase', 0))
        n_toilet = int(comp.get('toilet', 0))
        if n_stair >= 1 and n_toilet >= 1:
            results.append(Pattern(
                motif_id=f'{motif_meta.get("motif_id","?")}_C1',
                name='交通核 (楼梯+卫生间)',
                room_counts={'staircase': 1, 'toilet': 1},
                edge_counts={},
                frequency=motif_meta.get('frequency', 0),
                percentage=motif_meta.get('percentage', 0),
                layout=[
                    {'room_type': 'staircase', 'x': 0, 'y': 0, 'w': 4, 'h': 8},
                    {'room_type': 'toilet', 'x': 4, 'y': 0, 'w': 4, 'h': 5},
                ],
            ))

        # ── Office cluster ──
        n_office = int(comp.get('office', 0)) + int(comp.get('teacher_office', 0))
        if n_office >= 2:
            results.append(Pattern(
                motif_id=f'{motif_meta.get("motif_id","?")}_O1',
                name='办公区 (办公室×2)',
                room_counts={'office': 2},
                edge_counts={},
                frequency=motif_meta.get('frequency', 0),
                percentage=motif_meta.get('percentage', 0),
                layout=self._make_grid(2, 1, 'office', 4, 6),
            ))

        # ── Special classroom cluster ──
        n_special = int(comp.get('special_classroom', 0))
        if n_special >= 1:
            results.append(Pattern(
                motif_id=f'{motif_meta.get("motif_id","?")}_S1',
                name='专用教室区',
                room_counts={'special_classroom': min(n_special, 3)},
                edge_counts={},
                frequency=motif_meta.get('frequency', 0),
                percentage=motif_meta.get('percentage', 0),
                layout=self._make_grid(min(n_special, 3), 1, 'special_classroom', 8, 10),
            ))

        return results if results else self._make_default_patterns()

    def _make_default_patterns(self):
        """Create fallback patterns when no motif dictionary exists."""
        self.patterns = [
            Pattern('P1', '教学单元 (4教室+走道)', {'classroom': 4, 'corridor': 1}, {}, 10, 0.3,
                    layout=self._make_grid(4, 1, 'classroom', 8, 8)),
            Pattern('P2', '教学单元 (2教室)', {'classroom': 2}, {}, 8, 0.25,
                    layout=self._make_grid(2, 1, 'classroom', 8, 8)),
            Pattern('P3', '教师办公+教室', {'classroom': 2, 'teacher_office': 1}, {}, 6, 0.2,
                    layout=self._make_grid(2, 1, 'classroom', 8, 8)),
            Pattern('P4', '交通核 (楼梯+卫生间)', {'staircase': 1, 'toilet': 1}, {}, 5, 0.15,
                    layout=[
                        {'room_type': 'staircase', 'x': 0, 'y': 0, 'w': 4, 'h': 8},
                        {'room_type': 'toilet', 'x': 4, 'y': 0, 'w': 4, 'h': 8},
                    ]),
            Pattern('P5', '专用教室区', {'special_classroom': 2, 'music_room': 1}, {}, 4, 0.1,
                    layout=self._make_grid(2, 1, 'special_classroom', 8, 10)),
        ]

    def _make_grid(self, n: int, rows: int, room_type: str, w: float, h: float) -> List[Dict]:
        """Make a simple grid layout for a pattern."""
        layout = []
        for i in range(n):
            col = i % (n // max(1, rows))
            row = i // (n // max(1, rows))
            layout.append({
                'room_type': room_type,
                'x': col * (w + 0.3), 'y': row * (h + 0.3),
                'w': w, 'h': h,
            })
        return layout

    def _compute_pattern_layout(self, room_comp: Dict[str, float]) -> List[Dict]:
        """Compute a compact layout template using only the top 5 room types."""
        layout = []
        cx, cy = 0.0, 0.0
        # Filter to real room types, take top 5 by count
        valid_types = set(ROOM_COLORS.keys())
        top = [(rt, cnt) for rt, cnt in sorted(room_comp.items(), key=lambda x: -x[1])
               if rt in valid_types and rt != 'corridor'][:5]
        for rt, count in top:
            n = max(1, min(int(count), 6))  # cap at 6 per type
            dw, dh = DEFAULT_SIZES.get(rt, (6, 6))
            for _ in range(n):
                layout.append({
                    'room_type': rt, 'x': cx, 'y': cy,
                    'w': dw, 'h': dh,
                })
                cx += dw + 0.3
                if cx > 40:
                    cx = 0
                    cy += dh + 1.0
        return layout

    # ══════════════════════════════════════════════════════════════
    # Placement
    # ══════════════════════════════════════════════════════════════

    def place_pattern(self, pattern_idx: int, x: float = None, y: float = None):
        """Place a pattern's rooms onto the canvas at the given position."""
        if pattern_idx < 0 or pattern_idx >= len(self.patterns):
            return

        pat = self.patterns[pattern_idx]
        if x is None:
            x = self.placement_x
        if y is None:
            y = self.placement_y

        for room_spec in pat.layout:
            rt = room_spec['room_type']
            idx = self.next_label_counters[rt]
            self.next_label_counters[rt] += 1
            label = f'{ROOM_CN.get(rt, rt)}_{idx}'

            self.placed_rooms.append(PlacedRoom(
                room_type=rt,
                x=x + room_spec.get('x', 0),
                y=y + room_spec.get('y', 0),
                width=room_spec.get('w', 6),
                height=room_spec.get('h', 6),
                label=label,
                color=ROOM_COLORS.get(rt, '#CCC'),
                is_corridor=(rt == 'corridor'),
            ))

        # Advance placement cursor
        pat_w = max(r['x'] + r['w'] for r in pat.layout)
        self.placement_x += pat_w + 5.0
        if self.placement_x > self.CANVAS_W - 20:
            self.placement_x = 5.0
            self.placement_y += 15.0

    def auto_complete(self):
        """
        Add only missing essentials: corridor, staircases (≥2), toilets.
        Does NOT duplicate already-present room types.
        """
        types_present = set(r.room_type for r in self.placed_rooms)
        counts = defaultdict(int)
        for r in self.placed_rooms:
            counts[r.room_type] += 1

        if not types_present:
            return

        xs = [r.x for r in self.placed_rooms] + [r.x + r.width for r in self.placed_rooms]
        ys = [r.y for r in self.placed_rooms] + [r.y + r.height for r in self.placed_rooms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_w = max_x - min_x

        # Corridor at mid-height if missing
        if counts['corridor'] == 0:
            cy = (min_y + max_y) / 2 - self.CORRIDOR_H / 2
            self.placed_rooms.append(PlacedRoom(
                room_type='corridor', x=min_x - 2, y=cy,
                width=span_w + 4, height=self.CORRIDOR_H,
                label='走道_auto', color=ROOM_COLORS['corridor'], is_corridor=True,
            ))

        # Add staircases up to 2
        need_stairs = max(0, 2 - counts['staircase'])
        for i in range(need_stairs):
            sx = (min_x - 5) if i == 0 else (max_x + 1)
            self.placed_rooms.append(PlacedRoom(
                room_type='staircase', x=sx, y=min_y,
                width=4, height=8, label=f'楼梯间_auto{i}',
                color=ROOM_COLORS['staircase'],
            ))

        # Add toilets up to 1 per 30m span, only if missing
        if counts['toilet'] == 0:
            n_t = max(1, int(span_w / 35) + 1)
            for i in range(n_t):
                tx = min_x + i * span_w / n_t
                self.placed_rooms.append(PlacedRoom(
                    room_type='toilet', x=tx, y=max_y + 1,
                    width=4, height=5, label=f'卫生间_auto{i}',
                    color=ROOM_COLORS['toilet'],
                ))

    def clear(self):
        """Reset the canvas."""
        self.placed_rooms = []
        self.next_label_counters = defaultdict(int)
        self.placement_x = 5.0
        self.placement_y = 5.0
        self.gnn_score = None
        self.validation = {}

    # ══════════════════════════════════════════════════════════════
    # GNN Scoring
    # ══════════════════════════════════════════════════════════════

    def to_hetero_data(self):
        """
        Convert canvas rooms to a simplified HeteroData for GNN scoring.
        Creates room nodes and physical_connects edges between adjacent rooms.
        """
        from torch_geometric.data import HeteroData

        if not self.placed_rooms:
            return None

        # Build room features (simplified: just room type one-hot)
        import torch
        room_types = list(RoomType)
        N = len(self.placed_rooms)
        x = torch.zeros(N, 27)

        for i, pr in enumerate(self.placed_rooms):
            # RoomType one-hot
            for j, rt in enumerate(room_types):
                if rt.value == pr.room_type:
                    x[i, j] = 1.0
                    break
            # Area (normalized)
            area = pr.width * pr.height
            x[i, 13] = min(1.0, area / 800.0)
            # Floor (normalized)
            x[i, 19] = 0.5  # middle floor

        data = HeteroData()
        data['room'].x = x
        data['room'].num_nodes = N
        data['room'].room_ids = [pr.label for pr in self.placed_rooms]

        # Build physical edges between rooms that touch or overlap
        edges_src, edges_dst = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if self._rooms_adjacent(self.placed_rooms[i], self.placed_rooms[j]):
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])

        if edges_src:
            data['room', 'physical_connects', 'room'].edge_index = torch.tensor(
                [edges_src, edges_dst], dtype=torch.long
            )
        else:
            data['room', 'physical_connects', 'room'].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Empty edge types (required by model)
        for et in [('room', 'acoustic_blocks', 'room'),
                    ('room', 'sight_lines', 'room'),
                    ('room', 'sight_lines', 'environment'),
                    ('room', 'physical_connects', 'environment')]:
            data[et].edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Environment node (placeholder)
        data['environment'].x = torch.zeros(1, 6)
        data['environment'].num_nodes = 1
        data['environment'].env_ids = ['env_0']

        return data

    @staticmethod
    def _rooms_adjacent(a: PlacedRoom, b: PlacedRoom) -> bool:
        """Check if two placed rooms share a boundary or overlap."""
        gap = 0.5
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height

        # Overlap or touch within gap
        x_touch = (ax1 - gap <= bx2 and ax2 + gap >= bx1)
        y_touch = (ay1 - gap <= by2 and ay2 + gap >= by1)
        return x_touch and y_touch

    def score_with_gnn(self, model) -> Optional[float]:
        """Score the current canvas layout using the trained GNN."""
        data = self.to_hetero_data()
        if data is None or model is None:
            return None
        model.eval()
        with torch.no_grad():
            self.gnn_score = model(data).item()
        return self.gnn_score

    # ══════════════════════════════════════════════════════════════
    # Rendering
    # ══════════════════════════════════════════════════════════════

    def render_canvas(self, figsize=(14, 8), dpi=100):
        """Render the current canvas as a matplotlib figure."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.font_manager as fm
        _fonts = [f.name for f in fm.fontManager.ttflist]
        _cn = 'SimHei' if 'SimHei' in _fonts else ('Microsoft YaHei' if 'Microsoft YaHei' in _fonts else 'sans-serif')
        matplotlib.rcParams['font.sans-serif'] = [_cn, 'Times New Roman', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Canvas boundary
        ax.add_patch(mpatches.Rectangle(
            (0, 0), self.CANVAS_W, self.CANVAS_H,
            facecolor='#FAFBFC', edgecolor='#999', linewidth=1.5,
            linestyle='--', zorder=0,
        ))

        # Draw placed rooms
        for pr in self.placed_rooms:
            rect = mpatches.Rectangle(
                (pr.x, pr.y), pr.width, pr.height,
                facecolor=pr.color, edgecolor='#333', linewidth=1.0,
                alpha=0.88, zorder=2,
            )
            ax.add_patch(rect)
            if pr.width > 3 and pr.height > 2:
                ax.text(pr.x + pr.width/2, pr.y + pr.height/2,
                        pr.label, ha='center', va='center',
                        fontsize=6, fontweight='bold', zorder=3)

        # Placement cursor
        ax.plot(self.placement_x, self.placement_y, 'rx', markersize=10, zorder=5)

        # Title with score
        title = '交互式设计画布 — Interactive Design Canvas'
        if self.gnn_score is not None:
            title += f'  |  GNN Score: {self.gnn_score:.3f}'
        ax.set_title(title, fontsize=13, fontweight='bold')

        ax.set_xlim(-2, self.CANVAS_W + 2)
        ax.set_ylim(-2, self.CANVAS_H + 2)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout()
        return fig
