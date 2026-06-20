"""
正交平面图生成器 — Orthogonal Floor Plan Generator
====================================================
从 SchoolGraph-200 数据集中提取真实图数据，生成紧凑正交平面图。
输出: outputs/presentation/ 下的正交布局图

Usage: python create_ortho_plans.py
"""

import os, sys, json, math, random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# ── Chinese font ──
_cn_fonts = [f.name for f in fm.fontManager.ttflist]
_heiti = 'SimHei' if 'SimHei' in _cn_fonts else 'Microsoft YaHei' if 'Microsoft YaHei' in _cn_fonts else 'sans-serif'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [_heiti, 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──
OUTPUT_DIR = Path('outputs/presentation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path('outputs/dataset_200_new')

# Add src to path
sys.path.insert(0, 'src')

from utils.enums import RoomType, EnvNodeType, EdgeCategory
from ui.layout_engine import OrthogonalLayoutEngine, OrthoRoom, OrthoLayout

# ── Color scheme ──
ROOM_COLORS = {
    'classroom':         '#4C72B0',
    'special_classroom': '#55A868',
    'music_room':        '#C44E52',
    'gymnasium':         '#DD8452',
    'library':           '#937860',
    'office':            '#8CA5C8',
    'teacher_office':    '#7FB8D0',
    'corridor':          '#E8E8E8',
    'staircase':         '#8172B2',
    'toilet':            '#B0B0B0',
    'storage':           '#D8D8D8',
    'cafeteria':         '#E8B44F',
    'entrance_hall':     '#64B5CD',
    'unknown':           '#CCCCCC',
}

ROOM_CN = {
    'classroom': '教室', 'special_classroom': '专用教室',
    'music_room': '音乐教室', 'gymnasium': '体育馆',
    'library': '图书馆', 'office': '办公室',
    'teacher_office': '教师办公', 'corridor': '走道',
    'staircase': '楼梯间', 'toilet': '卫生间',
    'storage': '储藏室', 'cafeteria': '食堂',
    'entrance_hall': '门厅',
}


# ═══════════════════════════════════════════════════════════════════
# Step 1: Convert HeteroData → RoomNode list
# ═══════════════════════════════════════════════════════════════════

class RoomProxy:
    """Minimal room proxy compatible with OrthogonalLayoutEngine."""
    def __init__(self, room_id: str, room_type, area: float,
                 floor: int, floor_range: tuple, typical_floor: str,
                 centroid: tuple, zone_id: int):
        self.room_id = room_id
        self.room_type = room_type
        self.area = area
        self.floor = floor
        self.floor_range = floor_range
        self.typical_floor = typical_floor
        self.centroid = centroid
        self.zone_id = zone_id


def load_graph(pt_path: str):
    """Load a graph from .pt file and extract room proxies for layout."""
    bundle = torch.load(str(pt_path), weights_only=False)
    hd = bundle['hetero_data']
    meta = bundle.get('metadata', {})

    room_x = hd['room'].x
    num_rooms = room_x.shape[0]

    room_type_map = list(RoomType)

    rooms = []
    for i in range(num_rooms):
        feat = room_x[i]

        # Room type from one-hot [0:13]
        rt_idx = int(feat[:13].argmax().item())
        room_type = room_type_map[rt_idx] if rt_idx < len(room_type_map) else RoomType.CORRIDOR

        # Area (index 13, normalized by max_area=800)
        area = feat[13].item() * 800.0
        area = max(12.0, min(area, 200.0))  # Clamp to reasonable range

        # Floor range from features [19, 20]
        floor_lo = int(round(feat[19].item() * 6))
        floor_hi = int(round(feat[20].item() * 6))
        floor_lo = max(0, min(floor_lo, 5))
        floor_hi = max(floor_lo, min(floor_hi, 5))

        # Typical floor
        floor_mid = (floor_lo + floor_hi) / 2.0
        if floor_mid < 1.0:
            tf = 'ground'
        elif floor_mid > 3.5:
            tf = 'top'
        else:
            tf = 'teaching'

        # Centroid from features [22, 23]
        cx = feat[22].item() * 100.0
        cy = feat[23].item() * 100.0

        room_id = f"{room_type.value}_{i:03d}_{tf[0].upper()}{floor_lo}"

        rooms.append(RoomProxy(
            room_id=room_id,
            room_type=room_type,
            area=area,
            floor=floor_lo,
            floor_range=(floor_lo, floor_hi),
            typical_floor=tf,
            centroid=(cx, cy),
            zone_id=0,
        ))

    return rooms, meta, hd


# ═══════════════════════════════════════════════════════════════════
# Step 2: Draw orthogonal plan
# ═══════════════════════════════════════════════════════════════════

def draw_ortho_plan(
    layout: OrthoLayout,
    title: str,
    subtitle: str = "",
    figsize: tuple = None,
    dpi: int = 200,
):
    """Render an OrthoLayout as a presentation-quality architectural plan."""
    rooms = layout.rooms
    if not rooms:
        print("  [WARN] No rooms to draw")
        return None

    bw, bh = layout.boundary
    margin = 4.0

    # Scale factors for rendering
    max_dim = max(bw, bh * 0.5)
    target_w = 16.0
    scale = target_w / max_dim if max_dim > 0 else 1.0

    if figsize is None:
        figsize = (18, max(6, bh * scale * 1.3))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-margin, bw + margin)
    ax.set_ylim(-margin, bh + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Draw rooms ──
    for room in rooms:
        color = ROOM_COLORS.get(room.room_type, '#CCCCCC')
        edge_color = '#999999'

        if room.room_type == 'corridor':
            # Corridor: lighter, dashed border hint
            rect = Rectangle(
                (room.x, room.y), room.width, room.height,
                linewidth=1.0, edgecolor='#BBBBBB',
                facecolor=color, alpha=0.7, zorder=1,
            )
        else:
            rect = Rectangle(
                (room.x, room.y), room.width, room.height,
                linewidth=1.2, edgecolor=edge_color,
                facecolor=color, alpha=0.9, zorder=2,
            )
        ax.add_patch(rect)

        # ── Room label ──
        cn_name = ROOM_CN.get(room.room_type, room.room_type)
        label = cn_name

        # Add area for larger rooms
        if room.width >= 3.5 and room.height >= 3.0:
            label += f"\n{room.area:.0f} sqm"

        fontsize = 5.5 if room.width < 4 else 6.5

        # Use white text on dark backgrounds, black on light
        if room.room_type in ('classroom', 'staircase', 'music_room',
                              'library', 'gymnasium', 'cafeteria'):
            text_color = 'white'
        else:
            text_color = '#333333'

        ax.text(
            room.x + room.width / 2,
            room.y + room.height / 2,
            label,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=text_color,
            zorder=3,
        )

    # ── Boundary ──
    boundary = Rectangle(
        (0, 0), bw, bh,
        linewidth=2.0, edgecolor='#333333',
        facecolor='none', linestyle='-', zorder=0,
    )
    ax.add_patch(boundary)

    # ── Scale bar ──
    scale_bar_y = -margin + 1.0
    scale_bar_x = 2.0
    scale_bar_w = 10.0  # 10 meters
    ax.plot([scale_bar_x, scale_bar_x + scale_bar_w],
            [scale_bar_y, scale_bar_y],
            'k-', linewidth=2, zorder=10)
    ax.plot([scale_bar_x, scale_bar_x],
            [scale_bar_y - 0.4, scale_bar_y + 0.4],
            'k-', linewidth=1.5, zorder=10)
    ax.plot([scale_bar_x + scale_bar_w, scale_bar_x + scale_bar_w],
            [scale_bar_y - 0.4, scale_bar_y + 0.4],
            'k-', linewidth=1.5, zorder=10)
    ax.text(scale_bar_x + scale_bar_w / 2, scale_bar_y - 0.8,
            '10 m', ha='center', fontsize=8, color='#333333', zorder=10)

    # ── Compass rose (North arrow) ──
    compass_x = bw - 2.5
    compass_y = -margin + 1.0
    ax.annotate('', xy=(compass_x, compass_y + 2.5), xytext=(compass_x, compass_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.text(compass_x + 0.4, compass_y + 0.5, 'N', fontsize=9,
            fontweight='bold', color='#333333')

    # ── Title ──
    ax.set_title(title, fontsize=16, fontweight='bold', pad=18, color='#222222')
    if subtitle:
        ax.text(
            0.5, -0.06, subtitle,
            transform=ax.transAxes,
            ha='center', fontsize=9, color='#888888', style='italic',
        )

    # ── Legend ──
    legend_elements = []
    room_types_in_plan = sorted(set(r.room_type for r in rooms))
    for rt in room_types_in_plan:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=ROOM_COLORS.get(rt, '#CCC'),
                   markersize=12, label=ROOM_CN.get(rt, rt))
        )

    ncol = min(7, max(3, len(legend_elements) // 2))
    leg = ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=ncol, fontsize=7.5,
        frameon=True, fancybox=True,
        title='图例', title_fontsize=8,
    )

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Step 3: Generate plans from dataset
# ═══════════════════════════════════════════════════════════════════

def main():
    engine = OrthogonalLayoutEngine()

    # Pick representative graphs: small, medium, large
    raw_dir = DATASET_DIR / 'raw'
    all_files = sorted(raw_dir.glob('*.pt'))

    # Pick the best graphs by looking at metadata for quality
    small_files = [f for f in all_files if 'small' in f.stem]
    medium_files = [f for f in all_files if 'medium' in f.stem]
    large_files = [f for f in all_files if 'large' in f.stem]

    selections = []

    # Small: pick one with 4+ floors (more interesting layout)
    for f in small_files:
        bundle = torch.load(str(f), weights_only=False)
        n_floors = bundle.get('metadata', {}).get('num_floors', 0)
        n_rooms = bundle['hetero_data']['room'].x.shape[0]
        if n_floors >= 3 and 40 <= n_rooms <= 70:
            selections.append(('小型学校', f, n_floors, n_rooms))
            break

    # Medium: pick a well-structured one
    for f in medium_files:
        bundle = torch.load(str(f), weights_only=False)
        n_floors = bundle.get('metadata', {}).get('num_floors', 0)
        n_rooms = bundle['hetero_data']['room'].x.shape[0]
        if n_floors >= 4 and 65 <= n_rooms <= 95:
            selections.append(('中型学校', f, n_floors, n_rooms))
            break

    # Large: pick the biggest interesting one
    for f in large_files:
        bundle = torch.load(str(f), weights_only=False)
        n_floors = bundle.get('metadata', {}).get('num_floors', 0)
        n_rooms = bundle['hetero_data']['room'].x.shape[0]
        if n_floors >= 5 and n_rooms >= 80:
            selections.append(('大型学校', f, n_floors, n_rooms))
            break

    if len(selections) < 3:
        # Fallback: pick any
        remaining = [f for f in all_files if f not in [s[1] for s in selections]]
        for f in remaining[:3 - len(selections)]:
            bundle = torch.load(str(f), weights_only=False)
            n_floors = bundle.get('metadata', {}).get('num_floors', 0)
            n_rooms = bundle['hetero_data']['room'].x.shape[0]
            selections.append(('学校', f, n_floors, n_rooms))

    print(f"Selected {len(selections)} graphs for orthogonal plan generation:")
    for label, f, nf, nr in selections:
        print(f"  {label}: {f.stem} ({nf} floors, {nr} rooms)")

    # ── Generate each plan ──
    for idx, (label, pt_file, n_floors, n_rooms) in enumerate(selections):
        print(f"\n[{idx+1}/{len(selections)}] Generating plan for {label}: {pt_file.stem}...")

        rooms, meta, hd = load_graph(str(pt_file))

        # Run layout engine
        layout = engine.layout(rooms, num_floors=n_floors)

        if not layout.rooms:
            print(f"  [SKIP] No rooms placed by layout engine")
            continue

        # Count room types
        type_counts = defaultdict(int)
        for r in layout.rooms:
            type_counts[r.room_type] += 1
        type_str = ', '.join(f'{ROOM_CN.get(t, t)}×{c}'
                            for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:8])

        # Draw
        title = f"{label}标准层正交平面图"
        subtitle = (
            f"{pt_file.stem} | {n_floors}层 {n_rooms}间 | "
            f"布局尺寸 {layout.width:.0f}×{layout.height:.0f}m | {type_str}"
        )

        fig = draw_ortho_plan(layout, title=title, subtitle=subtitle)

        if fig:
            out_name = f'07_ortho_plan_{label}.png'
            fig.savefig(OUTPUT_DIR / out_name, dpi=200,
                       bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  [OK] {out_name}")

    # ── Bonus: Multi-floor stacked view for the medium school ──
    if len(selections) >= 2:
        print(f"\n[Bonus] Generating multi-floor stacked view...")
        _, pt_file, n_floors, n_rooms = selections[1]  # medium school
        rooms, meta, hd = load_graph(str(pt_file))

        # Create separate layouts for each typical floor
        engine2 = OrthogonalLayoutEngine()
        layout = engine2.layout(rooms, num_floors=n_floors)

        # Draw with floor annotations
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        bw, bh = layout.boundary
        margin = 4.0
        ax.set_xlim(-margin, bw + margin)
        ax.set_ylim(-margin, bh + margin)
        ax.set_aspect('equal')
        ax.axis('off')

        # Find floor boundaries
        floor_bounds = defaultdict(lambda: [float('inf'), 0])
        for r in layout.rooms:
            fy = int(r.y / (layout.height / max(1, layout.num_floors)))
            key = fy
            floor_bounds[key][0] = min(floor_bounds[key][0], r.y)
            floor_bounds[key][1] = max(floor_bounds[key][1], r.y + r.height)

        for room in layout.rooms:
            color = ROOM_COLORS.get(room.room_type, '#CCCCCC')
            rect = Rectangle(
                (room.x, room.y), room.width, room.height,
                linewidth=1.0 if room.room_type == 'corridor' else 1.2,
                edgecolor='#BBBBBB' if room.room_type == 'corridor' else '#999999',
                facecolor=color,
                alpha=0.7 if room.room_type == 'corridor' else 0.9,
                zorder=2,
            )
            ax.add_patch(rect)

            # Labels only for key rooms
            if room.room_type in ('classroom', 'staircase', 'toilet',
                                   'office', 'entrance_hall'):
                cn = ROOM_CN.get(room.room_type, room.room_type)
                fs = 4.5 if room.width < 4 else 5.5
                tc = 'white' if room.room_type in ('classroom', 'staircase') else '#333333'
                ax.text(room.x + room.width/2, room.y + room.height/2,
                       cn, ha='center', va='center', fontsize=fs,
                       fontweight='bold', color=tc, zorder=3)

        # Floor labels
        for floor_key, (y_min, y_max) in sorted(floor_bounds.items()):
            floor_names = {0: '首层 Ground', 1: '标准层 Teaching', 2: '顶层 Top'}
            fn = floor_names.get(floor_key, f'Floor {floor_key}')
            y_mid = (y_min + y_max) / 2
            ax.text(bw + 1.5, y_mid, fn, va='center', fontsize=11,
                   fontweight='bold', color='#333333', rotation=0)

        # Boundary
        ax.add_patch(Rectangle((0, 0), bw, bh, linewidth=2.0,
                               edgecolor='#333333', facecolor='none', zorder=0))

        ax.set_title(f'学校教学综合楼 — 全楼层叠加正交平面图\n{pt_file.stem} | {n_floors}层 {n_rooms}间',
                    fontsize=16, fontweight='bold', pad=15)

        # Legend
        rt_in_plan = sorted(set(r.room_type for r in layout.rooms))
        legend_elems = [Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=ROOM_COLORS.get(rt, '#CCC'),
                              markersize=12, label=ROOM_CN.get(rt, rt))
                       for rt in rt_in_plan]
        ax.legend(handles=legend_elems, loc='upper center',
                 bbox_to_anchor=(0.5, -0.06), ncol=7, fontsize=7,
                 frameon=True, title='图例', title_fontsize=8)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / '08_ortho_stacked_view.png', dpi=200,
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print("  [OK] 08_ortho_stacked_view.png")

    print(f"\nAll plans saved to: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
