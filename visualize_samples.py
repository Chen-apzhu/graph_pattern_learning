"""
Quick visualization: generate orthogonal floor plans from dataset samples.
"""
import sys, os
sys.path.insert(0, 'src')

import torch
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

# Chinese font
import matplotlib.font_manager as fm
_cn_fonts = [f.name for f in fm.fontManager.ttflist]
_heiti = 'SimHei' if 'SimHei' in _cn_fonts else 'Microsoft YaHei' if 'Microsoft YaHei' in _cn_fonts else 'sans-serif'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [_heiti, 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# Room type colors and Chinese names
ROOM_COLORS = {
    'classroom': '#4C72B0', 'special_classroom': '#55A868', 'music_room': '#C44E52',
    'gymnasium': '#DD8452', 'library': '#937860', 'office': '#8CA5C8',
    'teacher_office': '#7FB8D0', 'corridor': '#F0F0F0', 'staircase': '#8172B2',
    'toilet': '#B0B0B0', 'storage': '#D8D8D8', 'cafeteria': '#E8B44F',
    'entrance_hall': '#64B5CD',
}

ROOM_CN = {
    'classroom': '教室', 'special_classroom': '专用教室', 'music_room': '音乐教室',
    'gymnasium': '体育馆', 'library': '图书馆', 'office': '办公室',
    'teacher_office': '教师办公', 'corridor': '走道', 'staircase': '楼梯间',
    'toilet': '卫生间', 'storage': '储藏室', 'cafeteria': '食堂',
    'entrance_hall': '门厅',
}

# RoomType order (matches enum)
RT_NAMES = ['classroom', 'special_classroom', 'music_room', 'gymnasium',
            'library', 'office', 'teacher_office', 'corridor',
            'staircase', 'toilet', 'storage', 'cafeteria', 'entrance_hall']


def layout_from_hetero_data(hd, building_w=84.0, building_d=18.0):
    """Create a simple orthogonal layout from HeteroData room features."""
    room_x = hd['room'].x
    areas = room_x[:, 13] * 800  # Denormalize area
    type_idx = room_x[:, :13].argmax(dim=1)
    floor_mid = room_x[:, 19] * 4  # Denormalize floor

    # Find unique physical floors
    unique_floors = sorted(set(round(f.item()) for f in floor_mid))

    floor_layouts = {}
    for fl in unique_floors:
        mask = (floor_mid.round() == fl)
        indices = mask.nonzero(as_tuple=True)[0].tolist()

        # Sort rooms by area (largest first for packing)
        rooms_sorted = sorted(indices, key=lambda i: areas[i].item(), reverse=True)

        layout = _pack_rooms(rooms_sorted, areas, type_idx, building_w, building_d)
        floor_layouts[fl] = layout

    return floor_layouts


def _pack_rooms(indices, areas, type_idx, bld_w, bld_d):
    """Simple strip packing: place rooms in rows, largest first."""
    corridor_y = bld_d * 0.45  # Corridor center line
    corridor_h = 2.4
    margin = 0.5

    # Separate corridors from other rooms
    corr_idx = [i for i in indices if type_idx[i] == 7]
    other_idx = [i for i in indices if type_idx[i] != 7]

    # South-facing rooms (classrooms, offices) go below corridor
    # North-facing rooms (special, service) go above corridor
    south_rooms = []
    north_rooms = []

    target_corr_area = bld_w * corridor_h
    corr_count = len(corr_idx)
    if corr_count > 0:
        corr_area_each = target_corr_area / corr_count
    else:
        corr_area_each = 0

    # Assign rooms to south/north based on type
    south_types = {0, 5, 6}  # classroom, office, teacher_office
    for i in other_idx:
        rt = type_idx[i].item()
        if rt in south_types:
            south_rooms.append(i)
        else:
            north_rooms.append(i)

    placed = []

    # Place south row
    south_used = 0.0
    if south_rooms:
        total_south_area = sum(areas[i].item() for i in south_rooms)
        south_h = bld_d * 0.40
        x = margin
        for i in sorted(south_rooms, key=lambda i: areas[i].item(), reverse=True):
            area_i = areas[i].item()
            w = max(4.0, (area_i / total_south_area) * (bld_w - 2 * margin))
            w = min(w, bld_w - x - margin)
            h = south_h
            y = margin
            placed.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'type': type_idx[i].item(),
                'area': area_i,
            })
            x += w + 0.3
        south_used = x

    # Place north row
    north_used = 0.0
    if north_rooms:
        total_north_area = sum(areas[i].item() for i in north_rooms)
        north_h = bld_d * 0.35
        x = margin
        north_y = corridor_y + corridor_h / 2 + 0.5
        for i in sorted(north_rooms, key=lambda i: areas[i].item(), reverse=True):
            area_i = areas[i].item()
            w = max(4.0, (area_i / total_north_area) * (bld_w - 2 * margin))
            w = min(w, bld_w - x - margin)
            h = north_h
            y = north_y
            placed.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'type': type_idx[i].item(),
                'area': area_i,
            })
            x += w + 0.3
        north_used = x

    # Place corridor in the middle
    if corr_idx:
        n_corr = len(corr_idx)
        corr_w_each = (bld_w - 2 * margin) / n_corr
        for idx, i in enumerate(corr_idx):
            placed.append({
                'x': margin + idx * corr_w_each,
                'y': corridor_y - corridor_h / 2,
                'w': corr_w_each - 0.3,
                'h': corridor_h,
                'type': 7,
                'area': areas[i].item(),
            })

    return placed


def draw_floor(ax, layout, title, bld_w, bld_d):
    """Draw a single floor plan."""
    # Building outline
    ax.add_patch(Rectangle((0, 0), bld_w, bld_d, fill=False,
                            edgecolor='#333', linewidth=2, linestyle='-'))

    for room in layout:
        rt = room['type']
        rt_name = RT_NAMES[rt] if rt < len(RT_NAMES) else 'unknown'
        color = ROOM_COLORS.get(rt_name, '#CCCCCC')
        label = ROOM_CN.get(rt_name, rt_name)

        rect = FancyBboxPatch(
            (room['x'], room['y']), room['w'], room['h'],
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#555', linewidth=1.2,
            alpha=0.85
        )
        ax.add_patch(rect)

        # Label
        cx = room['x'] + room['w'] / 2
        cy = room['y'] + room['h'] / 2
        if room['w'] > 3 and room['h'] > 1.5:
            ax.text(cx, cy, f"{label}\n{room['area']:.0f}m²",
                    ha='center', va='center', fontsize=7, color='#222')

    # Compass
    ax.annotate('N', xy=(bld_w - 1.5, bld_d - 0.5), fontsize=10, fontweight='bold',
                ha='center', va='center', color='#666')

    ax.set_xlim(-1, bld_w + 1)
    ax.set_ylim(-1, bld_d + 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


# ── Main ────────────────────────────────────────────────────────

OUTPUT_DIR = Path('outputs/sample_plans')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load 3 random samples
raw = Path('outputs/dataset_200_v6/raw')
files = sorted(raw.glob('*.pt'))
random.seed(123)
samples = random.sample(files, 3)

for pt_file in samples:
    b = torch.load(str(pt_file), weights_only=False)
    hd = b['hetero_data']
    m = b['metadata']
    qs = m.get('quality_score', 0)
    size = m['school_size']
    nf = m['num_floors']

    # Building dimensions per school size
    dims = {'small': (60, 18), 'medium': (72, 18), 'large': (84, 18)}
    bld_w, bld_d = dims.get(size, (72, 18))

    layouts = layout_from_hetero_data(hd, bld_w, bld_d)

    n_rows = len(layouts)
    fig, axes = plt.subplots(1, n_rows, figsize=(4 * n_rows, 5))
    if n_rows == 1:
        axes = [axes]

    for ax, (fl, layout) in zip(axes, layouts.items()):
        fl_label = {0: '底层', 1: '标准层', 2: '标准层', 3: '标准层', 4: '顶层'}.get(fl, f'F{fl}')
        draw_floor(ax, layout, f'{fl_label} (物理层 {fl})', bld_w, bld_d)

    fig.suptitle(f'{size.upper()} 学校 — {nf}层 | 品质分: {qs:.3f}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = OUTPUT_DIR / f'plan_{size}_f{nf}_{pt_file.stem}.png'
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path.name}')

print(f'\nDone! Plans saved to {OUTPUT_DIR}/')
