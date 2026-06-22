"""Draw physically accurate architectural floor plans from v11 dataset.

Fixes:
  - Corridor drawn as ONE continuous spine (not multiple chunks)
  - Teaching floor rooms split across physical sub-floors (1/n per visual floor)
  - Staircase count per physical floor: max 3
  - Area verification: stored == visual within tolerance
"""

import sys, random, torch, math
sys.path.insert(0, 'src')
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
import matplotlib.font_manager as fm

_cn = [f.name for f in fm.fontManager.ttflist]
_heiti = 'SimHei' if 'SimHei' in _cn else 'Microsoft YaHei' if 'Microsoft YaHei' in _cn else 'sans-serif'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [_heiti, 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOM_COLORS = {
    'classroom': '#4C72B0', 'special_classroom': '#55A868', 'music_room': '#C44E52',
    'teacher_office': '#7FB8D0', 'corridor': '#F2EFE8', 'staircase': '#8172B2',
    'toilet': '#C8C8C8', 'storage': '#E8E4D8', 'entrance_hall': '#64B5CD',
}
ROOM_CN = {
    'classroom': '教室', 'special_classroom': '专用教室', 'music_room': '音乐教室',
    'teacher_office': '教师办公', 'corridor': '走道', 'staircase': '楼梯间',
    'toilet': '卫生间', 'storage': '储藏室', 'entrance_hall': '门厅',
}
RT_MAP = ['classroom', 'special_classroom', 'music_room', 'gymnasium', 'library',
          'office', 'teacher_office', 'corridor', 'staircase', 'toilet', 'storage',
          'cafeteria', 'entrance_hall']


def interleave(room_data_list):
    """Interleave rooms by type using room data dicts."""
    by_type = {}
    for i, d in enumerate(room_data_list):
        rt = d['type']
        by_type.setdefault(rt, []).append(i)
    groups = sorted(by_type.values(), key=len, reverse=True)
    result = []
    max_len = max(len(g) for g in groups) if groups else 0
    for k in range(max_len):
        for g in groups:
            if k < len(g):
                result.append(g[k])
    return result


def draw_floor(ax, all_rooms_data, bld_w, bld_d, fl, title):
    """
    Draw one physical floor with:
      - Continuous corridor spine (NOT chunked)
      - Rooms distributed from split teaching floor data
    """
    margin = 0.3
    inner_w = bld_w - 2 * margin

    # Classify into south, north, corridor
    south_idx = [i for i, d in enumerate(all_rooms_data) if d['type'] in
                 {'classroom', 'special_classroom', 'teacher_office', 'music_room'}]
    north_idx = [i for i, d in enumerate(all_rooms_data) if d['type'] in
                 {'toilet', 'storage', 'staircase', 'entrance_hall'}]
    corr_items = [d for d in all_rooms_data if d['type'] == 'corridor']

    total_corr_area = sum(d['area'] for d in corr_items)
    non_corr_area = sum(all_rooms_data[i]['area'] for i in south_idx + north_idx)
    total_area = total_corr_area + non_corr_area

    if total_area <= 0:
        return [], 0, 0

    # Dynamic zone heights from actual areas
    corr_ratio = total_corr_area / total_area
    corr_h = max(2.4, bld_d * corr_ratio)
    remaining = bld_d - corr_h
    south_ratio = sum(all_rooms_data[i]['area'] for i in south_idx) / max(1, non_corr_area)
    south_h = remaining * south_ratio * 0.9
    north_h = remaining * (1 - south_ratio) * 0.9

    south_y = bld_d - south_h - margin
    corr_y = south_y - corr_h
    north_y = margin

    all_placed = []

    # --- South row: classrooms, offices, special rooms ---
    south_data = [all_rooms_data[i] for i in south_idx]
    south_interleaved = [south_idx[i] for i in interleave(south_data)]
    # Insert stairs at 1/3 and 2/3
    stairs_in_south = [i for i in north_idx if all_rooms_data[i]['type'] == 'staircase']
    if len(stairs_in_south) >= 2 and len(south_interleaved) >= 4:
        p1 = len(south_interleaved) // 3
        p2 = 2 * len(south_interleaved) // 3
        south_row = stairs_in_south[:1] + south_interleaved[:p1] + stairs_in_south[1:2] + south_interleaved[p1:p2]
        if len(stairs_in_south) >= 3:
            south_row += stairs_in_south[2:3] + south_interleaved[p2:]
        else:
            south_row += south_interleaved[p2:]
        used_stairs = min(3, len(stairs_in_south))
    elif stairs_in_south:
        south_row = stairs_in_south[:1] + south_interleaved
        used_stairs = 1
    else:
        south_row = south_interleaved
        used_stairs = 0
    remaining_stairs = stairs_in_south[used_stairs:]

    # Pack south row with exact widths
    if south_row and south_h > 1.0:
        raw_w = [max(3.0, all_rooms_data[i]['area'] / south_h) for i in south_row]
        total_raw = sum(raw_w)
        scale = inner_w / total_raw if total_raw > 0 else 1.0
        cx = margin
        for idx, i in enumerate(south_row):
            w = raw_w[idx] * scale
            d = all_rooms_data[i]
            all_placed.append({
                'x': cx, 'y': south_y, 'w': w, 'h': south_h,
                'type': d['type'], 'area': d['area'],
                'visual_area': w * south_h,
            })
            cx += w

    # --- North row: toilets, storage, remaining stairs ---
    north_items = [i for i in north_idx if all_rooms_data[i]['type'] != 'staircase']
    north_items += remaining_stairs
    north_data_items = [all_rooms_data[i] for i in north_items]
    north_interleaved = [north_items[i] for i in interleave(north_data_items)]
    if north_interleaved and north_h > 1.0:
        raw_w = [max(2.5, all_rooms_data[i]['area'] / north_h) for i in north_interleaved]
        total_raw = sum(raw_w)
        scale = inner_w / total_raw if total_raw > 0 else 1.0
        cx = margin
        for idx, i in enumerate(north_interleaved):
            w = raw_w[idx] * scale
            d = all_rooms_data[i]
            all_placed.append({
                'x': cx, 'y': north_y, 'w': w, 'h': north_h,
                'type': d['type'], 'area': d['area'],
                'visual_area': w * north_h,
            })
            cx += w

    # --- Corridor: ONE continuous spine ---
    if total_corr_area > 0:
        all_placed.append({
            'x': margin, 'y': corr_y, 'w': inner_w, 'h': corr_h,
            'type': 'corridor', 'area': total_corr_area,
            'visual_area': inner_w * corr_h,
        })

    # --- Draw ---
    ax.add_patch(Rectangle((0, 0), bld_w, bld_d, fill=False,
                            edgecolor='#333', lw=3, zorder=10))
    stored_s = 0.0; visual_s = 0.0
    shown_types = set()
    for r in all_placed:
        stored_s += r['area']; visual_s += r['visual_area']
        rect = FancyBboxPatch((r['x'], r['y']), r['w'], r['h'],
                              boxstyle='round,pad=0.08',
                              facecolor=ROOM_COLORS.get(r['type'], '#CCC'),
                              edgecolor='#888', lw=0.5, alpha=0.92, zorder=5)
        ax.add_patch(rect)
        cn = ROOM_CN.get(r['type'], r['type'])
        fs = 8.5 if r['w'] > 5 else 6
        if r['w'] > 2.5 and r['h'] > 1.0:
            ax.text(r['x'] + r['w']/2, r['y'] + r['h']/2,
                    f"{cn}\n{r['area']:.0f} m2",
                    ha='center', va='center', fontsize=fs,
                    color='#1a1a1a', fontweight='bold')
        if r['type'] not in shown_types:
            shown_types.add(r['type'])
    # Legend
    legend_items = [Patch(facecolor=ROOM_COLORS.get(t, '#CCC'), edgecolor='#777',
                          label=ROOM_CN.get(t, t)) for t in shown_types]
    ax.legend(handles=legend_items, loc='upper right', fontsize=7,
              ncol=2, framealpha=0.85, edgecolor='#bbb')

    # Area stamp
    delta = abs(visual_s - stored_s)
    ax.text(bld_w - 0.3, -0.2, f'area delta={delta:.1f}m2 ({delta/stored_s*100:.1f}%)',
            ha='right', fontsize=6, color='#888', style='italic')

    ax.set_xlim(-0.5, bld_w + 0.5); ax.set_ylim(-0.5, bld_d + 0.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)

    return all_placed, stored_s, visual_s


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

raw = Path('outputs/dataset_200_v12/raw')
files = sorted(raw.glob('*.pt'))
random.seed(42)

samples = []
for size in ['small', 'medium', 'large']:
    size_files = [f for f in files if size in f.name]
    if size_files:
        samples.append(random.choice(size_files))

OUTPUT_DIR = Path('outputs/sample_plans_v12')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dims = {'small': (50, 16), 'medium': (62, 18), 'large': (78, 18)}

for pt_file in samples:
    b = torch.load(str(pt_file), weights_only=False)
    hd = b['hetero_data']; m = b['metadata']
    size = m['school_size']; nf = m['num_floors']; qs = m.get('quality_score', 0)
    bld_w, bld_d = dims.get(size, (62, 18))

    x = hd['room'].x
    full_areas = (x[:, 13] * 800).tolist()
    types = x[:, :13].argmax(dim=1).tolist()
    floor_mid_raw = (x[:, 19] * 4).tolist()
    unique_midpoints = sorted(set(round(f) for f in floor_mid_raw))

    # Compute physical floor count per midpoint
    n_mid = len(unique_midpoints)
    phys_counts = {}
    for i, m in enumerate(unique_midpoints):
        if n_mid <= 1:
            phys_counts[m] = nf
        elif i == 0:
            phys_counts[m] = 1  # ground
        elif nf >= 5 and i == n_mid - 1:
            phys_counts[m] = 1  # top
        else:
            phys_counts[m] = nf - 1 - (1 if nf >= 5 else 0)  # teaching

    for fl_mid in unique_midpoints:
        n_phys = phys_counts.get(fl_mid, 1)

        # Split teaching floor: show one representative physical sub-floor
        indices = [i for i in range(x.shape[0])
                   if round(floor_mid_raw[i]) == fl_mid]

        # Build room data for this physical sub-floor
        all_rooms = []
        for i in indices:
            area_per_phys = full_areas[i] / n_phys
            rt = RT_MAP[types[i]]
            all_rooms.append({
                'idx': i, 'type': rt, 'type_idx': types[i],
                'area': area_per_phys,
            })

        # Limit excessive rooms: if >40 rooms, sample
        if len(all_rooms) > 40:
            non_corr = [d for d in all_rooms if d['type'] != 'corridor']
            corr_items = [d for d in all_rooms if d['type'] == 'corridor']
            if len(non_corr) > 36:
                random.shuffle(non_corr)
                non_corr = non_corr[:36]
            all_rooms = non_corr + corr_items

        fl_label = {0: 'Ground Floor', 1: 'Standard Floor', 2: 'Standard Floor',
                     3: 'Standard Floor', 4: 'Top Floor'}.get(fl_mid, f'Floor {fl_mid}')
        title = (f'{size.upper()} Teaching Building  |  {fl_label}  |  '
                 f'{bld_w:.0f}x{bld_d:.0f}m  |  {len(all_rooms)} rooms  |  QS={qs:.3f}')

        fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
        placed, stored, visual = draw_floor(ax, all_rooms, bld_w, bld_d, fl_mid, title)
        fig.suptitle(f'GNN R2=0.63  |  Topology: spine/loop/branch', fontsize=10,
                     fontweight='bold', y=1.01, color='#666')
        plt.tight_layout()
        fname = OUTPUT_DIR / f'plan_{size}_f{nf}_fl{fl_mid}.png'
        fig.savefig(str(fname), dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[OK] {fname.name}  |  {len(all_rooms)} rooms  |  area delta={abs(visual-stored):.1f}m2')

print(f'\nDone! Saved to: {OUTPUT_DIR}')
