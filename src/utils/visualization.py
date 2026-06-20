"""
Graph Visualization — 图可视化工具

Renders school building heterogeneous graphs using matplotlib + networkx.
Supports:
  - Full-graph view (color by room type, edge type)
  - Floor-by-floor subgraph view
  - Constraint violation heatmap
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
_cn_fonts = [f.name for f in fm.fontManager.ttflist]
_simhei = 'SimHei' if 'SimHei' in _cn_fonts else None
_yahei = 'Microsoft YaHei' if 'Microsoft YaHei' in _cn_fonts else None
_heiti = _simhei or _yahei or 'sans-serif'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [_heiti, 'Times New Roman', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ── Color maps ──────────────────────────────────────────────────────────────

ROOM_TYPE_COLORS = {
    'classroom':          '#4C72B0',  # blue
    'special_classroom':  '#55A868',  # green
    'music_room':         '#C44E52',  # red (noisy)
    'gymnasium':          '#DD8452',  # orange
    'library':            '#937860',  # brown
    'office':             '#8CA5C8',  # light blue
    'teacher_office':     '#7FB8D0',  # cyan
    'corridor':           '#CCCCCC',  # grey
    'staircase':          '#8172B2',  # purple
    'toilet':             '#A0A0A0',  # dark grey
    'storage':            '#D0D0D0',  # light grey
    'cafeteria':          '#E8B44F',  # gold
    'entrance_hall':      '#64B5CD',  # teal
}

ENV_TYPE_COLORS = {
    'south_facing':      '#FDD835',  # yellow
    'main_road_access':  '#424242',  # dark
    'playground':        '#66BB6A',  # green
    'green_space':       '#81C784',  # light green
}

EDGE_TYPE_COLORS = {
    'physical_connects':  '#5DADE2',  # steel blue
    'acoustic_blocks':    '#E74C3C',  # red
    'sight_lines':        '#2ECC71',  # green
}


def _get_room_type_name(sg, idx: int) -> str:
    """Extract room type name from one-hot features."""
    from utils.enums import RoomType
    room_x = sg.room_features
    rt_idx = room_x[idx, :13].argmax().item()
    return list(RoomType)[rt_idx].value


def _get_floor(sg, idx: int) -> int:
    """Extract approximate floor number from features (midpoint of floor_range)."""
    return round(sg.room_features[idx, 19].item() * 4)


def _get_typical_floor(sg, idx: int) -> str:
    """Infer typical floor type from features."""
    floor_mid = sg.room_features[idx, 19].item() * 4
    if floor_mid < 0.5:
        return 'ground'
    elif floor_mid > 3.0:
        return 'top'
    return 'teaching'


def draw_full_graph(
    sg,
    ax=None,
    title: str = "School Building Heterogeneous Graph",
    figsize: Tuple[int, int] = (16, 12),
    node_size: int = 80,
    font_size: int = 6,
    seed: int = 42,
) -> "plt.Figure":
    """
    Draw the full heterogeneous graph.

    Nodes colored by room type (or env type for environment nodes).
    Edges colored by edge type.
    """
    import matplotlib.pyplot as plt

    G = sg.to_networkx()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # ── Node colors ──
    node_colors = []
    node_sizes = []
    for node, attrs in G.nodes(data=True):
        ntype = attrs.get('node_type', 'room')
        if ntype == 'room':
            rid = attrs.get('room_id', '')
            # Parse room type from ID or features
            for rt_name in ROOM_TYPE_COLORS:
                if rid.startswith(rt_name):
                    node_colors.append(ROOM_TYPE_COLORS[rt_name])
                    break
            else:
                node_colors.append('#999999')
            node_sizes.append(node_size)
        else:
            eid = attrs.get('env_id', '')
            for et_name in ENV_TYPE_COLORS:
                if eid.startswith(et_name):
                    node_colors.append(ENV_TYPE_COLORS[et_name])
                    break
            else:
                node_colors.append('#FFD700')
            node_sizes.append(node_size * 2.5)

    # ── Edge colors ──
    edge_colors = []
    edge_widths = []
    for u, v, attrs in G.edges(data=True):
        etype = attrs.get('edge_type', 'physical_connects')
        edge_colors.append(EDGE_TYPE_COLORS.get(etype, '#999999'))
        if etype == 'physical_connects':
            edge_widths.append(1.0)
        elif etype == 'acoustic_blocks':
            edge_widths.append(2.5)
        else:
            edge_widths.append(0.8)

    # ── Layout ──
    pos = nx.spring_layout(G, seed=seed, k=2.0, iterations=50)

    # ── Draw ──
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths,
        alpha=0.5, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors='#333333', linewidths=0.3, ax=ax,
    )

    # ── Legends ──
    # Room type legend
    from matplotlib.lines import Line2D
    room_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=8, label=name)
        for name, c in ROOM_TYPE_COLORS.items()
    ]
    env_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=10, label=name)
        for name, c in ENV_TYPE_COLORS.items()
    ]
    edge_legend = [
        Line2D([0], [0], color=c, linewidth=2, label=name)
        for name, c in EDGE_TYPE_COLORS.items()
    ]

    leg1 = ax.legend(handles=room_legend, title='Room Types',
                     loc='upper left', bbox_to_anchor=(1.01, 1.0),
                     fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=edge_legend + env_legend,
                     title='Edges & Env Nodes',
                     loc='upper left', bbox_to_anchor=(1.01, 0.45),
                     fontsize=7, title_fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    return fig


def draw_floor_subgraphs(
    sg,
    figsize: Tuple[int, int] = (18, 10),
    seed: int = 42,
) -> "plt.Figure":
    """
    Draw one subgraph per typical floor (standard floor), showing only
    physical connections.

    Standard floors: ground, teaching, top
    """
    import matplotlib.pyplot as plt

    G = sg.to_networkx()
    n_rooms = sg.num_rooms

    # Group rooms by typical floor type
    tf_groups: Dict[str, set] = {'ground': set(), 'teaching': set(), 'top': set()}
    for i in range(n_rooms):
        tf = _get_typical_floor(sg, i)
        tf_groups[tf].add(f"room_{i}")

    # Include env nodes in ground view
    for i in range(sg.num_env_nodes):
        tf_groups['ground'].add(f"env_{i}")

    tf_labels = {'ground': 'Ground Floor (1F)', 'teaching': 'Teaching Floors',
                 'top': 'Top Floor'}
    active_tfs = [(tf, nodes) for tf, nodes in tf_groups.items() if nodes]
    n_tf = len(active_tfs)
    cols = min(3, n_tf)

    fig, axes = plt.subplots(1, n_tf, figsize=figsize)
    if n_tf == 1:
        axes = [axes]
    axes = np.atleast_1d(axes).flatten()

    for idx, (tf_type, tf_nodes) in enumerate(active_tfs):
        ax = axes[idx]
        sub = G.subgraph(tf_nodes)

        # Only physical edges
        phys_edges = [(u, v) for u, v, a in sub.edges(data=True)
                      if a.get('edge_type') == 'physical_connects']
        phys_sub = nx.Graph()
        phys_sub.add_nodes_from(sub.nodes(data=True))
        phys_sub.add_edges_from(phys_edges)

        node_colors = []
        for node in phys_sub.nodes():
            attrs = phys_sub.nodes[node]
            ntype = attrs.get('node_type', 'room')
            if ntype == 'room':
                rid = attrs.get('room_id', '')
                for rt_name, c in ROOM_TYPE_COLORS.items():
                    if rid.startswith(rt_name):
                        node_colors.append(c)
                        break
                else:
                    node_colors.append('#999999')
            else:
                eid = attrs.get('env_id', '')
                for et_name, c in ENV_TYPE_COLORS.items():
                    if eid.startswith(et_name):
                        node_colors.append(c)
                        break
                else:
                    node_colors.append('#FFD700')

        pos = nx.spring_layout(phys_sub, seed=seed, k=1.5, iterations=30)

        nx.draw_networkx_edges(phys_sub, pos, edge_color='#5DADE2',
                               alpha=0.6, width=1.0, ax=ax)
        nx.draw_networkx_nodes(phys_sub, pos, node_color=node_colors,
                               node_size=100, edgecolors='#333',
                               linewidths=0.3, ax=ax)

        labels = {}
        for node in phys_sub.nodes():
            attrs = phys_sub.nodes[node]
            if attrs.get('node_type') == 'environment':
                labels[node] = attrs.get('env_id', '')[:8]
        if labels:
            nx.draw_networkx_labels(phys_sub, pos, labels, font_size=6, ax=ax)

        degree = sum(1 for n in phys_sub.nodes()
                     if phys_sub.nodes[n].get('node_type') == 'room')
        ax.set_title(f"{tf_labels.get(tf_type, tf_type)}\n({degree} rooms, {len(phys_edges)} edges)",
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    for i in range(n_tf, len(axes)):
        axes[i].axis('off')

    fig.suptitle("Physical Connection Graph — Standard Floors",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    return fig


def draw_constraint_report(
    sg,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """
    Draw a bar chart showing constraint pass/fail status for a single graph.

    Green = passed, Red = failed (with violation count).
    """
    import matplotlib.pyplot as plt

    # Get validation from graph metadata
    # We need to reconstruct from features + edges
    from graph.graph_utils import GraphAnalyzer
    from data.constraints import ConstraintValidator

    # For now, just show degree distribution and basic stats
    analyzer = GraphAnalyzer(sg)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Physical degree distribution
    ax = axes[0]
    phys_ei = sg.physical_edges
    degrees = defaultdict(int)
    if phys_ei.numel() > 0:
        for j in range(phys_ei.shape[1]):
            s, d = phys_ei[0, j].item(), phys_ei[1, j].item()
            degrees[s] += 1
            degrees[d] += 1

    deg_values = [degrees.get(i, 0) for i in range(sg.num_rooms)]
    ax.hist(deg_values, bins=range(0, max(deg_values) + 2), edgecolor='black',
            color='#5DADE2', alpha=0.8)
    ax.axvline(x=2, color='red', linestyle='--', label='Min fire exits (2)')
    ax.set_xlabel('Physical Degree')
    ax.set_ylabel('Room Count')
    ax.set_title('Physical Connection Degree Distribution')
    ax.legend(fontsize=8)

    # 2. Sight degree distribution
    ax = axes[1]
    sight_ei_rr = sg.sight_room_edges
    sight_ei_re = sg.sight_env_edges
    sight_deg = defaultdict(int)
    for ei in [sight_ei_rr, sight_ei_re]:
        if ei.numel() > 0:
            for j in range(ei.shape[1]):
                s = ei[0, j].item()
                sight_deg[s] += 1

    sight_values = [sight_deg.get(i, 0) for i in range(sg.num_rooms)]
    if sight_values:
        ax.hist(sight_values, bins=range(0, max(sight_values) + 2),
                edgecolor='black', color='#2ECC71', alpha=0.8)
    ax.set_xlabel('Sight Line Degree')
    ax.set_ylabel('Room Count')
    ax.set_title('Sight Line Degree Distribution')

    # 3. Room type counts
    ax = axes[2]
    from collections import Counter
    type_names = []
    for i in range(sg.num_rooms):
        type_names.append(_get_room_type_name(sg, i))
    counts = Counter(type_names)
    names = list(ROOM_TYPE_COLORS.keys())
    present_names = [n for n in names if n in counts]
    values = [counts[n] for n in present_names]
    colors = [ROOM_TYPE_COLORS[n] for n in present_names]

    bars = ax.barh(present_names, values, color=colors, edgecolor='#333',
                   linewidth=0.3)
    ax.set_xlabel('Count')
    ax.set_title('Room Type Distribution')

    fig.tight_layout()
    return fig


def save_all_views(
    sg,
    output_dir: str = 'outputs/figures',
    prefix: str = 'school',
    dpi: int = 150,
):
    """
    Generate and save all visualization views for a single graph.

    Creates:
      - {prefix}_full_graph.png
      - {prefix}_floors.png
      - {prefix}_constraints.png
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full graph
    fig1 = draw_full_graph(sg, title=f"School Graph — {prefix}")
    fig1.savefig(out / f'{prefix}_full_graph.png', dpi=dpi,
                 bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig1)

    # Floor subgraphs
    fig2 = draw_floor_subgraphs(sg)
    fig2.savefig(out / f'{prefix}_floors.png', dpi=dpi,
                 bbox_inches='tight')
    plt.close(fig2)

    # Constraint report
    fig3 = draw_constraint_report(sg)
    fig3.savefig(out / f'{prefix}_constraints.png', dpi=dpi,
                 bbox_inches='tight')
    plt.close(fig3)

    print(f"  Saved: {out / f'{prefix}_full_graph.png'}")
    print(f"  Saved: {out / f'{prefix}_floors.png'}")
    print(f"  Saved: {out / f'{prefix}_constraints.png'}")
