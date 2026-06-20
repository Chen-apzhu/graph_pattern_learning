"""
Visualization Components — 可视化组件

Reusable plotting/rendering functions for the Streamlit app:
  - Graph topology view (matplotlib)
  - Orthogonal floor plan view
  - Constraint violation panel
  - Motif dictionary cards
"""

from __future__ import annotations

from typing import List, Dict, Optional
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')

# ── Font configuration: Times New Roman (English) + SimHei/黑体 (Chinese) ──
# Must be set BEFORE importing pyplot
import matplotlib.font_manager as fm

# Find available Chinese fonts on Windows
_cn_fonts = [f.name for f in fm.fontManager.ttflist]
_simhei = 'SimHei' if 'SimHei' in _cn_fonts else None
_yahei = 'Microsoft YaHei' if 'Microsoft YaHei' in _cn_fonts else None
_heiti = _simhei or _yahei or 'sans-serif'

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [_heiti, 'Times New Roman', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Color maps reused from visualization.py
ROOM_TYPE_COLORS = {
    'classroom': '#4C72B0', 'special_classroom': '#55A868',
    'music_room': '#C44E52', 'gymnasium': '#DD8452',
    'library': '#937860', 'office': '#8CA5C8',
    'teacher_office': '#7FB8D0', 'corridor': '#E8E8E8',
    'staircase': '#8172B2', 'toilet': '#A0A0A0',
    'storage': '#D0D0D0', 'cafeteria': '#E8B44F',
    'entrance_hall': '#64B5CD',
}

EDGE_TYPE_COLORS = {
    'physical_connects': '#5DADE2',
    'acoustic_blocks': '#E74C3C',
    'sight_lines': '#2ECC71',
}


def render_topology_graph(
    sg,  # SchoolGraphData
    title: str = "School Building Topology Graph",
    figsize: tuple = (10, 8),
    dpi: int = 100,
) -> plt.Figure:
    """Render the full school graph with node/edge coloring."""
    G = sg.to_networkx()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    node_colors = []
    node_sizes = []
    for node, attrs in G.nodes(data=True):
        ntype = attrs.get('node_type', 'room')
        if ntype == 'room':
            rid = attrs.get('room_id', '')
            matched = False
            for rt, color in ROOM_TYPE_COLORS.items():
                if rid.startswith(rt):
                    node_colors.append(color)
                    matched = True
                    break
            if not matched:
                node_colors.append('#999999')
            node_sizes.append(60)
        else:
            node_colors.append('#FFD700')
            node_sizes.append(150)

    edge_colors = []
    edge_widths = []
    for u, v, attrs in G.edges(data=True):
        etype = attrs.get('edge_type', 'physical_connects')
        edge_colors.append(EDGE_TYPE_COLORS.get(etype, '#999999'))
        edge_widths.append(2.0 if etype == 'acoustic_blocks' else 1.0 if etype == 'physical_connects' else 0.6)

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=40)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.4, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='#333', linewidths=0.2, ax=ax)

    # Legends
    room_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=6, label=n)
        for n, c in ROOM_TYPE_COLORS.items()
    ]
    edge_legend = [
        Line2D([0], [0], color=c, linewidth=2, label=n)
        for n, c in EDGE_TYPE_COLORS.items()
    ]
    ax.legend(handles=room_legend[:6] + edge_legend, fontsize=6,
              loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    return fig


def render_floor_plan(
    ortho_layout,  # OrthoLayout
    figsize: tuple = (14, 7),
    dpi: int = 100,
) -> plt.Figure:
    """Render a single teaching building floor plan with compact layout."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bw, bh = ortho_layout.boundary

    # ── Building outline ──
    outline = mpatches.Rectangle(
        (0, 0), bw, bh,
        facecolor='#F8F9FA', edgecolor='#2C3E50',
        linewidth=2.5, fill=True, zorder=0,
    )
    ax.add_patch(outline)

    # ── Room labels ──
    type_abbr = {
        'classroom': '教室', 'special_classroom': '专用', 'music_room': '音乐',
        'library': '图书馆', 'office': '办公', 'teacher_office': '教师办公',
        'corridor': '走道', 'staircase': '楼梯', 'toilet': '卫生间',
        'storage': '储藏',
    }

    for room in ortho_layout.rooms:
        rect = mpatches.Rectangle(
            (room.x, room.y), room.width, room.height,
            facecolor=room.color, edgecolor='#555', linewidth=0.7,
            alpha=0.90, zorder=2,
        )
        ax.add_patch(rect)

        label = type_abbr.get(room.room_type, room.room_type[:4])
        if label and room.width > 2.0:
            fs = 6 if room.width > 4 else 5
            ax.text(room.x + room.width / 2, room.y + room.height / 2,
                    label, ha='center', va='center',
                    fontsize=fs, color='#222', fontweight='bold', zorder=3)

    # ── Floor separator ──
    if ortho_layout.num_floors > 1:
        fl_h = bh / ortho_layout.num_floors
        for i in range(1, ortho_layout.num_floors):
            ax.axhline(y=i * fl_h, color='#2C3E50', linewidth=1.5,
                      linestyle='--', alpha=0.4)

    # ── Floor labels ──
    tf_names = {0: '首层 (Ground)', 1: '标准层 (Teaching)', 2: '顶层 (Top)'}
    if ortho_layout.num_floors > 1:
        fl_h = bh / ortho_layout.num_floors
        for i in range(ortho_layout.num_floors):
            yc = i * fl_h + fl_h / 2
            ax.text(bw + 2, yc, tf_names.get(i, f'Floor {i}'),
                    fontsize=8, color='#555', va='center')
    else:
        ax.text(bw + 2, bh / 2, '首层', fontsize=8, color='#555', va='center')

    # ── Title ──
    ax.annotate('南 ↑', xy=(bw / 2, bh - 3), fontsize=9,
                color='#C44E52', fontweight='bold', ha='center')

    ax.set_xlim(-1, bw + 15)
    ax.set_ylim(-1, bh + 1)
    ax.set_aspect('equal')
    ax.set_title("教学楼平面图 (Teaching Building Floor Plan)",
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    return fig


def render_constraint_panel(
    validation: dict,
    figsize: tuple = (8, 4),
    dpi: int = 100,
) -> plt.Figure:
    """Render constraint pass/fail status as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    names = list(validation.keys())
    # Handle both tuple (bool, list) and dict {'passed': bool, 'num_violations': int} formats
    passed = [
        v[0] if isinstance(v, tuple) else v.get('passed', False)
        for v in validation.values()
    ]
    violations = [
        len(v[1]) if isinstance(v, tuple) else v.get('num_violations', 0)
        for v in validation.values()
    ]

    colors = ['#2ECC71' if p else '#E74C3C' for p in passed]
    bars = ax.barh(names, violations, color=colors, edgecolor='#333', linewidth=0.5)

    for bar, p, v in zip(bars, passed, violations):
        status = 'PASS' if p else f'FAIL ({v})'
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                status, va='center', fontsize=9,
                color='#27AE60' if p else '#C0392B', fontweight='bold')

    ax.set_xlabel('Violation Count')
    ax.set_title('Constraint Validation', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def render_score_gauge(
    score: float,
    figsize: tuple = (4, 4),
    dpi: int = 100,
) -> plt.Figure:
    """Render a semi-circular gauge for GNN quality score."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': 'polar'})

    # Gauge arc
    theta = np.linspace(np.pi, 0, 100)
    ax.fill_between(theta, 0.8, 1.0, color='#2ECC71', alpha=0.3)  # Good
    ax.fill_between(theta, 0.4, 0.8, color='#F39C12', alpha=0.3)  # OK
    ax.fill_between(theta, 0.0, 0.4, color='#E74C3C', alpha=0.3)  # Bad

    # Score needle
    needle_angle = np.pi * (1 - score)
    ax.plot([needle_angle, needle_angle], [0, 0.95], color='#2C3E50', linewidth=3)

    ax.text(np.pi / 2, 0.3, f'{score:.1%}', ha='center', va='center',
            fontsize=24, fontweight='bold')
    ax.text(np.pi / 2, 0.15, 'Quality Score', ha='center', va='center', fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location('W')

    fig.tight_layout()
    return fig


def render_motif_graph(
    centroid: dict,
    figsize: tuple = (4, 3),
    dpi: int = 80,
) -> plt.Figure:
    """Render a small motif graph from serialized centroid data."""
    G = nx.Graph()
    for n in centroid.get('nodes', []):
        G.add_node(n['id'], room_id=n.get('room_id', ''), room_type=n.get('room_type', ''))
    for e in centroid.get('edges', []):
        G.add_edge(e['source'], e['target'], edge_type=e.get('edge_type', 'physical_connects'))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    node_colors = []
    for node in G.nodes():
        attrs = G.nodes[node]
        rt = attrs.get('room_type', '')
        color = ROOM_TYPE_COLORS.get(rt, '#999999')
        node_colors.append(color)

    edge_colors = []
    for u, v, attrs in G.edges(data=True):
        etype = attrs.get('edge_type', 'physical_connects')
        edge_colors.append(EDGE_TYPE_COLORS.get(etype, '#999999'))

    pos = nx.kamada_kawai_layout(G) if G.number_of_nodes() > 2 else nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120,
                           edgecolors='#333', linewidths=0.5, ax=ax)

    # Labels: first 3 chars of room type
    labels = {n: G.nodes[n].get('room_type', '')[:3] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax)

    ax.set_title(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                 fontsize=8, color='#555')
    ax.axis('off')
    fig.tight_layout(pad=0.3)
    return fig


def render_motif_card(motif) -> str:
    """Render a single motif as HTML for Streamlit."""
    room_items = ''.join(
        f'<span style="background:{ROOM_TYPE_COLORS.get(k,"#ccc")};color:#fff;'
        f'padding:2px 6px;border-radius:3px;margin:2px;font-size:11px;">'
        f'{k}×{v:.0f}</span>'
        for k, v in sorted(motif.room_composition.items(), key=lambda x: -x[1])
        if v >= 0.5
    )

    constraint_tags = ''.join(
        f'<code>{c}</code> ' for c in motif.related_constraints
    )

    return f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:16px;margin:8px 0;
                background:#fafafa;">
        <h4 style="margin:0 0 8px 0;">{motif.motif_id}: {motif.name}</h4>
        <p style="color:#666;font-size:13px;margin:4px 0;">
            频率: {motif.percentage:.1%} ({motif.frequency}次) |
            平均节点: {motif.avg_nodes:.0f}
        </p>
        <div style="margin:8px 0;">{room_items}</div>
        <p style="font-size:12px;color:#555;margin:8px 0;">{motif.description[:200]}...</p>
        <div style="margin:8px 0;">{constraint_tags}</div>
    </div>
    """
