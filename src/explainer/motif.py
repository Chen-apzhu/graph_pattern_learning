"""
Spatial Motif Data Structure — 空间模体数据结构

Represents extracted architectural spatial motifs as structured data
with human-readable descriptions in Chinese.

A motif is a recurring subgraph pattern (room type composition + edge structure)
that appears frequently in high-quality school building graphs.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import networkx as nx


# Chinese display names for room types
ROOM_TYPE_CN = {
    'classroom': '普通教室',
    'special_classroom': '专用教室',
    'music_room': '音乐教室',
    'gymnasium': '体育馆',
    'library': '图书馆',
    'office': '行政办公室',
    'teacher_office': '教师办公室',
    'corridor': '走道',
    'staircase': '楼梯间',
    'toilet': '卫生间',
    'storage': '储藏室',
    'cafeteria': '食堂',
    'entrance_hall': '门厅',
}

EDGE_TYPE_CN = {
    'physical_connects': '物理连通',
    'acoustic_blocks': '声学阻断',
    'sight_lines': '视线/采光',
}


@dataclass
class Motif:
    """
    A discovered spatial motif from a cluster of similar subgraphs.

    Attributes:
        motif_id: Unique identifier (e.g., "MOTIF_01").
        name: Human-readable Chinese name (e.g., "南向双教室+交通核模体").
        room_composition: Dict[room_type_cn, avg_count] across cluster.
        edge_composition: Dict[edge_type_cn, avg_count].
        frequency: How many subgraphs in this cluster.
        percentage: Percentage of all extracted subgraphs.
        avg_nodes: Average number of nodes.
        centroid_graph: Representative NetworkX graph (cluster centroid).
        related_constraints: Which building code constraints this motif relates to.
        description: Full Chinese description paragraph.
    """
    motif_id: str
    name: str
    room_composition: Dict[str, float] = field(default_factory=dict)
    edge_composition: Dict[str, float] = field(default_factory=dict)
    frequency: int = 0
    percentage: float = 0.0
    avg_nodes: float = 0.0
    centroid_graph: Optional[nx.Graph] = None
    related_constraints: List[str] = field(default_factory=list)
    description: str = ""

    def summary(self) -> str:
        """One-line summary."""
        rooms_str = ', '.join(
            f'{ROOM_TYPE_CN.get(k, k)}×{v:.1f}'
            for k, v in sorted(self.room_composition.items(), key=lambda x: -x[1])
            if v >= 0.5
        )
        return (
            f"[{self.motif_id}] {self.name} | "
            f"{rooms_str} | "
            f"频率={self.percentage:.1%} (n={self.frequency})"
        )

    def full_description(self) -> str:
        """Full Chinese description for the pattern language dictionary."""
        lines = [
            f"━━━ {self.motif_id}: {self.name} ━━━",
            f"",
            f"  出现频率: {self.frequency} 次 ({self.percentage:.1%})",
            f"  平均节点数: {self.avg_nodes:.1f}",
            f"",
            f"  房间构成:",
        ]
        for rtype, count in sorted(self.room_composition.items(),
                                    key=lambda x: -x[1]):
            if count >= 0.5:
                cn = ROOM_TYPE_CN.get(rtype, rtype)
                lines.append(f"    {cn}: {count:.1f} 个")

        lines.append(f"")
        lines.append(f"  边构成:")
        for etype, count in sorted(self.edge_composition.items(),
                                    key=lambda x: -x[1]):
            if count >= 0.5:
                cn = EDGE_TYPE_CN.get(etype, etype)
                lines.append(f"    {cn}: {count:.1f} 条")

        if self.related_constraints:
            lines.append(f"")
            lines.append(f"  关联约束: {', '.join(self.related_constraints)}")

        if self.description:
            lines.append(f"")
            lines.append(f"  {self.description}")

        lines.append(f"")
        return '\n'.join(lines)


def name_motif(room_comp: Dict[str, float]) -> str:
    """
    Generate a Chinese name for a motif based on its room composition.

    Examples:
        - "教学翼标准模体" (classroom + corridor + staircase dominant)
        - "动静分区模体" (music_room + acoustic_blocks + classroom)
        - "公共服务核模体" (toilet + staircase + corridor)
        - "入口交通模体" (entrance_hall + corridor + office)
    """
    top_rooms = sorted(room_comp.items(), key=lambda x: -x[1])[:3]
    top_names = [ROOM_TYPE_CN.get(k, k) for k, _ in top_rooms]

    # Check for teaching wing pattern
    has_classroom = room_comp.get('classroom', 0) >= 2
    has_corridor = room_comp.get('corridor', 0) >= 1
    has_stair = room_comp.get('staircase', 0) >= 0.5
    has_music = room_comp.get('music_room', 0) >= 0.5
    has_entrance = room_comp.get('entrance_hall', 0) >= 0.5
    has_toilet = room_comp.get('toilet', 0) >= 1
    has_office = room_comp.get('office', 0) >= 1

    if has_music and has_classroom:
        return '动静分区模体'
    elif has_classroom and has_corridor and has_stair:
        return '教学翼标准模体'
    elif has_classroom and has_corridor:
        return '教室-走廊模体'
    elif has_entrance and has_office:
        return '入口行政模体'
    elif has_toilet and has_stair and has_corridor:
        return '公共服务核模体'
    elif has_stair and has_corridor:
        return '垂直交通核模体'
    elif has_corridor:
        return '走廊脊柱模体'
    else:
        return '复合功能模体'


def generate_description(motif: Motif):
    """Generate the Chinese description paragraph."""
    room_strs = [
        f'{ROOM_TYPE_CN.get(k, k)}{int(v)}间'
        for k, v in sorted(motif.room_composition.items(), key=lambda x: -x[1])
        if v >= 0.5
    ]
    desc = f'该模体由{", ".join(room_strs[:5])}组成，'
    desc += f'平均{len(room_strs)}类房间、{motif.avg_nodes:.0f}个图节点。'

    if '音乐教室' in ' '.join(room_strs):
        desc += '音乐教室与普通教室通过声学阻断边隔离，满足GB50099-2011 §7.3动静分区要求。'
    if '楼梯间' in ' '.join(room_strs):
        desc += '楼梯间作为垂直交通核，连接各标准层走廊，确保GB50016-2014 §5.5消防疏散路径畅通。'
    if '普通教室' in ' '.join(room_strs):
        desc += '普通教室沿南向布置并通过视线边连接日照节点，符合GB50099-2011 §5.1天然采光要求。'

    return desc
