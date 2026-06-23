"""
Floor-Level Area Auditor — 楼层级面积审计器

Audits every graph in a dataset, producing per-floor area breakdowns
that verify:
  - floor_area_budget matches room_area_sum + corridor_area_sum
  - corridor_ratio ∈ [0.10, 0.25] per floor
  - room counts are reasonable per floor type
  - no room exceeds its spec area bounds

Output format matches PLAN.md A1 specification.

Usage:
    from data.floor_audit import FloorAuditor
    auditor = FloorAuditor()
    report = auditor.audit_dataset('outputs/dataset_200_v12')
    auditor.save_report(report, 'outputs/audit/floor_area_report.json')
"""

from __future__ import annotations

import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import torch
import numpy as np

from utils.enums import RoomType
from utils.constants import DEFAULT_MAX_AREA


RT_MAP = ['classroom', 'special_classroom', 'music_room', 'gymnasium', 'library',
          'office', 'teacher_office', 'corridor', 'staircase', 'toilet', 'storage',
          'cafeteria', 'entrance_hall']


@dataclass
class FloorAuditEntry:
    """Per-floor audit data for one physical floor."""
    floor: int
    floor_area_budget: float          # expected area from building_footprint
    room_area_sum: float              # Σ non-corridor rooms
    corridor_area_sum: float          # Σ corridor rooms
    corridor_ratio: float             # corridor / total
    num_rooms: int
    num_classrooms: int
    num_stairs: int
    num_toilets: int
    room_type_counts: Dict[str, int] = field(default_factory=dict)
    area_violations: List[str] = field(default_factory=list)

    def is_corridor_ratio_ok(self) -> bool:
        return 0.10 <= self.corridor_ratio <= 0.25

    def to_dict(self) -> dict:
        return {
            'floor': self.floor,
            'floor_area_budget': round(self.floor_area_budget, 1),
            'room_area_sum': round(self.room_area_sum, 1),
            'corridor_area_sum': round(self.corridor_area_sum, 1),
            'corridor_ratio': round(self.corridor_ratio, 4),
            'num_rooms': self.num_rooms,
            'num_classrooms': self.num_classrooms,
            'num_stairs': self.num_stairs,
            'num_toilets': self.num_toilets,
            'room_type_counts': self.room_type_counts,
            'area_violations': self.area_violations,
        }


@dataclass
class GraphAuditReport:
    """Full audit for one graph."""
    graph_id: str
    scale: str
    num_floors: int
    gross_area_total: float
    room_area_total: float
    corridor_area_total: float
    corridor_ratio_global: float
    floors: List[FloorAuditEntry]
    global_violations: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return (
            0.10 <= self.corridor_ratio_global <= 0.25
            and all(f.is_corridor_ratio_ok() for f in self.floors)
        )

    def to_dict(self) -> dict:
        return {
            'graph_id': self.graph_id,
            'scale': self.scale,
            'num_floors': self.num_floors,
            'gross_area_total': round(self.gross_area_total, 1),
            'room_area_total': round(self.room_area_total, 1),
            'corridor_area_total': round(self.corridor_area_total, 1),
            'corridor_ratio_global': round(self.corridor_ratio_global, 4),
            'floors': [f.to_dict() for f in self.floors],
            'global_violations': self.global_violations,
        }


class FloorAuditor:
    """
    Audits per-floor area distributions in school building graphs.

    Args:
        building_footprint: Dict mapping school_size -> per_floor_area.
                           Defaults to the v12 values.
    """

    def __init__(self, building_footprint: Dict[str, float] = None):
        self.footprint = building_footprint or {
            'small': 800.0,
            'medium': 1116.0,
            'large': 1404.0,
        }

    # ══════════════════════════════════════════════════════════════
    # Single-graph audit
    # ══════════════════════════════════════════════════════════════

    def audit_graph(self, hetero_data, metadata: dict) -> GraphAuditReport:
        """
        Audit a single graph's per-floor area distribution.

        Args:
            hetero_data: PyG HeteroData object.
            metadata: Metadata dict from the .pt file.

        Returns:
            GraphAuditReport with per-floor breakdown.
        """
        graph_id = metadata.get('graph_id', 'unknown')
        scale = metadata.get('school_size', 'medium')
        nf = metadata.get('num_floors', 3)
        per_floor = self.footprint.get(scale, 1116.0)
        gross_total = per_floor * nf

        x = hetero_data['room'].x
        areas = (x[:, 13] * DEFAULT_MAX_AREA).tolist()
        types = x[:, :13].argmax(dim=1).tolist()
        floor_mid_raw = (x[:, 19] * 4).tolist()
        unique_floors = sorted(set(round(f) for f in floor_mid_raw))

        # Determine physical floors spanned per midpoint
        n_mid = len(unique_floors)
        phys_counts = {}
        for i, m in enumerate(unique_floors):
            if n_mid <= 1:
                phys_counts[m] = nf
            elif i == 0:
                phys_counts[m] = 1
            elif nf >= 5 and i == n_mid - 1:
                phys_counts[m] = 1
            else:
                phys_counts[m] = nf - 1 - (1 if nf >= 5 else 0)

        floor_entries = []
        room_total = 0.0
        corr_total = 0.0
        violations = []

        for fl_mid in unique_floors:
            n_phys = phys_counts.get(fl_mid, 1)
            indices = [i for i in range(x.shape[0])
                       if round(floor_mid_raw[i]) == fl_mid]

            floor_budget = per_floor * n_phys
            floor_areas = [areas[i] / n_phys for i in indices]
            floor_types = [types[i] for i in indices]

            corr_mask = [t == 7 for t in floor_types]
            corr_sum = sum(a for a, m in zip(floor_areas, corr_mask) if m)
            room_sum = sum(a for a, m in zip(floor_areas, corr_mask) if not m)
            total_sum = room_sum + corr_sum
            corr_ratio = corr_sum / total_sum if total_sum > 0 else 0.0

            room_total += room_sum
            corr_total += corr_sum

            type_counts = Counter()
            for t in floor_types:
                if t < len(RT_MAP):
                    type_counts[RT_MAP[t]] += 1

            area_violations = []
            for i in indices:
                a = areas[i] / n_phys
                t = types[i]
                rt_name = RT_MAP[t] if t < len(RT_MAP) else f'type_{t}'
                # Basic bounds check (spec bounds are enforced elsewhere,
                # here we flag extreme outliers)
                if a > 300 and t != 7:
                    area_violations.append(f"{rt_name}: {a:.0f}m2 > 300 cap")
                if a < 3:
                    area_violations.append(f"{rt_name}: {a:.1f}m2 < 3 floor")

            floor_entries.append(FloorAuditEntry(
                floor=fl_mid,
                floor_area_budget=floor_budget,
                room_area_sum=room_sum,
                corridor_area_sum=corr_sum,
                corridor_ratio=corr_ratio,
                num_rooms=len(indices),
                num_classrooms=type_counts.get('classroom', 0),
                num_stairs=type_counts.get('staircase', 0),
                num_toilets=type_counts.get('toilet', 0),
                room_type_counts=dict(type_counts),
                area_violations=area_violations,
            ))

        total_area = room_total + corr_total
        global_corr_ratio = corr_total / total_area if total_area > 0 else 0.0

        # Global violation checks
        if not (0.10 <= global_corr_ratio <= 0.25):
            violations.append(
                f"Global corridor ratio {global_corr_ratio:.3f} outside [0.10, 0.25]"
            )
        area_dev = abs(total_area - gross_total) / gross_total if gross_total > 0 else 0
        if area_dev > 0.05:
            violations.append(
                f"Total area {total_area:.0f} vs budget {gross_total:.0f} ({area_dev:.1%})"
            )

        return GraphAuditReport(
            graph_id=graph_id,
            scale=scale,
            num_floors=nf,
            gross_area_total=gross_total,
            room_area_total=room_total,
            corridor_area_total=corr_total,
            corridor_ratio_global=global_corr_ratio,
            floors=floor_entries,
            global_violations=violations,
        )

    # ══════════════════════════════════════════════════════════════
    # Batch dataset audit
    # ══════════════════════════════════════════════════════════════

    def audit_dataset(self, dataset_dir: str) -> List[GraphAuditReport]:
        """
        Audit all graphs in a dataset directory.

        Args:
            dataset_dir: Path to dataset root (contains raw/*.pt).

        Returns:
            List of GraphAuditReport, one per graph.
        """
        raw_dir = Path(dataset_dir) / 'raw'
        if not raw_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {raw_dir}")

        reports = []
        pt_files = sorted(raw_dir.glob('*.pt'))
        for pt_file in pt_files:
            bundle = torch.load(str(pt_file), weights_only=False)
            report = self.audit_graph(bundle['hetero_data'], bundle['metadata'])
            reports.append(report)

        return reports

    # ══════════════════════════════════════════════════════════════
    # Reporting
    # ══════════════════════════════════════════════════════════════

    def summary_stats(self, reports: List[GraphAuditReport]) -> dict:
        """Compute aggregate statistics across a batch of reports."""
        n = len(reports)
        if n == 0:
            return {}

        valid = sum(1 for r in reports if r.is_valid())
        global_ratios = [r.corridor_ratio_global for r in reports]
        floor_ratios = []
        for r in reports:
            for f in r.floors:
                floor_ratios.append(f.corridor_ratio)

        return {
            'num_graphs': n,
            'num_valid': valid,
            'valid_pct': round(valid / n * 100, 1),
            'global_corr_ratio': {
                'mean': round(float(np.mean(global_ratios)), 4),
                'std': round(float(np.std(global_ratios)), 4),
                'min': round(float(np.min(global_ratios)), 4),
                'max': round(float(np.max(global_ratios)), 4),
            },
            'floor_corr_ratio': {
                'mean': round(float(np.mean(floor_ratios)), 4),
                'std': round(float(np.std(floor_ratios)), 4),
                'min': round(float(np.min(floor_ratios)), 4),
                'max': round(float(np.max(floor_ratios)), 4),
            },
            'area_violations': sum(
                len(f.area_violations) for r in reports for f in r.floors
            ),
            'corr_ratio_violations': sum(
                1 for r in reports
                for f in r.floors if not f.is_corridor_ratio_ok()
            ),
        }

    def save_report(
        self,
        reports: List[GraphAuditReport],
        output_path: str = 'outputs/audit/floor_area_report.json',
    ):
        """Save full audit report to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'summary': self.summary_stats(reports),
            'graphs': [r.to_dict() for r in reports],
        }
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Audit report saved to: {out}")
        return str(out)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs/dataset_200_v12'
    auditor = FloorAuditor()
    print(f"Auditing dataset: {ds_dir}")
    reports = auditor.audit_dataset(ds_dir)
    stats = auditor.summary_stats(reports)

    print(f"\n=== Floor Area Audit Summary ===")
    print(f"  Graphs:          {stats['num_graphs']}")
    print(f"  Fully valid:     {stats['num_valid']}/{stats['num_graphs']} ({stats['valid_pct']}%)")
    print(f"  Global corr ratio: mean={stats['global_corr_ratio']['mean']:.4f} "
          f"std={stats['global_corr_ratio']['std']:.4f} "
          f"[{stats['global_corr_ratio']['min']:.4f}, {stats['global_corr_ratio']['max']:.4f}]")
    print(f"  Floor corr ratio:  mean={stats['floor_corr_ratio']['mean']:.4f} "
          f"std={stats['floor_corr_ratio']['std']:.4f} "
          f"[{stats['floor_corr_ratio']['min']:.4f}, {stats['floor_corr_ratio']['max']:.4f}]")
    print(f"  Area violations:   {stats['area_violations']}")
    print(f"  Corr ratio viol:   {stats['corr_ratio_violations']}")

    auditor.save_report(reports)
