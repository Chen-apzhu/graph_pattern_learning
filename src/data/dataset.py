"""
Batch Dataset Generator — 批量合成数据集生成器

Generates a complete dataset of school building graphs with:
  - Configurable count, sizes, floor ranges
  - Automatic train / val / test split
  - Per-graph metadata and constraint validation results
  - Dataset-level summary statistics (JSON)
  - Deterministic seeding for reproducibility

Usage:
    from data.dataset import SchoolDataset

    ds = SchoolDataset(output_dir='outputs/my_dataset')
    ds.generate(
        total_count=200,
        school_sizes={'small': 60, 'medium': 100, 'large': 40},
        num_floors_range=(2, 5),
        split=(0.7, 0.15, 0.15),
        seed=42,
    )
    ds.save_metadata()
    print(ds.summary())
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from collections import Counter, defaultdict

import numpy as np

from utils.enums import RoomType, EdgeCategory
from utils.constants import SCHOOL_SIZES
from data.generator import SchoolBuildingGenerator, GenerationResult
from data.constraints import ConstraintValidator


class SchoolDataset:
    """
    Orchestrates batch generation of school building graphs into a dataset.

    Directory layout after generation:
        outputs/<dataset_name>/
        ├── metadata.json          # Dataset-level stats, splits, config
        ├── train/
        │   ├── school_<id>.pt     # PyG HeteroData files
        │   └── ...
        ├── val/
        │   └── ...
        ├── test/
        │   └── ...
        └── raw/                   # All generated graphs before split
            └── ...

    Each .pt file is a torch.save bundle: {'hetero_data': HeteroData, 'metadata': {...}}
    """

    def __init__(self, output_dir: str = 'outputs/dataset'):
        """
        Args:
            output_dir: Root directory for dataset output.
        """
        self.output_dir = Path(output_dir)
        self.results: List[GenerationResult] = []
        self.split_info: Dict[str, List[int]] = {}
        self.gen_config: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}

    # ========================================================================
    # Main generation method
    # ========================================================================

    def generate(
        self,
        total_count: int = 200,
        school_sizes: Optional[Dict[str, int]] = None,
        num_floors_range: Tuple[int, int] = (2, 5),
        split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        base_seed: int = 42,
        validate: bool = True,
        save_graphs: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a batch dataset of school building graphs.

        Args:
            total_count: Total number of graphs to generate.
            school_sizes: Dict mapping size name → count. If None, balanced
                          across small/medium/large.
            num_floors_range: (min, max) for random floor assignment.
            split: (train_ratio, val_ratio, test_ratio). Must sum to 1.0.
            base_seed: Starting seed (each graph uses base_seed + i).
            validate: Run constraint validation on each graph.
            save_graphs: Save each .pt file to disk.
            verbose: Print progress during generation.

        Returns:
            Dict with metadata about the generation run.
        """
        # --- Validate inputs ---
        train_r, val_r, test_r = split
        if abs(train_r + val_r + test_r - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got sum={train_r+val_r+test_r}")

        min_floors, max_floors = num_floors_range
        if min_floors < 1 or max_floors < min_floors:
            raise ValueError(f"Invalid floor range: {num_floors_range}")

        # --- Resolve school size distribution ---
        if school_sizes is None:
            # Balanced: distribute total_count evenly
            per_size = total_count // 3
            remainder = total_count % 3
            school_sizes = {
                'small': per_size + (1 if remainder > 0 else 0),
                'medium': per_size + (1 if remainder > 1 else 0),
                'large': per_size,
            }
        else:
            total_requested = sum(school_sizes.values())
            if total_requested != total_count:
                if verbose:
                    print(f"[NOTE] school_sizes sum={total_requested} != total_count={total_count}. "
                          f"Using sum={total_requested}.")
                total_count = total_requested

        self.gen_config = {
            'total_count': total_count,
            'school_sizes': school_sizes,
            'num_floors_range': list(num_floors_range),
            'split': list(split),
            'base_seed': base_seed,
            'validate': validate,
            'generated_at': datetime.now().isoformat(),
        }

        # --- Generate graphs ---
        if verbose:
            print(f"Generating {total_count} school building graphs...")
            print(f"  Sizes: {school_sizes}")
            print(f"  Floors: {num_floors_range[0]}-{num_floors_range[1]}")
            print(f"  Split: train={train_r:.0%} val={val_r:.0%} test={test_r:.0%}")
            print(f"  Seed: {base_seed}")

        t_start = time.time()
        self.results = []
        idx = 0
        rng = np.random.default_rng(base_seed)

        for size_name, count in school_sizes.items():
            for _ in range(count):
                seed_i = base_seed + idx
                num_floors = int(rng.integers(min_floors, max_floors + 1))

                try:
                    gen = SchoolBuildingGenerator(seed=seed_i)
                    result = gen.generate(
                        num_floors=num_floors,
                        school_size=size_name,
                        validate=validate,
                    )
                except RuntimeError as e:
                    if verbose:
                        print(f"  [WARN] Graph {idx} (size={size_name}, seed={seed_i}) "
                              f"failed after retries: {e}")
                    continue

                self.results.append(result)
                idx += 1

                if verbose and idx % max(1, total_count // 20) == 0:
                    elapsed = time.time() - t_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (total_count - idx) / rate if rate > 0 else 0
                    print(f"  [{idx}/{total_count}] {idx/total_count:.0%}  "
                          f"| {rate:.1f} graphs/s | ETA: {eta:.0f}s")

        elapsed = time.time() - t_start
        actual_count = len(self.results)

        if verbose:
            print(f"\nGenerated {actual_count}/{total_count} graphs in {elapsed:.1f}s "
                  f"({actual_count/elapsed:.2f} graphs/s)")
            if actual_count < total_count:
                print(f"  ({total_count - actual_count} failed)")

        # --- Split into train/val/test ---
        indices = list(range(actual_count))
        rng.shuffle(indices)

        train_end = int(actual_count * train_r)
        val_end = train_end + int(actual_count * val_r)

        self.split_info = {
            'train': sorted(indices[:train_end]),
            'val': sorted(indices[train_end:val_end]),
            'test': sorted(indices[val_end:]),
        }

        # --- Compute statistics ---
        self.stats = self._compute_stats()

        # --- Save ---
        if save_graphs:
            self._save_all(verbose=verbose)

        return {
            'total_requested': total_count,
            'actual_generated': actual_count,
            'failed': total_count - actual_count,
            'elapsed_seconds': round(elapsed, 1),
            'graphs_per_second': round(actual_count / elapsed, 2) if elapsed > 0 else 0,
            'train_count': len(self.split_info['train']),
            'val_count': len(self.split_info['val']),
            'test_count': len(self.split_info['test']),
        }

    # ========================================================================
    # Statistics
    # ========================================================================

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute dataset-level statistics across all generated graphs."""

        if not self.results:
            return {}

        total_rooms = [len(r.rooms) for r in self.results]
        total_phys = [len(r.edges_by_category.get(EdgeCategory.PHYSICAL_CONNECTS, []))
                      for r in self.results]
        total_acous = [len(r.edges_by_category.get(EdgeCategory.ACOUSTIC_BLOCKS, []))
                       for r in self.results]
        total_sight = [len(r.edges_by_category.get(EdgeCategory.SIGHT_LINES, []))
                       for r in self.results]

        # Room type distribution (averaged)
        room_type_counts: Dict[str, List[int]] = defaultdict(list)
        for r in self.results:
            c = Counter(room.room_type.value for room in r.rooms)
            for rt_name in [rt.value for rt in RoomType]:
                room_type_counts[rt_name].append(c.get(rt_name, 0))

        # Constraint pass rates
        constraint_pass_rates: Dict[str, float] = {}
        if self.results[0].validation_results:
            constraint_names = list(self.results[0].validation_results.keys())
            for cname in constraint_names:
                passed = sum(
                    1 for r in self.results
                    if r.validation_results.get(cname, (False,))[0]
                )
                constraint_pass_rates[cname] = passed / len(self.results)

        # School size distribution
        size_dist = Counter(r.school_size for r in self.results)

        # Floor distribution
        floor_counts: List[int] = []
        for r in self.results:
            floors_in_graph = Counter(room.floor for room in r.rooms)
            floor_counts.append(max(floors_in_graph.keys()) + 1 if floors_in_graph else 0)

        return {
            'num_graphs': len(self.results),
            'rooms_per_graph': {
                'mean': float(np.mean(total_rooms)),
                'std': float(np.std(total_rooms)),
                'min': int(np.min(total_rooms)),
                'max': int(np.max(total_rooms)),
            },
            'edges_per_graph': {
                'physical_mean': float(np.mean(total_phys)),
                'acoustic_mean': float(np.mean(total_acous)),
                'sight_mean': float(np.mean(total_sight)),
            },
            'avg_room_type_counts': {
                rt: float(np.mean(counts))
                for rt, counts in sorted(room_type_counts.items())
            },
            'constraint_pass_rates': constraint_pass_rates,
            'school_size_distribution': dict(size_dist),
            'avg_floors': float(np.mean(floor_counts)),
            'split_counts': {
                split_name: len(indices)
                for split_name, indices in self.split_info.items()
            },
        }

    # ========================================================================
    # Save / Load
    # ========================================================================

    def _save_all(self, verbose: bool = True):
        """Save all graphs to disk with split directories."""
        splits_dirs = {
            'train': self.output_dir / 'train',
            'val': self.output_dir / 'val',
            'test': self.output_dir / 'test',
            'raw': self.output_dir / 'raw',
        }
        for d in splits_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Map index → split name
        idx_to_split: Dict[int, str] = {}
        for split_name, indices in self.split_info.items():
            for idx in indices:
                idx_to_split[idx] = split_name

        if verbose:
            print(f"\nSaving {len(self.results)} graphs...")

        for i, result in enumerate(self.results):
            # Determine split
            split_name = idx_to_split.get(i, 'raw')

            # Build filename
            graph_id = f"{result.school_size}_{i:04d}_f{result.num_floors}_s{result.seed}"
            filename = f"school_{graph_id}.pt"

            # Convert to HeteroData if needed
            hetero_data = result.hetero_data

            # Build metadata bundle
            metadata = {
                'graph_id': graph_id,
                'index': i,
                'split': split_name,
                'school_size': result.school_size,
                'num_floors': result.num_floors,
                'num_rooms': len(result.rooms),
                'num_env_nodes': len(result.env_nodes),
                'seed': result.seed,
                'attempt': result.attempt,
            }

            if result.validation_results:
                metadata['validation'] = {
                    name: {'passed': passed, 'num_violations': len(violations)}
                    for name, (passed, violations) in result.validation_results.items()
                }

            # Compute quality metrics from HeteroData
            try:
                from metrics.quality_metrics import QualityMetrics
                quality = QualityMetrics.compute_all(hetero_data)
                metadata['quality'] = quality
                metadata['quality_score'] = QualityMetrics.aggregate(quality)
            except Exception:
                pass  # Graceful degradation if metrics computation fails

            # Save to both the split dir and raw dir
            import torch
            bundle = {'hetero_data': hetero_data, 'metadata': metadata}
            torch.save(bundle, str(splits_dirs[split_name] / filename))
            torch.save(bundle, str(splits_dirs['raw'] / filename))

        if verbose:
            for split_name, d in splits_dirs.items():
                if split_name != 'raw':
                    count = len(list(d.glob('*.pt')))
                    print(f"  {split_name}: {count} graphs")

    def save_metadata(self):
        """Save dataset metadata.json to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'config': self.gen_config,
            'statistics': self.stats,
            'split_info': {
                split_name: {
                    'count': len(indices),
                    'indices': indices,
                }
                for split_name, indices in self.split_info.items()
            },
            'saved_at': datetime.now().isoformat(),
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return str(metadata_path)

    @classmethod
    def load_metadata(cls, dataset_dir: str) -> Dict[str, Any]:
        """Load metadata.json from a dataset directory."""
        path = Path(dataset_dir) / 'metadata.json'
        if not path.exists():
            raise FileNotFoundError(f"No metadata.json found in {dataset_dir}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ========================================================================
    # Summary
    # ========================================================================

    def summary(self) -> str:
        """Human-readable dataset summary."""
        if not self.stats:
            return "No data generated yet."

        s = self.stats
        lines = [
            "=" * 60,
            "  SCHOOL BUILDING GRAPH DATASET",
            "=" * 60,
            "",
            f"  Graphs:          {s['num_graphs']}",
            f"  Split:           "
            f"train={s['split_counts'].get('train', 0)}, "
            f"val={s['split_counts'].get('val', 0)}, "
            f"test={s['split_counts'].get('test', 0)}",
            "",
            "  ── Rooms ──",
            f"    Mean:   {s['rooms_per_graph']['mean']:.1f}",
            f"    Std:    {s['rooms_per_graph']['std']:.1f}",
            f"    Range:  [{s['rooms_per_graph']['min']}, {s['rooms_per_graph']['max']}]",
            "",
            "  ── Edges (avg per graph) ──",
            f"    Physical: {s['edges_per_graph']['physical_mean']:.1f}",
            f"    Acoustic: {s['edges_per_graph']['acoustic_mean']:.1f}",
            f"    Sight:    {s['edges_per_graph']['sight_mean']:.1f}",
            "",
            "  ── Constraint Pass Rates ──",
        ]

        for cname, rate in s.get('constraint_pass_rates', {}).items():
            status = "✓" if rate >= 0.90 else "✗" if rate < 0.50 else "~"
            lines.append(f"    [{status}] {cname}: {rate:.1%}")

        lines.extend([
            "",
            "  ── Size Distribution ──",
        ])
        for size, count in sorted(s.get('school_size_distribution', {}).items()):
            lines.append(f"    {size}: {count}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
