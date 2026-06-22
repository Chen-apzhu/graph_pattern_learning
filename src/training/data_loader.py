"""
School Graph Data Loader — 数据集加载器

Loads .pt files from the Phase 1 dataset, computes quality scores from
constraint metadata, and provides batched HeteroData for training.

Quality score:
    score = 1 - total_violations / max_possible_violations
    where max_possible_violations accounts for room count scaling.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import random

import torch

from models.losses import compute_quality_score


class SchoolDataLoader:
    """
    Loads and batches school building graphs for GNN training.

    Since HeteroData graphs have different sizes (different num_rooms),
    we use batch_size=1 by default (no padding needed).

    Args:
        dataset_dir: Path to dataset root (e.g., 'outputs/dataset_200_new').
        split: 'train', 'val', or 'test'.
        shuffle: Whether to shuffle files each epoch.
    """

    def __init__(
        self,
        dataset_dir: str = 'outputs/dataset_200_new',
        split: str = 'train',
        shuffle: bool = True,
    ):
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.files = sorted(split_dir.glob('*.pt'))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {split_dir}")

        self.split = split
        self.shuffle = shuffle
        self._indices = list(range(len(self.files)))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns (hetero_data, quality_score, metadata_dict).
        """
        pt_file = self.files[self._indices[idx]]
        bundle = torch.load(str(pt_file), weights_only=False)

        hetero_data = bundle['hetero_data']
        metadata = bundle.get('metadata', {})

        # Use quality metrics if available (new), fallback to constraint score
        quality = metadata.get('quality', None)
        if quality is not None and isinstance(quality, dict):
            from metrics.quality_metrics import QualityMetrics
            score = torch.tensor(QualityMetrics.aggregate(quality), dtype=torch.float32)
        elif 'quality_score' in metadata:
            score = torch.tensor(metadata['quality_score'], dtype=torch.float32)
        else:
            # Backward compat: compute from constraint validation
            validation = metadata.get('validation', {})
            score = compute_quality_score(validation)

        return hetero_data, score, metadata

    def on_epoch_end(self):
        """Shuffle indices (call after each epoch)."""
        if self.shuffle:
            random.shuffle(self._indices)

    def iter_all(self):
        """Iterate over all graphs in order (no shuffle)."""
        for i in range(len(self.files)):
            yield self[i]

    def get_scores_stats(self, num_samples: int = None) -> dict:
        """
        Compute score statistics across the dataset split.

        Returns:
            Dict with mean, std, min, max of quality scores.
        """
        files = self.files[:num_samples] if num_samples else self.files
        scores = []
        for pt_file in files:
            bundle = torch.load(str(pt_file), weights_only=False)
            validation = bundle.get('metadata', {}).get('validation', {})
            scores.append(compute_quality_score(validation).item())

        scores_t = torch.tensor(scores)
        return {
            'count': len(scores_t),
            'mean': scores_t.mean().item(),
            'std': scores_t.std().item(),
            'min': scores_t.min().item(),
            'max': scores_t.max().item(),
        }
