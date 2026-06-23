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
        Returns (hetero_data, score_dict, metadata_dict).

        score_dict contains:
          - Per-task scores: daylight_quality, circulation_efficiency, etc.
          - overall_quality: aggregated quality score (for multi-task training)
          - 'score': legacy scalar (backward compat)
        """
        pt_file = self.files[self._indices[idx]]
        bundle = torch.load(str(pt_file), weights_only=False)

        hetero_data = bundle['hetero_data']
        metadata = bundle.get('metadata', {})

        quality = metadata.get('quality', None)
        if quality is not None and isinstance(quality, dict):
            score_dict = {}
            for k, v in quality.items():
                score_dict[k] = torch.tensor(v, dtype=torch.float32)
            score_dict['overall_quality'] = torch.tensor(
                metadata.get('quality_score', 0.5), dtype=torch.float32
            )
            score_dict['score'] = score_dict['overall_quality']  # legacy compat
        elif 'quality_score' in metadata:
            s = torch.tensor(metadata['quality_score'], dtype=torch.float32)
            score_dict = {'overall_quality': s, 'score': s}
        else:
            validation = metadata.get('validation', {})
            s = compute_quality_score(validation)
            score_dict = {'overall_quality': s, 'score': s}

        return hetero_data, score_dict, metadata

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
            metadata = bundle.get('metadata', {})
            quality = metadata.get('quality')
            if quality:
                from metrics.quality_metrics import QualityMetrics
                scores.append(QualityMetrics.aggregate(quality))
            else:
                validation = metadata.get('validation', {})
                scores.append(compute_quality_score(validation).item())

        scores_t = torch.tensor(scores)
        return {
            'count': len(scores_t),
            'mean': scores_t.mean().item(),
            'std': scores_t.std().item(),
            'min': scores_t.min().item(),
            'max': scores_t.max().item(),
        }
