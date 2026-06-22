"""
Quality Audit: 数据集质量真实性 + GNN 学习有效性审计
========================================================
检查三个核心问题：
  1. 质量分数是否全部集中在高分区域？（天花板效应）
  2. GNN 是否学到了什么，还是只预测均值？
  3. 约束通过率是否因为生成器太宽松导致无区分度？
"""

import sys, os
sys.path.insert(0, 'src')

import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

# ── Load dataset ──────────────────────────────────────────────────
from training.data_loader import SchoolDataLoader

DATASET = 'outputs/dataset_200_new'

print("=" * 72)
print("  AUDIT 1: 质量分数分布分析")
print("=" * 72)
print()

# Load all scores from all splits
all_scores = {}
all_meta_scores = {}

for split in ['train', 'val', 'test']:
    loader = SchoolDataLoader(DATASET, split, shuffle=False)
    scores = []
    meta_scores = []
    violations_per_graph = []
    pass_counts = []

    for data, score, meta in loader.iter_all():
        scores.append(score.item())
        val = meta.get('validation', {})
        n_violations = sum(v['num_violations'] for v in val.values())
        n_passed = sum(1 for v in val.values() if v['passed'])
        violations_per_graph.append(n_violations)
        pass_counts.append(n_passed)

    scores_t = torch.tensor(scores)
    print(f"  [{split.upper():>5}] n={len(scores):3d}  "
          f"mean={scores_t.mean():.4f}  std={scores_t.std():.4f}  "
          f"min={scores_t.min():.2f}  max={scores_t.max():.2f}  "
          f"median={scores_t.median():.2f}")

    # Distribution of scores
    bins = [0.0, 0.5, 0.67, 0.83, 0.95, 1.0, 1.01]
    hist = torch.histc(scores_t, bins=len(bins)-1, min=0.0, max=1.0)
    print(f"         Score distribution:  ", end="")
    for i in range(len(bins)-1):
        print(f"[{bins[i]:.2f}-{bins[i+1]:.2f}]:{int(hist[i]):4d}  ", end="")
    print()

    # Pass count distribution
    pass_dist = Counter(pass_counts)
    print(f"         Pass count distribution (out of 6):")
    for k in sorted(pass_dist.keys(), reverse=True):
        bar = '█' * pass_dist[k]
        print(f"           {k}/6: {pass_dist[k]:3d} {bar}")

    # Average violations
    viol_t = torch.tensor(violations_per_graph, dtype=torch.float32)
    print(f"         Violations/graph: mean={viol_t.mean():.1f}  std={viol_t.std():.1f}  "
          f"min={viol_t.min():.0f}  max={viol_t.max():.0f}")
    print()

    all_scores[split] = scores_t
    all_meta_scores[split] = {
        'violations': violations_per_graph,
        'pass_counts': pass_counts,
    }

# Overall distribution
all_s = torch.cat(list(all_scores.values()))
print(f"  [ALL]   n={len(all_s):3d}  "
      f"mean={all_s.mean():.4f}  std={all_s.std():.4f}  "
      f"min={all_s.min():.2f}  max={all_s.max():.2f}")
unique_scores = len(set(round(s, 4) for s in all_s.tolist()))
print(f"         Unique score values: {unique_scores} / {len(all_s)}")
print(f"         Score = 1.0: {(all_s == 1.0).sum().item()} graphs ({(all_s == 1.0).float().mean()*100:.1f}%)")
print(f"         Score >= 0.83: {(all_s >= 0.83).sum().item()} graphs ({(all_s >= 0.83).float().mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  AUDIT 2: 逐约束失败详情")
print("=" * 72)
print()

# Load one graph and check individual constraint results
from pathlib import Path
raw_dir = Path(DATASET) / 'raw'
all_graphs = []
for pt_file in sorted(raw_dir.glob('*.pt')):
    bundle = torch.load(str(pt_file), weights_only=False)
    all_graphs.append(bundle['metadata'])

# Per-constraint failure stats
constraint_stats = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total_violations': 0, 'sample_violations': []})
for g in all_graphs:
    val = g.get('validation', {})
    for cname, v in val.items():
        if v['passed']:
            constraint_stats[cname]['passed'] += 1
        else:
            constraint_stats[cname]['failed'] += 1
        constraint_stats[cname]['total_violations'] += v['num_violations']
        if v['num_violations'] > 0 and len(constraint_stats[cname]['sample_violations']) < 3:
            constraint_stats[cname]['sample_violations'].append(
                f"num_violations={v['num_violations']} (details not persisted)"
            )

print(f"  {'Constraint':<22} {'Passed':>7} {'Failed':>7} {'Pass%':>8} {'TotalVio':>9}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")
for cname in ['fire_exits', 'daylight', 'acoustic', 'connectivity', 'area_bounds', 'circulation_ratio']:
    cs = constraint_stats[cname]
    n = cs['passed'] + cs['failed']
    pct = cs['passed'] / n * 100 if n > 0 else 0
    print(f"  {cname:<22} {cs['passed']:>7} {cs['failed']:>7} {pct:>7.1f}% {cs['total_violations']:>9}")

# Show sample violations for constraints that actually fail
print()
for cname in ['fire_exits', 'daylight', 'acoustic', 'connectivity', 'area_bounds', 'circulation_ratio']:
    cs = constraint_stats[cname]
    if cs['failed'] > 0:
        print(f"  --- {cname} sample violations ---")
        for sample in cs['sample_violations']:
            for v in sample[:2]:
                print(f"    {v}")
        print()

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  AUDIT 3: GNN 预测 vs 基线对比")
print("=" * 72)
print()

# Load trained model
from models.scorer import SchoolGraphScorer

ckpt_path = 'outputs/model_checkpoint.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = SchoolGraphScorer(hidden_dim=128, num_layers=3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"  Train loss at checkpoint: {ckpt.get('train_loss', 'N/A')}")
    print()
else:
    print("  No checkpoint found - skipping GNN audit")
    print()
    model = None

if model is not None:
    # Collect all predictions
    test_loader = SchoolDataLoader(DATASET, 'test', shuffle=False)
    preds = []
    targets = []
    for data, score, meta in test_loader.iter_all():
        with torch.no_grad():
            pred = model(data)
        preds.append(pred.item())
        targets.append(score.item())

    preds_t = torch.tensor(preds)
    targets_t = torch.tensor(targets)

    # ── Baseline 1: Always predict mean ──
    mean_pred = torch.ones_like(targets_t) * targets_t.mean()
    mse_baseline_mean = nn.functional.mse_loss(mean_pred, targets_t).item()

    # ── Baseline 2: Random uniform ──
    torch.manual_seed(42)
    random_pred = torch.rand_like(targets_t)
    mse_baseline_random = nn.functional.mse_loss(random_pred, targets_t).item()

    # ── Baseline 3: Predict from room count only ──
    room_counts = []
    for data, _, _ in test_loader.iter_all():
        room_counts.append(data['room'].num_nodes)
    room_counts_t = torch.tensor(room_counts, dtype=torch.float32)
    # Simple linear regression on room count
    X = torch.stack([torch.ones_like(room_counts_t), room_counts_t], dim=1)
    w = torch.linalg.lstsq(X, targets_t.unsqueeze(1)).solution
    count_preds = (X @ w).squeeze()
    mse_baseline_count = nn.functional.mse_loss(count_preds, targets_t).item()

    # ── Model MSE ──
    mse_model = nn.functional.mse_loss(preds_t, targets_t).item()
    mae_model = (preds_t - targets_t).abs().mean().item()

    print(f"  {'Method':<30} {'MSE':>10} {'MAE':>10} {'R2':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # R2 = 1 - MSE_model / MSE_baseline (baseline = always-mean)
    ss_res = ((targets_t - preds_t) ** 2).sum()
    ss_tot = ((targets_t - targets_t.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    print(f"  {'GNN Model':<30} {mse_model:>10.6f} {mae_model:>10.6f} {r2:>10.4f}")
    print(f"  {'Baseline: always mean':<30} {mse_baseline_mean:>10.6f} {'-':>10} {'0.0':>10}")
    print(f"  {'Baseline: random':<30} {mse_baseline_random:>10.6f} {'-':>10} {'-':>10}")
    print(f"  {'Baseline: room count LR':<30} {mse_baseline_count:>10.6f} {'-':>10} {'-':>10}")
    print()

    # Model advantage over baselines
    print(f"  Model vs always-mean: {((mse_baseline_mean - mse_model) / mse_baseline_mean * 100):.1f}% improvement")
    print(f"  Model vs room count:  {((mse_baseline_count - mse_model) / mse_baseline_count * 100):.1f}% improvement")
    print(f"  R2 (coeff. of determination): {r2:.4f}")
    print()

    # ── Distribution comparison ──
    print(f"  Prediction stats:  mean={preds_t.mean():.4f}  std={preds_t.std():.4f}  "
          f"min={preds_t.min():.4f}  max={preds_t.max():.4f}")
    print(f"  Target stats:      mean={targets_t.mean():.4f}  std={targets_t.std():.4f}  "
          f"min={targets_t.min():.4f}  max={targets_t.max():.4f}")
    print()

    # Correlation
    corr = torch.corrcoef(torch.stack([preds_t, targets_t]))[0, 1].item()
    print(f"  Pearson correlation (pred vs target): {corr:.4f}")
    print()

    # ── Per-sample analysis ──
    errors = (preds_t - targets_t).abs()
    print(f"  Error distribution:")
    print(f"    |error| < 0.05:  {(errors < 0.05).sum().item():>4d} graphs")
    print(f"    |error| < 0.10:  {(errors < 0.10).sum().item():>4d} graphs")
    print(f"    |error| < 0.15:  {(errors < 0.15).sum().item():>4d} graphs")
    print(f"    |error| >= 0.20: {(errors >= 0.20).sum().item():>4d} graphs")

    # ── What is the model actually predicting? ──
    pred_std = preds_t.std().item()
    target_std = targets_t.std().item()
    print()
    print(f"  Prediction std / Target std = {pred_std:.4f} / {target_std:.4f} = {pred_std / target_std:.2f}x")
    if pred_std < target_std * 0.3:
        print(f"  WARNING: Model predictions have very low variance -- may be degenerate (mean predictor)!")
    elif pred_std < target_std * 0.6:
        print(f"  WARNING: Model predictions somewhat compressed -- may need more training")
    else:
        print(f"  OK: Model predictions have reasonable variance")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  AUDIT 4: 图结构多样性检查")
print("=" * 72)
print()

# Check if graphs are structurally diverse
room_counts = []
edge_counts_phys = []
edge_counts_acous = []
edge_counts_sight = []
room_type_dists = []

for pt_file in sorted(raw_dir.glob('*.pt')):
    bundle = torch.load(str(pt_file), weights_only=False)
    hd = bundle['hetero_data']
    room_counts.append(hd['room'].num_nodes)
    try:
        edge_counts_phys.append(hd['room', 'physical_connects', 'room'].edge_index.shape[1])
    except (KeyError, AttributeError):
        edge_counts_phys.append(0)
    try:
        edge_counts_acous.append(hd['room', 'acoustic_blocks', 'room'].edge_index.shape[1])
    except (KeyError, AttributeError):
        edge_counts_acous.append(0)
    try:
        edge_counts_sight.append(hd['room', 'sight_lines', 'room'].edge_index.shape[1])
    except (KeyError, AttributeError):
        edge_counts_sight.append(0)

    # Room type distribution
    x = hd['room'].x
    type_onehot = x[:, :13]
    room_type_dists.append(type_onehot.sum(dim=0).tolist())

room_counts_t = torch.tensor(room_counts, dtype=torch.float32)
phys_t = torch.tensor(edge_counts_phys, dtype=torch.float32)
acous_t = torch.tensor(edge_counts_acous, dtype=torch.float32)
sight_t = torch.tensor(edge_counts_sight, dtype=torch.float32)

print(f"  Room count:        mean={room_counts_t.mean():.1f}  std={room_counts_t.std():.1f}  "
      f"min={room_counts_t.min():.0f}  max={room_counts_t.max():.0f}")
print(f"  Physical edges:    mean={phys_t.mean():.1f}  std={phys_t.std():.1f}  "
      f"min={phys_t.min():.0f}  max={phys_t.max():.0f}")
print(f"  Acoustic edges:    mean={acous_t.mean():.1f}  std={acous_t.std():.1f}  "
      f"min={acous_t.min():.0f}  max={acous_t.max():.0f}")
print(f"  Sight edges:       mean={sight_t.mean():.1f}  std={sight_t.std():.1f}  "
      f"min={sight_t.min():.0f}  max={sight_t.max():.0f}")
print()

# Room type diversity (CV of each type count across graphs)
room_type_dists_t = torch.tensor(room_type_dists)
room_type_names = ['classroom','special_cls','music_rm','gym','library',
                   'office','tchr_office','corridor','stair','toilet',
                   'storage','cafeteria','entrance']
for i, name in enumerate(room_type_names):
    col = room_type_dists_t[:, i]
    cv = col.std() / (col.mean() + 1e-8)
    print(f"  {name:>16}: mean={col.mean():5.1f}  std={col.std():5.1f}  CV={cv:.3f}  "
          f"range=[{col.min():.0f}, {col.max():.0f}]")

print()
print("=" * 72)
print("  AUDIT COMPLETE")
print("=" * 72)
