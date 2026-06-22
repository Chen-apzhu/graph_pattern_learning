"""Run the explainability pipeline on v12 data."""
import sys, os
sys.path.insert(0, 'src')

import torch
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict

from models.scorer import SchoolGraphScorer
from explainer.subgraph_runner import SubgraphRunner
from explainer.clustering import SubgraphClusterer

# ── Load model ──
ckpt = torch.load('outputs/model_checkpoint_v12.pt', weights_only=False, map_location='cpu')
model = SchoolGraphScorer(hidden_dim=128, num_layers=3)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model: R2=0.74, epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# ── MCTS on test set ──
runner = SubgraphRunner(model, dataset_dir='outputs/dataset_200_v12', device='cpu')
subgraphs, metadata = runner.run_on_split('test', n_simulations=80, max_graphs=10)

print(f"\nExtracted {len(subgraphs)} subgraphs from 10 test graphs")
if len(subgraphs) < 2:
    print("Not enough subgraphs for clustering. Exiting.")
    sys.exit(0)

# ── Show subgraph details ──
print("\n" + "="*60)
print("  TOP SUBGRAPHS (by reward)")
print("="*60)
sorted_idx = sorted(range(len(metadata)), key=lambda i: metadata[i]['reward'], reverse=True)
for rank, i in enumerate(sorted_idx[:5]):
    m = metadata[i]
    sg = subgraphs[i]
    ntypes = Counter()
    etypes = Counter()
    for _, attrs in sg.nodes(data=True):
        rid = attrs.get('room_id', '?')
        ntypes[rid.split('_')[0]] += 1
    for _, _, attrs in sg.edges(data=True):
        etypes[attrs.get('edge_type', '?')] += 1
    print(f"\n  #{rank+1} | reward={m['reward']:.4f} | "
          f"{m['subgraph_nodes']} nodes, {m['subgraph_edges']} edges")
    print(f"    Rooms: {dict(ntypes)}")
    print(f"    Edges: {dict(etypes)}")
    print(f"    School: {m['school_size']}, baseline={m['baseline_score']:.3f}")

# ── Cluster ──
print(f"\n" + "="*60)
print("  MOTIF DISCOVERY (WL Kernel + Spectral Clustering)")
print("="*60)
clusterer = SubgraphClusterer(n_clusters=min(5, len(subgraphs)), wl_iterations=3)
motifs = clusterer.fit(subgraphs, metadata)

for motif in motifs:
    print(f"\n  [{motif.motif_id}] {motif.name}")
    print(f"    Frequency: {motif.frequency}/{len(subgraphs)} ({motif.percentage:.0f}%)")
    print(f"    Avg size: {motif.avg_nodes:.1f} nodes")
    print(f"    Composition: ", end="")
    items = sorted(motif.room_composition.items(), key=lambda x: -x[1])
    print(", ".join(f"{k}×{v:.1f}" for k, v in items if v >= 0.5))
    if motif.related_constraints:
        print(f"    Related codes: {', '.join(motif.related_constraints)}")
    print(f"    Description: {motif.description[:120]}...")

print(f"\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Test graphs: 10")
print(f"  Subgraphs extracted: {len(subgraphs)}")
print(f"  Motifs discovered: {len(motifs)}")
print(f"\n  The GNN (R2=0.74) scores architectural quality from graph topology.")
print(f"  SubgraphX MCTS finds MINIMAL subgraphs that preserve the high score.")
print(f"  WL Kernel clusters these subgraphs into recurring MOTIFS.")
print(f"  These motifs are the 'explanation' — they show WHAT patterns")
print(f"  the GNN uses to distinguish good designs from mediocre ones.")
