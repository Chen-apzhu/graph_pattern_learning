"""
Subgraph Clustering — 子图聚类与模体原型提取

Clusters extracted subgraphs using the WL subtree kernel + spectral clustering
to discover recurring architectural spatial motifs.

Algorithm:
  1. Compute WL kernel matrix K (N×N) for subgraph similarity
  2. Spectral clustering on K to find k clusters
  3. For each cluster, extract the centroid as the "motif prototype"
  4. Build Motif objects with room/edge composition statistics
"""

from __future__ import annotations

from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import math

import numpy as np
import networkx as nx

from explainer.wl_kernel import WLKernel
from explainer.motif import Motif, name_motif, generate_description


class SubgraphClusterer:
    """
    Clusters subgraphs using WL kernel + spectral clustering.

    Args:
        n_clusters: Target number of motif clusters (default 8).
        wl_iterations: WL refinement rounds (default 3).
    """

    def __init__(self, n_clusters: int = 8, wl_iterations: int = 3):
        self.n_clusters = n_clusters
        self.wl = WLKernel(num_iterations=wl_iterations)

    def fit(
        self,
        subgraphs: List[nx.Graph],
        subgraph_metadata: List[dict] = None,
    ) -> List[Motif]:
        """
        Cluster subgraphs and extract motif prototypes.

        Args:
            subgraphs: List of NetworkX subgraph candidates.
            subgraph_metadata: Optional metadata per subgraph (scores, constraints).

        Returns:
            List of Motif objects, one per cluster.
        """
        if len(subgraphs) < 2:
            return []

        n = len(subgraphs)
        n_clusters = min(self.n_clusters, n)

        # Step 1: WL kernel matrix
        K = self.wl.compute_kernel_matrix(subgraphs)

        # Step 2: Spectral clustering
        labels = self._spectral_cluster(K, n_clusters)

        # Step 3: Build motif per cluster
        motifs = []
        for cluster_id in range(n_clusters):
            indices = [i for i, l in enumerate(labels) if l == cluster_id]
            if len(indices) < 2:
                continue

            cluster_graphs = [subgraphs[i] for i in indices]
            motif = self._build_motif(cluster_id, cluster_graphs, indices, n)
            motifs.append(motif)

        # Sort by frequency
        motifs.sort(key=lambda m: -m.frequency)
        for i, m in enumerate(motifs):
            m.motif_id = f'MOTIF_{i+1:02d}'

        return motifs

    def _spectral_cluster(
        self,
        K: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """
        Spectral clustering on the kernel (similarity) matrix.

        Uses the normalized Laplacian of K and k-means on eigenvectors.
        """
        n = K.shape[0]

        # Degree matrix
        D = np.diag(np.sum(K, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))

        # Normalized Laplacian: I - D^{-1/2} K D^{-1/2}
        L_norm = np.eye(n) - D_inv_sqrt @ K @ D_inv_sqrt

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(L_norm)

        # Take k smallest eigenvectors
        X = eigvecs[:, :n_clusters]

        # Normalize rows
        row_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / row_norms

        # Simple k-means
        labels = self._kmeans(X, n_clusters)

        return labels

    def _kmeans(
        self,
        X: np.ndarray,
        k: int,
        max_iters: int = 50,
    ) -> np.ndarray:
        """Simple k-means implementation."""
        n = X.shape[0]
        # Initialize centroids randomly
        rng = np.random.default_rng(42)
        centroids = X[rng.choice(n, k, replace=False)]

        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.zeros((n, k))
            for j in range(k):
                distances[:, j] = np.sum((X - centroids[j]) ** 2, axis=1)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels

    def _build_motif(
        self,
        cluster_id: int,
        graphs: List[nx.Graph],
        indices: List[int],
        total_n: int,
    ) -> Motif:
        """Build a Motif object from a cluster of subgraphs."""
        # Room composition
        room_counts: Dict[str, List[int]] = defaultdict(list)
        for g in graphs:
            room_types_in_g = Counter()
            for node in g.nodes():
                rid = g.nodes[node].get('room_id', str(node))
                rt = rid.split('_')[0] if '_' in rid else 'unknown'
                room_types_in_g[rt] += 1
            for rt, c in room_types_in_g.items():
                room_counts[rt].append(c)

        room_comp = {
            rt: np.mean(counts) for rt, counts in room_counts.items()
        }

        # Edge composition
        edge_counts: Dict[str, List[int]] = defaultdict(list)
        for g in graphs:
            etypes_in_g = Counter()
            for u, v, attrs in g.edges(data=True):
                etype = attrs.get('edge_type', 'physical_connects')
                etypes_in_g[etype] += 1
            for et, c in etypes_in_g.items():
                edge_counts[et].append(c)

        edge_comp = {
            et: np.mean(counts) for et, counts in edge_counts.items()
        }

        # Centroid: graph with median distance to others
        n_local = len(graphs)
        if n_local > 1:
            K_local = self.wl.compute_kernel_matrix(graphs)
            avg_sim = K_local.sum(axis=1)
            centroid_idx = int(np.argmax(avg_sim))
            centroid_graph = graphs[centroid_idx]
        else:
            centroid_graph = graphs[0]

        # Name and description
        name = name_motif(room_comp)
        motif = Motif(
            motif_id=f'MOTIF_{cluster_id+1:02d}',
            name=name,
            room_composition=room_comp,
            edge_composition=edge_comp,
            frequency=len(graphs),
            percentage=len(graphs) / total_n,
            avg_nodes=float(np.mean([g.number_of_nodes() for g in graphs])),
            centroid_graph=centroid_graph,
            related_constraints=self._infer_constraints(room_comp, edge_comp),
        )
        motif.description = generate_description(motif)

        return motif

    @staticmethod
    def _infer_constraints(
        room_comp: Dict[str, float],
        edge_comp: Dict[str, float],
    ) -> List[str]:
        """Infer which building code constraints are relevant to this motif."""
        constraints = []
        if room_comp.get('classroom', 0) >= 1:
            constraints.append('GB50099-2011 §5.1 采光')
            constraints.append('GB50016-2014 §5.5 消防')
        if room_comp.get('music_room', 0) >= 0.5:
            constraints.append('GB50099-2011 §7.3 声学')
        if edge_comp.get('acoustic_blocks', 0) >= 0.5:
            constraints.append('GB50099-2011 §7.3 声学')
        if room_comp.get('staircase', 0) >= 0.5:
            constraints.append('GB50016-2014 §5.5.17 疏散')
        if room_comp.get('corridor', 0) >= 1:
            constraints.append('GB50099-2011 §8.2.3 走廊比')
        return list(set(constraints))
