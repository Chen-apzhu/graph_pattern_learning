"""
Weisfeiler-Lehman Subtree Kernel — WL 子树核

Computes graph similarity using the WL subtree kernel for clustering
subgraph candidates into motif families.

Algorithm (Shervashidze et al., JMLR 2011):
  1. Initialize node labels from room type
  2. For h iterations: compress multi-set of neighbor labels + own label into new label
  3. Kernel = inner product of concatenated label histograms across all iterations

Usage:
    kernel = WLKernel(num_iterations=3)
    K = kernel.compute_kernel_matrix(subgraphs)  # [N, N]
"""

from __future__ import annotations

import hashlib
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

import numpy as np
import networkx as nx


class WLKernel:
    """
    Weisfeiler-Lehman Subtree Kernel for graph similarity.

    Args:
        num_iterations: Number of WL refinement rounds (default 3).
    """

    def __init__(self, num_iterations: int = 3):
        self.num_iterations = num_iterations

    def compute_kernel_matrix(
        self,
        graphs: List[nx.Graph],
    ) -> np.ndarray:
        """
        Compute the N×N WL subtree kernel matrix.

        Args:
            graphs: List of NetworkX graphs (subgraph candidates).

        Returns:
            Kernel matrix of shape [len(graphs), len(graphs)].
        """
        n = len(graphs)
        K = np.zeros((n, n))

        # Compute feature vectors for all graphs
        features = [self._compute_feature_vector(g) for g in graphs]

        # Inner product
        for i in range(n):
            K[i, i] = np.dot(features[i], features[i])
            for j in range(i + 1, n):
                sim = np.dot(features[i], features[j])
                # Normalize: cosine similarity
                norm_i = np.linalg.norm(features[i]) + 1e-8
                norm_j = np.linalg.norm(features[j]) + 1e-8
                sim = sim / (norm_i * norm_j)
                K[i, j] = sim
                K[j, i] = sim

        return K

    def _compute_feature_vector(self, graph: nx.Graph) -> np.ndarray:
        """
        Compute the WL feature vector for a single graph.

        Returns:
            1D numpy array: concatenation of label histograms across iterations.
        """
        if graph.number_of_nodes() == 0:
            return np.array([1.0])

        # Initialize labels from node attributes
        labels = {}
        for node in graph.nodes():
            attrs = graph.nodes[node]
            room_id = attrs.get('room_id', str(node))
            # Extract room type prefix as initial label
            label = room_id.split('_')[0] if '_' in room_id else room_id
            labels[node] = label

        histograms = []

        for iteration in range(self.num_iterations + 1):
            # Build histogram of current labels
            label_counts = Counter(labels.values())
            hist = np.zeros(100)  # Fixed-size histogram

            # Hash labels to bins
            for label_str, count in label_counts.items():
                h = int(hashlib.md5(label_str.encode()).hexdigest()[:8], 16)
                hist[h % 100] += count

            histograms.append(hist)

            if iteration < self.num_iterations:
                # Refine labels
                labels = self._wl_refine(graph, labels)

        # Concatenate all histograms
        return np.concatenate(histograms)

    def _wl_refine(
        self,
        graph: nx.Graph,
        labels: Dict[str, str],
    ) -> Dict[str, str]:
        """
        One round of WL label refinement.

        New label(v) = hash(label(v), sorted([label(u) for u in N(v)]))
        """
        new_labels = {}
        for node in graph.nodes():
            # Get sorted neighbor labels
            neighbor_labels = sorted(
                labels.get(nb, '?') for nb in graph.neighbors(node)
            )
            # Concatenate own label + neighbor multi-set
            combined = labels[node] + '|' + ','.join(neighbor_labels)
            # Hash to new label
            new_label = hashlib.md5(combined.encode()).hexdigest()[:12]
            new_labels[node] = new_label

        return new_labels
