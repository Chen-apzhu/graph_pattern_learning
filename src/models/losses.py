"""
Differentiable Constraint Loss Functions — 可微约束损失函数

Converts building code hard/soft constraints into differentiable PyTorch losses
for integration into the GNN training loop (§4 软约束, Loss Penalty).

Every loss function:
  - Takes node features and edge indices as input
  - Returns a scalar tensor (differentiable)
  - References the specific GB code

Losses:
  - fire_exit_loss: penalize high-occupancy rooms with too few physical connections
  - circulation_loss: penalize corridor area ratio outside [10%, 30%]
  - connectivity_loss: penalize disconnected physical graph
  - daylight_loss: penalize high-daylight rooms without sight_lines
"""

import torch
import torch.nn.functional as F

from utils.constants import DEFAULT_MAX_OCCUPANCY, DEFAULT_MAX_AREA

# Room feature column indices (from feature_engineering.py)
COL_ROOM_TYPE_START = 0
COL_ROOM_TYPE_END = 13
COL_AREA = 13
COL_ASPECT = 14
COL_OCCUPANCY = 15
COL_DAYLIGHT = 16
COL_NOISE_LVL = 17
COL_NOISE_TOL = 18
COL_FLOOR = 19
COL_ZONE_START = 20
COL_ZONE_END = 26
COL_FIRE_EXITS = 26

# Room type index for corridor
CORRIDOR_IDX = 7  # list(RoomType).index(RoomType.CORRIDOR)


def fire_exit_loss(
    room_x: torch.Tensor,
    phys_edge_index: torch.Tensor,
    occupancy_threshold: float = 50.0,
) -> torch.Tensor:
    """
    Fire Safety Loss — GB50016-2014 §5.5

    Formula:
        L_fire = mean_{i in high_occ} ReLU(fire_exits_min_i - degree_phys_i)

    Args:
        room_x: [N, 27] room feature tensor.
        phys_edge_index: [2, E] physical edge index tensor.
        occupancy_threshold: Occupancy threshold for fire exit requirement.

    Returns:
        Scalar loss tensor (differentiable via scatter_add).
    """
    N = room_x.shape[0]
    occupancy = room_x[:, COL_OCCUPANCY] * DEFAULT_MAX_OCCUPANCY  # denormalize
    fire_exits_norm = room_x[:, COL_FIRE_EXITS]
    fire_exits = torch.clamp((fire_exits_norm * 4.0).round(), min=1).float()

    # Compute degree from edge_index
    degree = torch.zeros(N, device=room_x.device)
    if phys_edge_index.numel() > 0:
        ones = torch.ones(phys_edge_index.shape[1], device=room_x.device)
        degree = degree.scatter_add(0, phys_edge_index[0], ones)
        degree = degree.scatter_add(0, phys_edge_index[1], ones)

    high_occ_mask = (occupancy >= occupancy_threshold).float()
    deficit = F.relu(fire_exits - degree)
    loss = (high_occ_mask * deficit).sum() / (high_occ_mask.sum() + 1e-8)

    return loss


def circulation_loss(room_x: torch.Tensor) -> torch.Tensor:
    """
    Circulation Ratio Loss — GB50099-2011 §8.2.3

    Formula:
        L_circ = ReLU(0.10 - ratio) + ReLU(ratio - 0.30)
        where ratio = corridor_area / total_area

    Args:
        room_x: [N, 27] room feature tensor.

    Returns:
        Scalar loss tensor (differentiable via soft room-type selection).
    """
    areas = room_x[:, COL_AREA] * DEFAULT_MAX_AREA  # [N], denormalized
    total_area = areas.sum() + 1e-8

    # Soft corridor indicator: RoomType one-hot at corridor index
    is_corridor_soft = room_x[:, CORRIDOR_IDX]  # [N], in [0, 1]

    # Differentiable corridor area
    corridor_area = (is_corridor_soft * areas).sum()
    ratio = corridor_area / total_area

    loss = F.relu(0.10 - ratio) + F.relu(ratio - 0.30)
    return loss


def daylight_loss(
    room_x: torch.Tensor,
    sight_edge_index_rr: torch.Tensor,
    sight_edge_index_re: torch.Tensor,
) -> torch.Tensor:
    """
    Daylight Compliance Loss — GB50099-2011 §5.1

    Formula:
        L_daylight = mean_{i in high_daylight} ReLU(1 - sight_degree_i)

    Args:
        room_x: [N, 27] room feature tensor.
        sight_edge_index_rr: [2, E_rr] room-to-room sight edge index.
        sight_edge_index_re: [2, E_re] room-to-env sight edge index.

    Returns:
        Scalar loss tensor.
    """
    N = room_x.shape[0]
    daylight_norm = room_x[:, COL_DAYLIGHT]  # [N], normalized to [0,1]
    high_daylight_mask = (daylight_norm >= 0.75).float()  # >= HIGH (3/4)

    # Sight degree from both room-room and room-env edges
    degree = torch.zeros(N, device=room_x.device)
    for ei in [sight_edge_index_rr, sight_edge_index_re]:
        if ei.numel() > 0:
            ones = torch.ones(ei.shape[1], device=room_x.device)
            degree = degree.scatter_add(0, ei[0], ones)

    deficit = F.relu(1.0 - degree)
    loss = (high_daylight_mask * deficit).sum() / (high_daylight_mask.sum() + 1e-8)

    return loss


def connectivity_loss(
    phys_edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Connectivity Loss — GB50016-2014 §5.5.17

    Uses the algebraic connectivity (lambda_2 of normalized Laplacian).
    Maximizing lambda_2 encourages a fully connected graph.

    Formula:
        L_conn = ReLU(epsilon - lambda_2(L_norm))
        where L_norm is the normalized graph Laplacian

    Args:
        phys_edge_index: [2, E] physical edge index.
        num_nodes: Number of room nodes.

    Returns:
        Scalar loss tensor.
    """
    if phys_edge_index.numel() < 2:
        # No edges → high connectivity loss
        return torch.tensor(0.5, device=phys_edge_index.device, requires_grad=True)

    # Build sparse normalized Laplacian
    row, col = phys_edge_index[0], phys_edge_index[1]
    N = num_nodes

    # Add self-loops implicitly (normalized Laplacian absorbs them)
    # Build adjacency as undirected
    data = torch.ones(row.shape[0], device=phys_edge_index.device)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), data, (N, N)
    ).coalesce()

    # Make symmetric
    adj = adj + adj.t()
    adj = adj.coalesce()

    # Compute degrees
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    idx = adj.indices()
    vals = adj.values()
    norm_vals = vals * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    L_norm = torch.sparse_coo_tensor(idx, -norm_vals, (N, N))
    # Add I: L_norm = I - D^{-1/2} A D^{-1/2}
    L_norm = L_norm + torch.sparse_coo_tensor(
        torch.arange(N, device=phys_edge_index.device).unsqueeze(0).repeat(2, 1),
        torch.ones(N, device=phys_edge_index.device),
        (N, N),
    )
    L_norm = L_norm.coalesce()

    # Compute lambda_2 via eigvalsh on dense matrix (N is small: <200)
    L_dense = L_norm.to_dense()
    try:
        eigvals = torch.linalg.eigvalsh(L_dense)
        lambda_2 = eigvals[1] if eigvals.shape[0] > 1 else eigvals[0]
    except Exception:
        lambda_2 = torch.tensor(0.0, device=phys_edge_index.device)

    return F.relu(0.05 - lambda_2)


def compute_constraint_score(validation: dict) -> torch.Tensor:
    """
    Compute a quality score from constraint validation results (legacy).

    Score = proportion of constraints passed (value in [0, 1]).
    If all 6 constraints pass, score = 1.0.

    Args:
        validation: Dict of {constraint_name: {'passed': bool, 'num_violations': int}}

    Returns:
        Scalar quality score tensor in [0, 1].
    """
    if not validation:
        return torch.tensor(0.5)

    total = len(validation)
    passed = sum(1 for v in validation.values() if v.get('passed', False))
    return torch.tensor(passed / max(1, total), dtype=torch.float32)


# Backward compatibility alias
compute_quality_score = compute_constraint_score
