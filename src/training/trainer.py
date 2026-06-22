"""
GNN Training Loop — 训练循环

Trains the SchoolGraphScorer on the Phase 1 dataset using MSE loss
plus constraint-specific differentiable losses.

Usage:
    python -m training.trainer
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.scorer import SchoolGraphScorer
from models.losses import (
    fire_exit_loss,
    circulation_loss,
    daylight_loss,
    connectivity_loss,
)
from training.data_loader import SchoolDataLoader


class Trainer:
    """
    Trains the SchoolGraphScorer model.

    Training uses per-graph (batch_size=1) due to variable graph sizes.
    """

    def __init__(
        self,
        dataset_dir: str = 'outputs/dataset_200_new',
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str = None,
        lambda_fire: float = 0.1,
        lambda_conn: float = 0.05,
        lambda_circ: float = 0.05,
        lambda_daylight: float = 0.05,
    ):
        self.dataset_dir = dataset_dir

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model
        self.model = SchoolGraphScorer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # Loss weights
        self.lambda_fire = lambda_fire
        self.lambda_conn = lambda_conn
        self.lambda_circ = lambda_circ
        self.lambda_daylight = lambda_daylight

        # Data
        self.train_loader = SchoolDataLoader(dataset_dir, 'train')
        self.val_loader = SchoolDataLoader(dataset_dir, 'val', shuffle=False)

        # Tracking
        self.train_losses: list = []
        self.val_losses: list = []
        self.best_val_loss = float('inf')

    def _compute_constraint_losses(
        self,
        data,
        pred_score: torch.Tensor,
    ) -> dict:
        """Compute all constraint-specific losses."""
        room_x = data['room'].x.to(self.device)

        losses = {}

        # Fire exit loss
        try:
            phys_ei = data['room', 'physical_connects', 'room'].edge_index.to(self.device)
        except (KeyError, AttributeError):
            phys_ei = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        losses['fire'] = fire_exit_loss(room_x, phys_ei)

        # Circulation loss
        losses['circ'] = circulation_loss(room_x)

        # Daylight loss
        try:
            sight_rr = data['room', 'sight_lines', 'room'].edge_index.to(self.device)
        except (KeyError, AttributeError):
            sight_rr = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        try:
            sight_re = data['room', 'sight_lines', 'environment'].edge_index.to(self.device)
        except (KeyError, AttributeError):
            sight_re = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        losses['daylight'] = daylight_loss(room_x, sight_rr, sight_re)

        # Connectivity loss
        losses['conn'] = connectivity_loss(phys_ei, data['room'].num_nodes)

        return losses

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_graphs = 0

        for data, target_score, _metadata in self.train_loader.iter_all():
            data = data.to(self.device)
            target_score = target_score.to(self.device)

            # Forward
            pred_score = self.model(data)

            # MSE loss
            loss_mse = nn.functional.mse_loss(pred_score, target_score)

            # Constraint losses
            constr_losses = self._compute_constraint_losses(data, pred_score)

            loss = (
                loss_mse
                + self.lambda_fire * constr_losses['fire']
                + self.lambda_conn * constr_losses['conn']
                + self.lambda_circ * constr_losses['circ']
                + self.lambda_daylight * constr_losses['daylight']
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_graphs += 1

        avg_loss = total_loss / max(1, n_graphs)
        self.train_losses.append(avg_loss)
        self.train_loader.on_epoch_end()
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Validate on val set. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_graphs = 0

        for data, target_score, _metadata in self.val_loader.iter_all():
            data = data.to(self.device)
            target_score = target_score.to(self.device)

            pred_score = self.model(data)
            loss_mse = nn.functional.mse_loss(pred_score, target_score)

            total_loss += loss_mse.item()
            n_graphs += 1

        avg_loss = total_loss / max(1, n_graphs)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(
        self,
        num_epochs: int = 80,
        save_path: str = 'outputs/model_checkpoint.pt',
        print_every: int = 10,
    ):
        """
        Full training loop.

        Args:
            num_epochs: Maximum number of epochs.
            save_path: Where to save the best model.
            print_every: Print progress every N epochs.
        """
        print(f"Training on {self.device}")
        print(f"  Train graphs: {len(self.train_loader)}")
        print(f"  Val graphs:   {len(self.val_loader)}")
        print(f"  Hidden dim:   {self.model.encoder.hidden_dim}")
        print(f"  Layers:       {self.model.encoder.num_layers}")
        print()

        t_start = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, save_path)

            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - t_start
                print(
                    f"  Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"elapsed: {elapsed:.0f}s"
                )

        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed:.0f}s")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Model saved to: {save_path}")

    def evaluate_test(self, test_dir: str = None) -> dict:
        """Evaluate on the test set."""
        if test_dir is None:
            test_loader = SchoolDataLoader(self.dataset_dir, 'test', shuffle=False)
        else:
            test_loader = SchoolDataLoader(test_dir, 'test', shuffle=False)

        self.model.eval()
        preds = []
        targets = []
        all_meta = []

        for data, target, meta in test_loader.iter_all():
            data = data.to(self.device)
            with torch.no_grad():
                pred = self.model(data)
            preds.append(pred.cpu().item())
            targets.append(target.item())
            all_meta.append(meta)

        preds_t = torch.tensor(preds)
        targets_t = torch.tensor(targets)

        mse = nn.functional.mse_loss(preds_t, targets_t).item()
        mae = (preds_t - targets_t).abs().mean().item()

        # AUC: treat score > 0.8 as "high quality"
        high_quality = (targets_t > 0.8).float()
        if high_quality.sum() > 0 and high_quality.sum() < len(high_quality):
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(high_quality.numpy(), preds_t.numpy())
        else:
            auc = 1.0

        # R-squared (coefficient of determination)
        ss_res = ((targets_t - preds_t) ** 2).sum()
        ss_tot = ((targets_t - targets_t.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results = {
            'mse': mse,
            'mae': mae,
            'auc': auc,
            'r2': r2,
            'mean_pred': preds_t.mean().item(),
            'mean_target': targets_t.mean().item(),
            'pred_std': preds_t.std().item(),
            'target_std': targets_t.std().item(),
            'num_graphs': len(preds),
        }

        print(f"\nTest Results ({results['num_graphs']} graphs):")
        print(f"  MSE:        {mse:.4f}")
        print(f"  MAE:        {mae:.4f}")
        print(f"  R2:         {r2:.4f}")
        print(f"  AUC:        {auc:.4f}")
        print(f"  Mean pred:  {preds_t.mean().item():.4f}  (std={preds_t.std().item():.4f})")
        print(f"  Mean target:{targets_t.mean().item():.4f}  (std={targets_t.std().item():.4f})")

        return results


# ==========================================================================
# Main entry point
# ==========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 2: GNN School Graph Scorer Training")
    print("=" * 60)
    print()

    trainer = Trainer(
        dataset_dir='outputs/dataset_200_new',
        hidden_dim=128,
        num_layers=3,
        lr=1e-3,
    )

    trainer.train(num_epochs=80)
    trainer.evaluate_test()
