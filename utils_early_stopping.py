"""
Prototype Margin-based Early Stopping for Exemplar Refinement.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class PrototypeMarginStopper:
    def __init__(self, model, ex_loader, fixed_loader, device, 
                 eps_improve=1e-4, k_patience=3, max_samples=512, min_epochs=5):
        self.model = model
        self.ex_loader = ex_loader
        self.fixed_loader = fixed_loader
        self.device = device
        self.eps_improve = eps_improve
        self.k_patience = k_patience
        self.max_samples = max_samples
        self.min_epochs = min_epochs
        
        # State
        self.best_margin = -float('inf')
        self.no_improve = 0
        self.history = []  # [(epoch, mean_margin), ...]
        self.current_epoch = 0
        
        # Cached fixed samples for margin computation
        self.fixed_features = None
        self.fixed_labels = None
        
    def start(self):
        """Cache fixed samples for consistent margin evaluation."""
        print(f'[EarlyStopping] Caching {self.max_samples} samples from fixed_loader...')
        
        self.model.eval()
        # Collect features from fixed samples
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.fixed_loader:
                inputs = inputs.to(self.device)
                
                # Feature extraction
                if hasattr(self.model, 'feature_extract'):
                    features = self.model.feature_extract(inputs)
                else:
                    # model(x) returns (logits, features)
                    _, features = self.model(inputs)
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                
                # Check if we have enough samples
                total_samples = sum(f.size(0) for f in all_features)
                if total_samples >= self.max_samples:
                    break
        self.fixed_features = torch.cat(all_features, dim=0)[:self.max_samples]
        self.fixed_labels = torch.cat(all_labels, dim=0)[:self.max_samples]
        
        print(f'[EarlyStopping] Cached {self.fixed_features.size(0)} samples')
        self.best_margin = -float('inf')
        self.no_improve = 0
        self.history = []
        self.current_epoch = 0
        
    def step(self, tag='epoch'):
        """
        Evaluate margin and check for early stopping.
        
        Returns dict with 'mean_margin', 'best_margin', 'no_improve', 'stop'.
        """
        if self.fixed_features is None:
            raise RuntimeError("Call start() before step()")
        
        self.current_epoch += 1
        
        prototypes = self._compute_prototypes()
        mean_margin = self._compute_margin(prototypes)
        if self.current_epoch > self.min_epochs:
            improved = mean_margin > self.best_margin + self.eps_improve
            
            if improved:
                self.best_margin = mean_margin
                self.no_improve = 0
            else:
                self.no_improve += 1
        else:
            # Before min_epochs, just update best_margin without checking
            self.best_margin = mean_margin
            improved = True  # Always consider improved before min_epochs
        
        # 4. Check stopping condition (only after min_epochs)
        should_stop = (self.current_epoch > self.min_epochs) and (self.no_improve >= self.k_patience)
        
        # 5. Record history
        self.history.append((tag, mean_margin))
        
        # 6. Print info
        status = '↑' if improved else '→'
        print(f'[EarlyStopping] {tag}: margin={mean_margin:.4f} (best={self.best_margin:.4f}) '
              f'{status} no_improve={self.no_improve}/{self.k_patience}')
        
        if should_stop:
            print(f'[EarlyStopping] Early stopping triggered at {tag}')
        
        return {
            'mean_margin': mean_margin,
            'best_margin': self.best_margin,
            'no_improve': self.no_improve,
            'stop': should_stop
        }
    
    def _compute_prototypes(self):
        """Compute class prototypes as mean of features per class."""
        self.model.eval()
        
        # Collect features by class
        class_features = defaultdict(list)
        
        with torch.no_grad():
            for inputs, labels in self.ex_loader:
                inputs = inputs.to(self.device)
                
                # Feature extraction
                if hasattr(self.model, 'feature_extract'):
                    features = self.model.feature_extract(inputs)
                else:
                    _, features = self.model(inputs)
                
                # Group by class
                for feat, label in zip(features, labels):
                    class_features[label.item()].append(feat.cpu())
        
        # Compute mean for each class
        prototypes = {}
        for class_idx, feats in class_features.items():
            if len(feats) > 0:
                prototypes[class_idx] = torch.stack(feats).mean(dim=0)
        
        return prototypes
    
    def _compute_margin(self, prototypes):
        """Compute mean margin (d2 - d1) for fixed samples."""
        if len(prototypes) < 2:
            return 0.0
        
        # Stack all class prototypes
        proto_classes = sorted(prototypes.keys())
        proto_stack = torch.stack([prototypes[c] for c in proto_classes], dim=0)
        proto_stack = proto_stack.to(self.device)
        
        margins = []
        
        batch_size = 256
        num_samples = self.fixed_features.size(0)
        
        for i in range(0, num_samples, batch_size):
            batch_features = self.fixed_features[i:i+batch_size].to(self.device)
            
            # Compute Euclidean distances to all prototypes
            distances = torch.cdist(batch_features, proto_stack.unsqueeze(0).expand(batch_features.size(0), -1, -1))
            distances = distances.squeeze(0) if distances.dim() == 3 else distances
            
            # For each sample, find d1 (min) and d2 (2nd min)
            sorted_dists, _ = torch.sort(distances, dim=1)
            
            if sorted_dists.size(1) >= 2:
                d1 = sorted_dists[:, 0]  # minimum distance
                d2 = sorted_dists[:, 1]  # second minimum distance
                margin = d2 - d1
                margins.append(margin.cpu())
        
        if len(margins) == 0:
            return 0.0
        
        # Compute mean margin
        all_margins = torch.cat(margins, dim=0)
        mean_margin = all_margins.mean().item()
        
        return mean_margin
