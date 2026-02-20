#!/usr/bin/env python3
"""Analyze which action dimensions contribute to failure predictions.

Uses gradient-based attribution to identify:
1. Which dimensions the model associates with failure risk
2. Per-step risk scores for course correction

This provides the per-dimension failure attribution without needing 
fabricated labels - derived directly from what the model learned.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json


class FailurePredictor(nn.Module):
    """Same architecture as training script."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, action_dim: int = 7):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        feat_dim = hidden_dim // 2
        
        self.ttf_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.failure_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )
        
        self.action_dim = action_dim
    
    def forward(self, x):
        features = self.encoder(x)
        
        ttf = self.ttf_head(features).squeeze(-1)
        failure_logit = self.failure_head(features).squeeze(-1)
        predicted_action = self.action_head(features)
        
        return {
            'ttf': ttf,
            'failure_logit': failure_logit,
            'predicted_action': predicted_action,
            'features': features
        }


def compute_gradient_attribution(model, hidden_state, device):
    """
    Compute gradient-based attribution for failure prediction.
    
    Returns per-dimension attribution scores indicating which parts of
    the hidden state contribute most to failure prediction.
    """
    model.eval()
    
    hidden = torch.FloatTensor(hidden_state).unsqueeze(0).to(device)
    hidden.requires_grad_(True)
    
    outputs = model(hidden)
    failure_prob = torch.sigmoid(outputs['failure_logit'])
    
    # Compute gradients w.r.t. input
    failure_prob.backward()
    
    # Get gradient magnitude
    grad = hidden.grad.abs().squeeze(0)
    
    return grad.cpu().numpy()


def compute_action_sensitivity(model, hidden_state, device):
    """
    Compute how sensitive each action dimension is to hidden state changes.
    
    Higher sensitivity = model relies more on this dimension for prediction.
    """
    model.eval()
    
    hidden = torch.FloatTensor(hidden_state).unsqueeze(0).to(device)
    hidden.requires_grad_(True)
    
    outputs = model(hidden)
    predicted_action = outputs['predicted_action']
    
    sensitivities = []
    for i in range(predicted_action.shape[1]):
        hidden.grad = None
        predicted_action[0, i].backward(retain_graph=True)
        grad_norm = hidden.grad.norm().item()
        sensitivities.append(grad_norm)
    
    return np.array(sensitivities)


def analyze_trajectory(model, features, actions, device):
    """
    Analyze a full trajectory for risk attribution.
    
    Returns:
    - failure_probs: Per-step failure probability
    - ttf_preds: Per-step TTF predictions
    - action_errors: Per-step action prediction error per dimension
    - gradient_importance: Which hidden dimensions matter most for failure
    """
    model.eval()
    
    hidden = torch.FloatTensor(features).to(device)
    actual_actions = torch.FloatTensor(actions).to(device)
    
    with torch.no_grad():
        outputs = model(hidden)
        
        failure_probs = torch.sigmoid(outputs['failure_logit']).cpu().numpy()
        ttf_preds = outputs['ttf'].cpu().numpy()
        predicted_actions = outputs['predicted_action'].cpu().numpy()
    
    # Action prediction error per dimension
    action_errors = np.abs(predicted_actions - actions)
    
    # Compute gradient importance for a few key steps
    gradient_importance = []
    for t in [0, len(features)//4, len(features)//2, 3*len(features)//4, len(features)-1]:
        if t < len(features):
            grad = compute_gradient_attribution(model, features[t], device)
            gradient_importance.append({
                'step': t,
                'grad_norm': np.linalg.norm(grad),
                'grad_mean': grad.mean(),
                'grad_std': grad.std()
            })
    
    return {
        'failure_probs': failure_probs,
        'ttf_preds': ttf_preds * 200,  # Denormalize
        'action_errors': action_errors,
        'gradient_importance': gradient_importance
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze failure attribution')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--failure-data', type=str, required=True)
    parser.add_argument('--success-data', type=str, required=True)
    parser.add_argument('--output', type=str, default='analysis/failure_attribution.json')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Load model
    print('Loading model...')
    checkpoint = torch.load(args.model, map_location='cpu')
    
    model = FailurePredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        action_dim=checkpoint['action_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Load data
    print('Loading data...')
    with open(args.failure_data, 'rb') as f:
        fail_rollouts = pickle.load(f)
    with open(args.success_data, 'rb') as f:
        succ_rollouts = pickle.load(f)
    
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    # Analyze failure trajectories
    print('\nAnalyzing failure trajectories...')
    failure_analysis = []
    
    for i, rollout in enumerate(fail_rollouts[:10]):  # Analyze first 10
        features = np.array(rollout['features'])
        actions = np.array(rollout['actions'])
        
        results = analyze_trajectory(model, features, actions, args.device)
        
        # Compute per-dimension risk scores
        # Risk = action error weighted by failure probability
        weighted_errors = results['action_errors'] * results['failure_probs'][:, None]
        per_dim_risk = weighted_errors.mean(axis=0)
        
        failure_analysis.append({
            'rollout_id': i,
            'task_id': rollout.get('task_id', -1),
            'trajectory_length': len(features),
            'final_failure_prob': float(results['failure_probs'][-1]),
            'initial_failure_prob': float(results['failure_probs'][0]),
            'per_dim_risk': {dim_names[j]: float(per_dim_risk[j]) for j in range(7)},
            'per_dim_action_error': {dim_names[j]: float(results['action_errors'].mean(axis=0)[j]) for j in range(7)},
            'ttf_pred_final': float(results['ttf_preds'][-1]),
            'failure_prob_progression': results['failure_probs'][::20].tolist()  # Every 20 steps
        })
        
        print(f'  Rollout {i}: Final failure prob={results["failure_probs"][-1]:.4f}')
        print(f'    Per-dim risk: ' + ' '.join([f'{dim_names[j]}={per_dim_risk[j]:.4f}' for j in range(7)]))
    
    # Analyze success trajectories for comparison
    print('\nAnalyzing success trajectories...')
    success_analysis = []
    
    for i, rollout in enumerate(succ_rollouts[:10]):
        features = np.array(rollout['features'])
        actions = np.array(rollout['actions'])
        
        results = analyze_trajectory(model, features, actions, args.device)
        
        weighted_errors = results['action_errors'] * results['failure_probs'][:, None]
        per_dim_risk = weighted_errors.mean(axis=0)
        
        success_analysis.append({
            'rollout_id': i,
            'task_id': rollout.get('task_id', -1),
            'trajectory_length': len(features),
            'final_failure_prob': float(results['failure_probs'][-1]),
            'per_dim_risk': {dim_names[j]: float(per_dim_risk[j]) for j in range(7)},
        })
        
        print(f'  Rollout {i}: Final failure prob={results["failure_probs"][-1]:.4f}')
    
    # Aggregate analysis
    print('\n' + '='*70)
    print('AGGREGATE ANALYSIS')
    print('='*70)
    
    fail_per_dim = np.array([[r['per_dim_risk'][d] for d in dim_names] for r in failure_analysis])
    succ_per_dim = np.array([[r['per_dim_risk'][d] for d in dim_names] for r in success_analysis])
    
    print('\nPer-Dimension Risk Scores (mean):')
    print(f'{"Dimension":<10} {"Failure":>10} {"Success":>10} {"Ratio":>10}')
    print('-' * 45)
    for i, dim in enumerate(dim_names):
        fail_mean = fail_per_dim[:, i].mean()
        succ_mean = succ_per_dim[:, i].mean()
        ratio = fail_mean / (succ_mean + 1e-8)
        print(f'{dim.upper():<10} {fail_mean:>10.4f} {succ_mean:>10.4f} {ratio:>10.2f}x')
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'failure_analysis': failure_analysis,
        'success_analysis': success_analysis,
        'aggregate': {
            'failure_mean_risk': {dim_names[i]: float(fail_per_dim[:, i].mean()) for i in range(7)},
            'success_mean_risk': {dim_names[i]: float(succ_per_dim[:, i].mean()) for i in range(7)},
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n✅ Results saved to {args.output}')


if __name__ == '__main__':
    main()
