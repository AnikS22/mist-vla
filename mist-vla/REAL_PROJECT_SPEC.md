# MIST-VLA: Real Project Specification

## What This Actually Is

A collision avoidance system for VLAs using:
1. **Per-dimension risk prediction** (not binary failure)
2. **MuJoCo collision detection** (real physics-based)
3. **Directional risk labels** based on collision geometry
4. **Opposition-based steering** (if moving right is risky → steer left)

## Complete Implementation Plan

### Phase 0: Environment Setup (Week 1)

**Hardware Requirements:**
- Minimum: 1× RTX 4090 (24GB)
- Recommended: 1× A100 (40GB+)
- Disk: ~500GB

**Software Setup:**
```bash
conda create -n mist-vla python=3.10
conda activate mist-vla

# LIBERO benchmark
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# OpenVLA
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Core dependencies
pip install torch torchvision transformers
pip install mujoco robosuite
pip install wandb scikit-learn
```

**Verification:**
```python
# Test LIBERO
from libero.libero import benchmark
task_suite = benchmark.get_benchmark("libero_spatial")

# Test OpenVLA
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
```

---

### Phase 1: Data Collection (Weeks 2-3)

**Goal:** Collect ~2000 rollouts with:
- Hidden states from VLA
- Actions executed
- Collision events with positions
- Per-dimension risk labels

#### Phase 1.1: Hidden State Collection

**File:** `src/data_collection/hooks.py`

```python
class HiddenStateCollector:
    """Collect hidden states from OpenVLA during inference."""

    def __init__(self, model):
        self.hidden_states = {}
        self.hooks = []

        # Register hooks on transformer layers
        for i, layer in enumerate(model.language_model.model.layers):
            hook = layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._save_hidden(idx, out)
            )
            self.hooks.append(hook)

    def _save_hidden(self, layer_idx, output):
        # output[0] is hidden states: [batch, seq_len, hidden_dim]
        self.hidden_states[layer_idx] = output[0].detach().cpu()

    def get_last_layer(self):
        """Get final layer hidden state, pooled over action tokens."""
        h = self.hidden_states[max(self.hidden_states.keys())]
        return h.mean(dim=1)  # [batch, hidden_dim]

    def clear(self):
        self.hidden_states = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

#### Phase 1.2: MuJoCo Collision Detection

**File:** `src/data_collection/collision_detection.py`

```python
import numpy as np

def check_collision(env):
    """
    Check if robot is in collision using MuJoCo contact detection.

    Returns:
        collision (bool): Whether collision occurred
        collision_pos (np.array): 3D position of collision point
    """
    sim = env.sim

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom1 = sim.model.geom_id2name(contact.geom1)
        geom2 = sim.model.geom_id2name(contact.geom2)

        # Robot geom names (adjust based on your robot)
        robot_geoms = ['robot0_link0', 'robot0_link1', 'robot0_link2',
                       'robot0_link3', 'robot0_link4', 'robot0_link5',
                       'robot0_link6', 'robot0_link7']

        # Check if robot geoms are involved
        robot_involved = any(rg in geom1 or rg in geom2 for rg in robot_geoms)

        # Exclude gripper-object grasping contacts
        gripper_grasping = ('gripper' in geom1 or 'gripper' in geom2) and \
                          ('object' in geom1 or 'object' in geom2)

        if robot_involved and not gripper_grasping:
            return True, contact.pos.copy()

    return False, None
```

#### Phase 1.3: Rollout Collection

**File:** `src/data_collection/collect_rollouts.py`

```python
def collect_episode(env, policy, collector, max_steps=200):
    """
    Collect one episode with hidden states and collision labels.

    Returns:
        trajectory: List of timestep dicts with:
            - hidden_state: [hidden_dim]
            - action: [7] (x,y,z,roll,pitch,yaw,grip)
            - ee_pos: [3] end-effector position
            - collision: bool
            - collision_pos: [3] or None
            - timestep: int
    """
    obs = env.reset()
    trajectory = []

    for t in range(max_steps):
        # Get action and hidden states
        collector.clear()
        action = policy.predict(obs)
        hidden = collector.get_last_layer()

        # Get end-effector position
        ee_pos = env.robots[0].get_ee_position()

        # Execute action
        next_obs, reward, done, info = env.step(action)

        # Check for collision
        collision, collision_pos = check_collision(env)

        # Store timestep data
        trajectory.append({
            'hidden_state': hidden.numpy(),
            'action': action,
            'ee_pos': ee_pos,
            'collision': collision,
            'collision_pos': collision_pos,
            'timestep': t,
            'reward': reward,
            'done': done
        })

        if done or collision:
            break

        obs = next_obs

    return trajectory

def collect_dataset(task_suite, policy, n_rollouts=2000, save_dir='data/'):
    """Collect full dataset across all tasks."""
    all_trajectories = []

    for task_id in range(len(task_suite.tasks)):
        env = task_suite.make_env(task_id)
        instruction = task_suite.get_task_instruction(task_id)

        collector = HiddenStateCollector(policy.model)

        for i in range(n_rollouts // len(task_suite.tasks)):
            traj = collect_episode(env, policy, collector)
            traj_data = {
                'trajectory': traj,
                'task_id': task_id,
                'instruction': instruction
            }
            all_trajectories.append(traj_data)

            if (i + 1) % 10 == 0:
                print(f"Task {task_id}, rollout {i+1}/{n_rollouts//len(task_suite.tasks)}")

        collector.remove_hooks()

    # Save
    import pickle
    with open(f"{save_dir}/trajectories.pkl", 'wb') as f:
        pickle.dump(all_trajectories, f)

    return all_trajectories
```

#### Phase 1.4: Per-Dimension Risk Labels

**File:** `src/data_collection/compute_risk_labels.py`

```python
import numpy as np

def compute_risk_labels(trajectory, K=10):
    """
    Compute per-dimension risk labels for each timestep.

    For timesteps within K steps of collision:
        risk_i = max(0, action_i * v_i)
    where v is the unit vector from end-effector to collision point.

    Args:
        trajectory: List of timestep dicts
        K: Lookahead window (steps before collision)

    Returns:
        labels: List of [7] arrays (per-dimension risk)
    """
    labels = []

    # Find collision timestep
    collision_t = None
    collision_pos = None
    for t, step in enumerate(trajectory):
        if step['collision']:
            collision_t = t
            collision_pos = step['collision_pos']
            break

    for t, step in enumerate(trajectory):
        if collision_t is None or t > collision_t:
            # No collision or past collision - all zeros
            labels.append(np.zeros(7))
            continue

        if collision_t - t > K:
            # Too far from collision - all zeros
            labels.append(np.zeros(7))
            continue

        # Within K steps of collision - compute directional risk
        ee_pos = step['ee_pos']
        action = step['action']

        # Direction from end-effector to collision point
        v = collision_pos - ee_pos
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            v = np.zeros(3)
        else:
            v = v / v_norm

        # Per-dimension risk for translational dimensions
        risk = np.zeros(7)
        risk[0] = max(0, action[0] * v[0])  # x
        risk[1] = max(0, action[1] * v[1])  # y
        risk[2] = max(0, action[2] * v[2])  # z

        # Rotational dimensions: simplified (can use Jacobian later)
        # For now, set to 0
        risk[3] = 0  # roll
        risk[4] = 0  # pitch
        risk[5] = 0  # yaw

        # Gripper: 0 unless gripper-specific collision
        risk[6] = 0

        # Normalize to [0, 1]
        if risk.max() > 0:
            risk = risk / (risk.max() + 1e-8)

        labels.append(risk)

    return labels

def add_labels_to_dataset(trajectories_file, output_file):
    """Add risk labels to collected trajectories."""
    import pickle

    with open(trajectories_file, 'rb') as f:
        data = pickle.load(f)

    for traj_data in data:
        traj = traj_data['trajectory']
        labels = compute_risk_labels(traj)

        # Add labels to each timestep
        for step, label in zip(traj, labels):
            step['risk_labels'] = label

    # Save labeled data
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Labeled {len(data)} trajectories")
    return data
```

**Expected Output:**
- `data/trajectories.pkl`: Raw rollouts
- `data/trajectories_labeled.pkl`: With per-dimension risk labels
- ~2000 trajectories total
- Each with collision events and risk labels

---

### Phase 2: Train Risk Predictor (Week 4)

**Goal:** Train MLP to predict per-dimension collision risk from hidden states

#### Phase 2.1: Dataset Class

**File:** `src/training/dataset.py`

```python
import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path

class CollisionRiskDataset(Dataset):
    """Dataset for per-dimension risk prediction."""

    def __init__(self, data_file, split='train', train_ratio=0.8):
        self.samples = []

        # Load trajectories
        with open(data_file, 'rb') as f:
            trajectories = pickle.load(f)

        # Flatten to timesteps
        for traj_data in trajectories:
            for step in traj_data['trajectory']:
                if 'risk_labels' in step:
                    self.samples.append({
                        'hidden': torch.tensor(step['hidden_state'], dtype=torch.float32),
                        'risk': torch.tensor(step['risk_labels'], dtype=torch.float32)
                    })

        # Train/val split
        n = len(self.samples)
        split_idx = int(train_ratio * n)

        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['hidden'], self.samples[idx]['risk']
```

#### Phase 2.2: Risk Predictor Model

**File:** `src/training/risk_predictor.py`

```python
import torch
import torch.nn as nn

class RiskPredictor(nn.Module):
    """
    MLP probe to predict per-dimension collision risk.

    Input: hidden states [hidden_dim]
    Output: risk vector [7] (x, y, z, roll, pitch, yaw, grip)
    """

    def __init__(self, input_dim=4096, hidden_dims=[512, 256], output_dim=7):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch, hidden_dim]
        Returns:
            risk: [batch, 7]
        """
        return self.mlp(x)
```

#### Phase 2.3: Training Script

**File:** `src/training/train_risk_predictor.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

def train_risk_predictor(data_file, epochs=50, batch_size=256, lr=1e-4):
    """Train per-dimension risk predictor."""

    # Create datasets
    train_dataset = CollisionRiskDataset(data_file, split='train')
    val_dataset = CollisionRiskDataset(data_file, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = RiskPredictor(input_dim=4096, hidden_dims=[512, 256], output_dim=7)
    model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    dimension_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grip']

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for hidden, risk in train_loader:
            hidden = hidden.cuda()
            risk = risk.cuda()

            optimizer.zero_grad()
            pred = model(hidden)
            loss = criterion(pred, risk)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for hidden, risk in val_loader:
                hidden = hidden.cuda()
                risk = risk.cuda()

                pred = model(hidden)
                val_loss += criterion(pred, risk).item()

                all_preds.append(pred.cpu())
                all_labels.append(risk.cpu())

        # Compute per-dimension AUC
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print("  Per-dimension AUC:")

        for i, dim in enumerate(dimension_names):
            # Only compute AUC if we have both classes
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                print(f"    {dim}: {auc:.3f}")
            else:
                print(f"    {dim}: N/A (single class)")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/risk_predictor.pt')
            patience_counter = 0
            print("  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("  Early stopping triggered")
                break

    return model

if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)

    model = train_risk_predictor('data/trajectories_labeled.pkl')
    print("\n✓ Training complete!")
```

**Expected Output:**
- `checkpoints/risk_predictor.pt`
- Per-dimension AUC > 0.75 (goal)
- Training curves showing convergence

---

### Phase 3: Extract Steering Vectors (Weeks 4-5)

**Goal:** Find neurons aligned with directional concepts

#### Phase 3.1: Token-Neuron Alignment

**File:** `src/steering/extract_steering_vectors.py`

```python
import torch
from transformers import AutoTokenizer

def get_token_neuron_alignments(model):
    """
    Project FFN neurons onto token embedding space.

    Returns:
        alignments: Dict[layer_idx -> [ffn_dim, vocab_size]]
    """
    # Get token embeddings
    embed = model.get_input_embeddings().weight  # [vocab_size, hidden_dim]

    alignments = {}

    for layer_idx, layer in enumerate(model.language_model.model.layers):
        # Get FFN output projection
        W_out = layer.mlp.down_proj.weight.T  # [ffn_dim, hidden_dim]

        # Project onto embedding space
        # Shape: [ffn_dim, vocab_size]
        neuron_projections = W_out @ embed.T

        alignments[layer_idx] = neuron_projections

    return alignments
```

#### Phase 3.2: Find Concept Neurons

```python
def find_concept_neurons(alignments, tokenizer, concept_tokens, threshold=0.5):
    """
    Find neurons that align with specific directional concepts.

    Args:
        alignments: Dict from get_token_neuron_alignments
        tokenizer: Model tokenizer
        concept_tokens: Dict mapping concepts to token lists
            e.g., {'left': ['left', 'leftward'],
                   'slow': ['slow', 'careful', 'gently']}
        threshold: Alignment threshold

    Returns:
        concept_neurons: Dict[concept -> Dict[layer_idx -> List[neuron_idx]]]
    """
    concept_neurons = {}

    for concept, tokens in concept_tokens.items():
        # Get token IDs
        token_ids = []
        for t in tokens:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                token_ids.append(ids[0])

        concept_neurons[concept] = {}

        for layer_idx, projections in alignments.items():
            # projections: [ffn_dim, vocab_size]
            # Get max alignment with any concept token
            concept_alignments = projections[:, token_ids].max(dim=1).values

            # Find neurons above threshold
            aligned = (concept_alignments > threshold).nonzero().squeeze()
            if aligned.dim() == 0:
                aligned = [aligned.item()] if aligned.numel() > 0 else []
            else:
                aligned = aligned.tolist()

            concept_neurons[concept][layer_idx] = aligned

    return concept_neurons
```

#### Phase 3.3: Compute Steering Vectors

```python
def compute_steering_vectors(model, concept_neurons):
    """
    Compute steering vectors as mean of aligned neuron columns.

    Returns:
        steering_vectors: Dict[concept -> Dict[layer_idx -> Tensor[hidden_dim]]]
    """
    steering_vectors = {}

    for concept, layer_neurons in concept_neurons.items():
        steering_vectors[concept] = {}

        for layer_idx, neuron_indices in layer_neurons.items():
            if len(neuron_indices) == 0:
                continue

            # Get FFN output projection
            layer = model.language_model.model.layers[layer_idx]
            W_out = layer.mlp.down_proj.weight  # [hidden_dim, ffn_dim]

            # Mean of aligned neuron columns
            steering_vec = W_out[:, neuron_indices].mean(dim=1)

            # Normalize
            steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

            steering_vectors[concept][layer_idx] = steering_vec

    return steering_vectors

# Define directional concepts
CONCEPT_TOKENS = {
    'left': ['left', 'leftward'],
    'right': ['right', 'rightward'],
    'up': ['up', 'upward', 'raise', 'lift'],
    'down': ['down', 'downward', 'lower'],
    'forward': ['forward', 'ahead', 'front'],
    'backward': ['back', 'backward', 'behind'],
    'slow': ['slow', 'slowly', 'careful', 'gently']
}

def extract_and_save_steering_vectors(model, tokenizer, save_path='checkpoints/steering_vectors.pt'):
    """Full extraction pipeline."""
    print("Extracting token-neuron alignments...")
    alignments = get_token_neuron_alignments(model)

    print("Finding concept neurons...")
    concept_neurons = find_concept_neurons(alignments, tokenizer, CONCEPT_TOKENS)

    # Print statistics
    for concept, layer_dict in concept_neurons.items():
        total = sum(len(neurons) for neurons in layer_dict.values())
        print(f"  {concept}: {total} neurons across {len(layer_dict)} layers")

    print("Computing steering vectors...")
    steering_vectors = compute_steering_vectors(model, concept_neurons)

    # Save
    torch.save(steering_vectors, save_path)
    print(f"✓ Saved to {save_path}")

    return steering_vectors
```

**Expected Output:**
- `checkpoints/steering_vectors.pt`
- Neurons found for each concept (if not, lower threshold)

---

### Phase 4: Steering Implementation (Week 5)

**Goal:** Inject steering during inference based on predicted risks

#### Phase 4.1: Steering Module

**File:** `src/steering/steering_module.py`

```python
import torch
import torch.nn as nn

class SteeringModule:
    """
    Injects steering vectors into VLA hidden states based on risk predictions.
    """

    def __init__(self, model, steering_vectors, intervention_layers=None):
        """
        Args:
            model: OpenVLA model
            steering_vectors: Dict from Phase 3
            intervention_layers: Which layers to intervene (default: last 4)
        """
        self.model = model
        self.steering_vectors = steering_vectors

        # Default: last 4 layers
        if intervention_layers is None:
            n_layers = len(model.language_model.model.layers)
            self.intervention_layers = list(range(n_layers - 4, n_layers))
        else:
            self.intervention_layers = intervention_layers

        self.active_steering = None
        self.hooks = []

    def _steering_hook(self, module, input, output, layer_idx):
        """Hook to inject steering vector into hidden states."""
        if self.active_steering is None:
            return output

        if layer_idx not in self.intervention_layers:
            return output

        # output[0] is hidden states: [batch, seq_len, hidden_dim]
        hidden = output[0]

        # Add steering vector (broadcast across batch and sequence)
        steering_vec = self.active_steering.get(layer_idx)
        if steering_vec is not None:
            # steering_vec: [hidden_dim]
            # Broadcast to [1, 1, hidden_dim]
            steering_vec = steering_vec.unsqueeze(0).unsqueeze(0)
            hidden = hidden + steering_vec

        return (hidden,) + output[1:]

    def register_hooks(self):
        """Register forward hooks on model layers."""
        for i, layer in enumerate(self.model.language_model.model.layers):
            hook = layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._steering_hook(m, inp, out, idx)
            )
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_steering(self, risk_vector, action, beta=1.0):
        """
        Set active steering based on predicted risks and current action.

        Args:
            risk_vector: [7] tensor of per-dimension risks
            action: [7] tensor of current action
            beta: steering strength multiplier
        """
        self.active_steering = {}

        # Mapping: dimension -> (concept if action positive, concept if action negative)
        OPPOSITION = {
            0: ('left', 'right'),       # x: +x right, -x left
            1: ('backward', 'forward'), # y: +y forward, -y back
            2: ('down', 'up'),          # z: +z up, -z down
        }

        for layer_idx in self.intervention_layers:
            combined = torch.zeros(self.model.config.hidden_size, device=action.device)

            for dim_idx, (neg_concept, pos_concept) in OPPOSITION.items():
                if risk_vector[dim_idx] > 0.5:  # Risk threshold
                    # Choose opposing direction based on action sign
                    if action[dim_idx] > 0:
                        concept = neg_concept  # Moving positive -> steer negative
                    else:
                        concept = pos_concept  # Moving negative -> steer positive

                    # Add steering vector if available
                    if concept in self.steering_vectors:
                        if layer_idx in self.steering_vectors[concept]:
                            vec = self.steering_vectors[concept][layer_idx]
                            combined += risk_vector[dim_idx] * vec.to(action.device)

            self.active_steering[layer_idx] = beta * combined

    def clear_steering(self):
        """Clear active steering."""
        self.active_steering = None
```

**Expected Behavior:**
- If moving right (action[0] > 0) is risky → steer left
- If moving up (action[2] > 0) is risky → steer down
- Steering strength proportional to risk

---

### Phase 5: Evaluation (Weeks 6-7)

**Goal:** Compare MIST against baselines

#### Phase 5.1: Evaluation Harness

**File:** `src/evaluation/evaluate.py`

```python
import torch
import numpy as np
from src.data_collection.collision_detection import check_collision

def evaluate_method(env, policy, risk_predictor, steering_module,
                    hidden_collector, method='mist', n_episodes=50):
    """
    Evaluate a safety method on LIBERO tasks.

    Args:
        method: 'none', 'safe_stop', 'random_steer', 'generic_slow', 'mist'

    Returns:
        results: Dict with collision_rate, success_rate, recovery_rate
    """
    results = {
        'collisions': 0,
        'successes': 0,
        'interventions': 0,
        'recoveries': 0,
    }

    for ep in range(n_episodes):
        obs = env.reset()
        episode_intervened = False

        for t in range(200):  # max_steps
            # Get action and hidden states
            hidden_collector.clear()
            steering_module.clear_steering()

            action = policy.predict(obs)
            hidden = hidden_collector.get_last_layer()

            # Predict risk
            with torch.no_grad():
                risk = risk_predictor(hidden.cuda()).cpu()

            # Apply method
            if method == 'none':
                final_action = action

            elif method == 'safe_stop':
                if risk.max() > 0.5:
                    final_action = None  # Stop
                    episode_intervened = True
                else:
                    final_action = action

            elif method == 'random_steer':
                if risk.max() > 0.5:
                    episode_intervened = True
                    # Random steering
                    random_steering = {
                        layer_idx: torch.randn(4096) * 0.1
                        for layer_idx in steering_module.intervention_layers
                    }
                    steering_module.active_steering = random_steering
                    final_action = policy.predict(obs)
                else:
                    final_action = action

            elif method == 'generic_slow':
                if risk.max() > 0.5:
                    episode_intervened = True
                    # Apply generic "slow" steering
                    if 'slow' in steering_module.steering_vectors:
                        for layer_idx in steering_module.intervention_layers:
                            if layer_idx in steering_module.steering_vectors['slow']:
                                steering_module.active_steering = {
                                    layer_idx: steering_module.steering_vectors['slow'][layer_idx]
                                }
                    final_action = policy.predict(obs)
                else:
                    final_action = action

            elif method == 'mist':
                if risk.max() > 0.5:
                    episode_intervened = True
                    # Opposition-based steering
                    steering_module.set_steering(risk, torch.tensor(action), beta=1.0)
                    final_action = policy.predict(obs)

                    # Check if still risky - emergency stop
                    hidden_collector.clear()
                    _ = policy.predict(obs)  # Dummy forward to get hidden
                    new_hidden = hidden_collector.get_last_layer()
                    new_risk = risk_predictor(new_hidden.cuda()).cpu()
                    if new_risk.max() > 0.7:  # Higher threshold
                        final_action = None
                else:
                    final_action = action

            # Execute action
            if final_action is None:
                break  # Emergency stop

            next_obs, reward, done, info = env.step(final_action)

            # Check collision
            collision, _ = check_collision(env)
            if collision:
                results['collisions'] += 1
                break

            # Check success
            if done and info.get('success', False):
                results['successes'] += 1
                if episode_intervened:
                    results['recoveries'] += 1
                break

            obs = next_obs

        if episode_intervened:
            results['interventions'] += 1

    # Compute rates
    results['collision_rate'] = results['collisions'] / n_episodes
    results['success_rate'] = results['successes'] / n_episodes
    results['recovery_rate'] = (results['recoveries'] / results['interventions']
                                if results['interventions'] > 0 else 0)

    return results
```

#### Phase 5.2: Run Baselines

**File:** `scripts/run_evaluation.py`

```python
from src.evaluation.evaluate import evaluate_method

def run_full_evaluation(task_suite, policy, risk_predictor, steering_module,
                       hidden_collector, n_episodes=50):
    """Run all baseline comparisons."""

    methods = ['none', 'safe_stop', 'random_steer', 'generic_slow', 'mist']
    all_results = {}

    for task_id in range(len(task_suite.tasks)):
        env = task_suite.make_env(task_id)
        instruction = task_suite.get_task_instruction(task_id)

        print(f"\nTask {task_id}: {instruction}")
        print("-" * 60)

        for method in methods:
            print(f"\nEvaluating {method}...")
            results = evaluate_method(
                env, policy, risk_predictor, steering_module,
                hidden_collector, method=method, n_episodes=n_episodes
            )

            all_results[f"task_{task_id}_{method}"] = results

            print(f"  Collision Rate: {results['collision_rate']:.2%}")
            print(f"  Success Rate: {results['success_rate']:.2%}")
            print(f"  Recovery Rate: {results['recovery_rate']:.2%}")

    # Save results
    import json
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results
```

**Expected Output:**
- `results/evaluation_results.json`
- Per-method metrics: collision rate, success rate, recovery rate
- MIST should show lower collision + higher recovery than baselines

---

## Directory Structure

```
mist-vla/
├── src/
│   ├── data_collection/
│   │   ├── hooks.py
│   │   ├── collision_detection.py
│   │   ├── collect_rollouts.py
│   │   └── compute_risk_labels.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── risk_predictor.py
│   │   └── train_risk_predictor.py
│   ├── steering/
│   │   ├── extract_steering_vectors.py
│   │   └── steering_module.py
│   └── evaluation/
│       └── evaluate.py
├── scripts/
│   ├── collect_data.py
│   ├── train_probe.py
│   ├── extract_steering.py
│   └── run_evaluation.py
├── data/
│   ├── trajectories.pkl
│   └── trajectories_labeled.pkl
├── checkpoints/
│   ├── risk_predictor.pt
│   └── steering_vectors.pt
└── results/
    └── evaluation_results.json
```

## Critical Success Metrics

1. **Phase 2**: Per-dimension AUC > 0.75
2. **Phase 3**: Find neurons for each concept (lower threshold if needed)
3. **Phase 5**: MIST shows collision_rate < baselines AND recovery_rate > baselines

## Troubleshooting

**If AUC < 0.7:**
- Try different layers for hidden states
- Collect more data (especially collision cases)
- Simplify to binary prediction as backup

**If steering doesn't work:**
- Increase β (steering strength)
- Try more layers
- Verify steering vectors are non-zero

**If recovery rate similar to generic:**
- Per-dimension may not help much
- Pivot to interpretability angle
- Focus on "why" explanations
