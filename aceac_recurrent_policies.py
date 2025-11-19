"""
Recurrent Policy Architectures for Adaptive Learning
====================================================

Implements LSTM-based policies that can learn temporal patterns and
remember past interactions, enabling more sophisticated strategies.

Key Features:
- LSTM/GRU networks for sequence modeling
- Attention mechanisms for focusing on important past events
- Memory-augmented architectures
- Temporal pattern recognition
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


# ============================================================================
# LSTM FEATURE EXTRACTOR - Processes sequential observations
# ============================================================================

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for processing temporal sequences.

    Captures long-term dependencies in the observation history,
    enabling agents to learn strategies that depend on past events.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2
    ):
        super().__init__(observation_space, features_dim)

        # Get observation dimension
        obs_dim = observation_space.shape[0]

        # LSTM layers for temporal processing
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )

        # Projection to feature dimension
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Hidden state (maintained across steps)
        self.hidden = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process observations through LSTM.

        Args:
            observations: Shape (batch, obs_dim) or (batch, seq_len, obs_dim)

        Returns:
            features: Shape (batch, features_dim)
        """
        batch_size = observations.shape[0]

        # If observations is 2D, add sequence dimension
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)  # (batch, 1, obs_dim)

        # Initialize hidden state if None or batch size changed
        if self.hidden is None or self.hidden[0].shape[1] != batch_size:
            self.hidden = self._init_hidden(batch_size, observations.device)

        # LSTM forward pass
        lstm_out, self.hidden = self.lstm(observations, self.hidden)

        # Detach hidden state to prevent backprop through time across episodes
        self.hidden = tuple(h.detach() for h in self.hidden)

        # Get last timestep output
        features = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)

        # Project to feature dimension
        features = self.projection(features)

        return features

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state"""
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(device)
        return (h0, c0)

    def reset_hidden(self):
        """Reset hidden state (call at episode start)"""
        self.hidden = None


# ============================================================================
# ATTENTION FEATURE EXTRACTOR - Focuses on important observations
# ============================================================================

class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor that learns to focus on
    important parts of the observation.

    Uses self-attention to identify which features are most relevant
    for decision-making in the current context.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]

        # Input embedding
        self.input_embedding = nn.Linear(obs_dim, features_dim)

        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=num_heads,
            dim_feedforward=features_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process observations through attention mechanism.

        Args:
            observations: Shape (batch, obs_dim)

        Returns:
            features: Shape (batch, features_dim)
        """
        # Embed observations
        embedded = self.input_embedding(observations)  # (batch, features_dim)

        # Add sequence dimension for transformer
        embedded = embedded.unsqueeze(1)  # (batch, 1, features_dim)

        # Apply transformer
        attended = self.transformer(embedded)  # (batch, 1, features_dim)

        # Remove sequence dimension
        features = attended.squeeze(1)  # (batch, features_dim)

        # Output projection
        features = self.output_projection(features)

        return features


# ============================================================================
# MEMORY-AUGMENTED NETWORK - External memory for complex strategies
# ============================================================================

class MemoryAugmentedNetwork(nn.Module):
    """
    Neural network with external memory bank.

    Inspired by Neural Turing Machines and Differentiable Neural Computers.
    Agents can read from and write to memory to store and retrieve information.
    """

    def __init__(
        self,
        input_dim: int,
        memory_size: int = 64,
        memory_dim: int = 32,
        output_dim: int = 256
    ):
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Controller network (processes input)
        self.controller = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Memory addressing (where to read/write)
        self.key_generator = nn.Linear(128, memory_dim)  # What to look for
        self.write_head = nn.Linear(128, memory_dim)     # What to write
        self.write_gate = nn.Linear(128, 1)              # Whether to write

        # Output network
        self.output = nn.Sequential(
            nn.Linear(128 + memory_dim, output_dim),
            nn.ReLU()
        )

        # Memory bank (initialized with zeros)
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory read/write.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            output: Output tensor (batch, output_dim)
        """
        batch_size = x.shape[0]

        # Process input through controller
        controller_out = self.controller(x)  # (batch, 128)

        # Generate query key for memory read
        query_key = self.key_generator(controller_out)  # (batch, memory_dim)

        # Compute attention over memory (cosine similarity)
        # Normalize keys and memory
        query_key_norm = query_key / (torch.norm(query_key, dim=1, keepdim=True) + 1e-8)
        memory_norm = self.memory / (torch.norm(self.memory, dim=1, keepdim=True) + 1e-8)

        # Compute similarity scores
        similarity = torch.matmul(query_key_norm, memory_norm.t())  # (batch, memory_size)
        attention = torch.softmax(similarity, dim=1)  # (batch, memory_size)

        # Read from memory (weighted sum)
        read_vector = torch.matmul(attention, self.memory)  # (batch, memory_dim)

        # Combine controller output with read vector
        combined = torch.cat([controller_out, read_vector], dim=1)

        # Generate output
        output = self.output(combined)

        # Memory write (only update during training)
        if self.training:
            write_content = self.write_head(controller_out)  # (batch, memory_dim)
            write_gate = torch.sigmoid(self.write_gate(controller_out))  # (batch, 1)

            # Update memory using attention weights (write where we read)
            # Average across batch for memory update
            avg_attention = attention.mean(dim=0, keepdim=True)  # (1, memory_size)
            avg_write_content = write_content.mean(dim=0, keepdim=True)  # (1, memory_dim)
            avg_write_gate = write_gate.mean()

            # Weighted write to memory
            write_weights = avg_attention.t()  # (memory_size, 1)
            memory_update = write_weights * avg_write_content  # (memory_size, memory_dim)

            self.memory = (1 - avg_write_gate) * self.memory + avg_write_gate * memory_update

        return output

    def reset_memory(self):
        """Reset memory bank to zeros"""
        self.memory.zero_()


# ============================================================================
# HIERARCHICAL FEATURE EXTRACTOR - Multi-timescale learning
# ============================================================================

class HierarchicalFeatureExtractor(BaseFeaturesExtractor):
    """
    Hierarchical architecture that processes observations at multiple timescales.

    - Fast timescale: Immediate reactive behaviors
    - Slow timescale: Strategic long-term patterns

    Enables learning both reactive and strategic behaviors simultaneously.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256
    ):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]

        # Fast pathway (immediate reactions)
        self.fast_pathway = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Slow pathway (strategic patterns)
        self.slow_pathway = nn.LSTM(
            input_size=obs_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.slow_hidden = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process observations through both pathways.

        Args:
            observations: Shape (batch, obs_dim)

        Returns:
            features: Shape (batch, features_dim)
        """
        batch_size = observations.shape[0]

        # Fast pathway (immediate processing)
        fast_features = self.fast_pathway(observations)  # (batch, 128)

        # Slow pathway (temporal processing)
        obs_seq = observations.unsqueeze(1)  # (batch, 1, obs_dim)

        if self.slow_hidden is None or self.slow_hidden[0].shape[1] != batch_size:
            device = observations.device
            h0 = torch.zeros(2, batch_size, 128).to(device)
            c0 = torch.zeros(2, batch_size, 128).to(device)
            self.slow_hidden = (h0, c0)

        slow_out, self.slow_hidden = self.slow_pathway(obs_seq, self.slow_hidden)
        self.slow_hidden = tuple(h.detach() for h in self.slow_hidden)
        slow_features = slow_out[:, -1, :]  # (batch, 128)

        # Combine pathways
        combined = torch.cat([fast_features, slow_features], dim=1)  # (batch, 256)

        # Integration
        features = self.integration(combined)

        return features

    def reset_hidden(self):
        """Reset slow pathway hidden state"""
        self.slow_hidden = None


# ============================================================================
# ADAPTIVE META-LEARNER - Learns how to learn
# ============================================================================

class MetaLearningModule(nn.Module):
    """
    Meta-learning module that adapts the learning process itself.

    Implements a simplified version of Model-Agnostic Meta-Learning (MAML).
    Learns to quickly adapt to new opponents or scenarios.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        meta_lr: float = 0.001
    ):
        super().__init__()

        # Meta-parameters (learned across tasks)
        self.meta_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Adaptation network (fine-tunes per task)
        self.adaptation_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.meta_lr = meta_lr

    def forward(self, x: torch.Tensor, adapt: bool = False) -> torch.Tensor:
        """
        Forward pass through meta-learner.

        Args:
            x: Input tensor
            adapt: Whether to perform adaptation (task-specific fine-tuning)

        Returns:
            features: Processed features
        """
        # Meta-level features (general knowledge)
        meta_features = self.meta_network(x)

        if adapt:
            # Task-specific adaptation
            adapted_features = self.adaptation_network(meta_features)
            return adapted_features
        else:
            return meta_features

    def adapt_to_task(self, task_data: List[torch.Tensor], num_steps: int = 5):
        """
        Adapt to a new task using few-shot learning.

        Args:
            task_data: List of (input, target) pairs from the new task
            num_steps: Number of adaptation steps
        """
        optimizer = torch.optim.SGD(
            self.adaptation_network.parameters(),
            lr=self.meta_lr
        )

        for step in range(num_steps):
            for x, target in task_data:
                # Forward pass
                meta_features = self.meta_network(x)
                adapted = self.adaptation_network(meta_features)

                # Compute loss (task-specific)
                loss = nn.MSELoss()(adapted, target)

                # Adaptation step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_lstm_policy_kwargs(
    lstm_hidden_size: int = 128,
    num_lstm_layers: int = 2,
    features_dim: int = 256
) -> Dict:
    """
    Create policy kwargs for LSTM-based policy.

    Returns:
        policy_kwargs: Dictionary to pass to PPO/other algorithms
    """
    policy_kwargs = {
        'features_extractor_class': LSTMFeatureExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'lstm_hidden_size': lstm_hidden_size,
            'num_lstm_layers': num_lstm_layers
        },
        'net_arch': [256, 256]  # Additional MLP layers after LSTM
    }
    return policy_kwargs


def create_attention_policy_kwargs(
    num_heads: int = 4,
    num_layers: int = 2,
    features_dim: int = 256
) -> Dict:
    """
    Create policy kwargs for attention-based policy.

    Returns:
        policy_kwargs: Dictionary to pass to PPO/other algorithms
    """
    policy_kwargs = {
        'features_extractor_class': AttentionFeatureExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'num_heads': num_heads,
            'num_layers': num_layers
        },
        'net_arch': [256, 128]
    }
    return policy_kwargs


def create_hierarchical_policy_kwargs(features_dim: int = 256) -> Dict:
    """
    Create policy kwargs for hierarchical policy.

    Returns:
        policy_kwargs: Dictionary to pass to PPO/other algorithms
    """
    policy_kwargs = {
        'features_extractor_class': HierarchicalFeatureExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim
        },
        'net_arch': [128, 128]
    }
    return policy_kwargs


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Recurrent Policy Architectures for Adaptive Learning")
    print("="*60)

    # Example: Create observation space
    obs_space = gym.spaces.Box(low=0, high=1, shape=(42,), dtype=np.float32)

    # Test LSTM extractor
    print("\n1. Testing LSTM Feature Extractor...")
    lstm_extractor = LSTMFeatureExtractor(obs_space, features_dim=256)
    test_obs = torch.randn(32, 42)  # Batch of 32 observations
    lstm_features = lstm_extractor(test_obs)
    print(f"   Input shape: {test_obs.shape}")
    print(f"   Output shape: {lstm_features.shape}")
    print(f"   ✓ LSTM extractor working")

    # Test Attention extractor
    print("\n2. Testing Attention Feature Extractor...")
    attention_extractor = AttentionFeatureExtractor(obs_space, features_dim=256)
    attention_features = attention_extractor(test_obs)
    print(f"   Input shape: {test_obs.shape}")
    print(f"   Output shape: {attention_features.shape}")
    print(f"   ✓ Attention extractor working")

    # Test Memory-Augmented Network
    print("\n3. Testing Memory-Augmented Network...")
    memory_net = MemoryAugmentedNetwork(input_dim=42, output_dim=256)
    memory_output = memory_net(test_obs)
    print(f"   Input shape: {test_obs.shape}")
    print(f"   Output shape: {memory_output.shape}")
    print(f"   Memory size: {memory_net.memory.shape}")
    print(f"   ✓ Memory network working")

    # Test Hierarchical extractor
    print("\n4. Testing Hierarchical Feature Extractor...")
    hierarchical_extractor = HierarchicalFeatureExtractor(obs_space, features_dim=256)
    hierarchical_features = hierarchical_extractor(test_obs)
    print(f"   Input shape: {test_obs.shape}")
    print(f"   Output shape: {hierarchical_features.shape}")
    print(f"   ✓ Hierarchical extractor working")

    # Test Meta-Learning Module
    print("\n5. Testing Meta-Learning Module...")
    meta_learner = MetaLearningModule(input_dim=42)
    meta_output = meta_learner(test_obs)
    print(f"   Input shape: {test_obs.shape}")
    print(f"   Output shape: {meta_output.shape}")
    print(f"   ✓ Meta-learner working")

    print("\n" + "="*60)
    print("✅ All recurrent architectures tested successfully!")
    print("   These can be integrated with PPO for adaptive learning")
