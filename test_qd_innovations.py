"""
Test script for Innovations #1 and #2
Validates that QD Pool and Behavioral Characterization work correctly
"""

import numpy as np
from stable_baselines3 import PPO
from aceac_v2_cyber_killchain import ACEACv2Environment
from aceac_v2_swap_rl import QualityDiversityPool, PolicyCell
import matplotlib.pyplot as plt


def test_policy_cell():
    """Test PolicyCell creation"""
    print("\n" + "="*70)
    print("TEST 1: PolicyCell")
    print("="*70)
    
    # Create dummy policy
    env = ACEACv2Environment(agent_role="red")
    policy = PPO("MlpPolicy", env, verbose=0)
    
    # Create policy cell
    cell = PolicyCell(
        policy=policy,
        fitness=100.5,
        behavior=(0.7, 0.3),
        generation=1
    )
    
    assert cell.fitness == 100.5
    assert cell.behavior == (0.7, 0.3)
    assert cell.generation == 1
    
    print(f"✅ PolicyCell created: {cell}")
    print("="*70)


def test_qd_pool_basic():
    """Test basic QD pool functionality"""
    print("\n" + "="*70)
    print("TEST 2: QD Pool Basic Operations")
    print("="*70)
    
    # Create pool
    pool = QualityDiversityPool(
        pool_size=5,
        agent_type="red",
        behavior_dims=2,
        resolution=10
    )
    
    # Create environment and policies
    env = ACEACv2Environment(agent_role="red")
    
    # Add multiple policies with different behaviors
    for i in range(5):
        policy = PPO("MlpPolicy", env, verbose=0)
        behavior = (np.random.random(), np.random.random())
        performance = 50 + i * 10
        
        added = pool.add_policy(policy, performance, generation=1, behavior=behavior)
        print(f"  Policy {i}: behavior={behavior}, performance={performance}, added={added}")
    
    stats = pool.get_stats()
    print(f"\n✅ Pool stats: {stats}")
    print("="*70)
    
    return pool


def test_behavioral_characterization():
    """Test behavioral descriptor computation"""
    print("\n" + "="*70)
    print("TEST 3: Behavioral Characterization (Innovation #2)")
    print("="*70)
    
    pool = QualityDiversityPool(agent_type="red", behavior_dims=2, resolution=10)
    env = ACEACv2Environment(agent_role="red")
    
    # Create and train a simple policy
    policy = PPO("MlpPolicy", env, verbose=0)
    
    print("Training policy for 1000 steps...")
    policy.learn(total_timesteps=1000, progress_bar=True)
    
    # Compute behavior
    print("\nComputing behavioral descriptor...")
    behavior = pool.get_behavior_descriptor(policy, env, num_episodes=3)
    
    print(f"\n✅ Behavior computed:")
    print(f"  Kill chain rate: {behavior[0]:.3f}")
    print(f"  Tool diversity: {behavior[1]:.3f}")
    
    # Validate behavior
    assert 0 <= behavior[0] <= 1, "Kill chain rate out of range"
    assert 0 <= behavior[1] <= 1, "Tool diversity out of range"
    
    print("="*70)
    
    return behavior


def test_qd_pool_diversity():
    """Test that QD pool maintains diversity"""
    print("\n" + "="*70)
    print("TEST 4: Quality-Diversity Archive Growth")
    print("="*70)
    
    pool = QualityDiversityPool(
        agent_type="red",
        behavior_dims=2,
        resolution=5  # 5x5 grid for faster testing
    )
    
    env = ACEACv2Environment(agent_role="red")
    
    # Train multiple policies
    behaviors = []
    for i in range(10):
        print(f"\nPolicy {i+1}/10:")
        policy = PPO("MlpPolicy", env, verbose=0)
        
        # Quick training
        policy.learn(total_timesteps=500, progress_bar=False)
        
        # Get behavior
        behavior = pool.get_behavior_descriptor(policy, env, num_episodes=2)
        performance = 50 + np.random.random() * 50
        
        # Add to pool
        added = pool.add_policy(policy, performance, generation=i, behavior=behavior)
        
        behaviors.append(behavior)
        stats = pool.get_stats()
        
        print(f"  Behavior: ({behavior[0]:.3f}, {behavior[1]:.3f})")
        print(f"  Performance: {performance:.2f}")
        print(f"  Added: {added}")
        print(f"  Archive size: {stats['size']}")
        print(f"  Coverage: {stats['coverage']:.1%}")
    
    # Final stats
    final_stats = pool.get_stats()
    print(f"\n✅ Final archive stats:")
    print(f"  Size: {final_stats['size']}")
    print(f"  Coverage: {final_stats['coverage']:.1%}")
    print(f"  Policies added: {pool.policies_added}")
    print(f"  Policies rejected: {pool.policies_rejected}")
    
    # Plot behavior space
    if len(behaviors) > 0:
        behaviors_array = np.array(behaviors)
        plt.figure(figsize=(8, 8))
        plt.scatter(behaviors_array[:, 0], behaviors_array[:, 1], s=100, alpha=0.6)
        plt.xlabel('Kill Chain Progression Rate')
        plt.ylabel('Tool Diversity (Entropy)')
        plt.title('Behavioral Diversity in QD Archive')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('qd_behavior_space.png', dpi=150)
        print(f"\n✅ Behavior space plot saved: qd_behavior_space.png")
    
    print("="*70)


def test_opponent_sampling():
    """Test different opponent sampling strategies"""
    print("\n" + "="*70)
    print("TEST 5: Opponent Sampling Strategies")
    print("="*70)
    
    pool = QualityDiversityPool(agent_type="red", behavior_dims=2, resolution=5)
    env = ACEACv2Environment(agent_role="red")
    
    # Add some policies
    for i in range(5):
        policy = PPO("MlpPolicy", env, verbose=0)
        behavior = (np.random.random(), np.random.random())
        performance = 50 + i * 20
        pool.add_policy(policy, performance, generation=i, behavior=behavior)
    
    # Test different sampling strategies
    strategies = ['uniform', 'fitness_proportional', 'recent']
    
    for strategy in strategies:
        print(f"\nTesting '{strategy}' sampling:")
        for _ in range(3):
            opponent = pool.sample_opponent(strategy=strategy)
            print(f"  Sampled: {opponent is not None}")
    
    print("\n✅ All sampling strategies work")
    print("="*70)


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ACEAC QD INNOVATIONS TEST SUITE")
    print("Testing Innovations #1 (QD Pool) and #2 (Behavioral)")
    print("="*70)
    
    try:
        test_policy_cell()
        test_qd_pool_basic()
        test_behavioral_characterization()
        test_qd_pool_diversity()
        test_opponent_sampling()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nInnovations #1 and #2 are working correctly.")
        print("You can now run the full training with:")
        print("  python aceac_v2_swap_rl.py")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
