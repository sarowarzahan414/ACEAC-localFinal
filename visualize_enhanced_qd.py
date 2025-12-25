"""
Enhanced Visualization for 3D QD Archive
Shows improvements from A, B, C, D
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle


def plot_3d_behavior_space():
    """Plot 3D behavioral space"""
    
    try:
        with open('logs/aceac_v2_enhanced.json', 'r') as f:
            log = json.load(f)
    except:
        print("Run enhanced training first!")
        return
    
    history = log['history']
    
    # Extract 3D behaviors
    red_behaviors = np.array([h['red_behavior'] for h in history])
    blue_behaviors = np.array([h['blue_behavior'] for h in history])
    generations = list(range(1, len(history) + 1))
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Red behaviors in 3D
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(
        red_behaviors[:, 0],
        red_behaviors[:, 1],
        red_behaviors[:, 2],
        c=generations,
        cmap='Reds',
        s=100,
        alpha=0.7,
        edgecolors='darkred'
    )
    ax1.set_xlabel('Kill Chain Rate', fontsize=10)
    ax1.set_ylabel('Tool Diversity', fontsize=10)
    ax1.set_zlabel('Effectiveness', fontsize=10)
    ax1.set_title('Red Agent 3D Behavioral Space', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Generation')
    
    # Plot 2: Blue behaviors in 3D
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(
        blue_behaviors[:, 0],
        blue_behaviors[:, 1],
        blue_behaviors[:, 2],
        c=generations,
        cmap='Blues',
        s=100,
        alpha=0.7,
        edgecolors='darkblue'
    )
    ax2.set_xlabel('Kill Chain Rate', fontsize=10)
    ax2.set_ylabel('Tool Diversity', fontsize=10)
    ax2.set_zlabel('Effectiveness', fontsize=10)
    ax2.set_title('Blue Agent 3D Behavioral Space', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Generation')
    
    # Plot 3: Both combined
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(
        red_behaviors[:, 0],
        red_behaviors[:, 1],
        red_behaviors[:, 2],
        c='red',
        s=80,
        alpha=0.6,
        label='Red',
        edgecolors='darkred'
    )
    ax3.scatter(
        blue_behaviors[:, 0],
        blue_behaviors[:, 1],
        blue_behaviors[:, 2],
        c='blue',
        s=80,
        alpha=0.6,
        label='Blue',
        edgecolors='darkblue'
    )
    ax3.set_xlabel('Kill Chain Rate', fontsize=10)
    ax3.set_ylabel('Tool Diversity', fontsize=10)
    ax3.set_zlabel('Effectiveness', fontsize=10)
    ax3.set_title('Combined 3D Behavioral Space', fontsize=12, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_3d_behaviors.png', dpi=150)
    print("Saved: enhanced_3d_behaviors.png")
    plt.show()


def plot_enhanced_coverage():
    """Plot coverage growth with enhanced settings"""
    
    try:
        with open('logs/aceac_v2_enhanced.json', 'r') as f:
            log = json.load(f)
    except:
        print("Run enhanced training first!")
        return
    
    history = log['history']
    config = log['config']
    
    generations = [h['generation'] for h in history]
    red_coverage = [h['red_pool']['coverage'] * 100 for h in history]
    blue_coverage = [h['blue_pool']['coverage'] * 100 for h in history]
    red_sizes = [h['red_pool']['size'] for h in history]
    blue_sizes = [h['blue_pool']['size'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Coverage over time
    axes[0, 0].plot(generations, red_coverage, 'r-o', label='Red', linewidth=2, markersize=6)
    axes[0, 0].plot(generations, blue_coverage, 'b-o', label='Blue', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Generation', fontsize=11)
    axes[0, 0].set_ylabel('Coverage (%)', fontsize=11)
    axes[0, 0].set_title(f'Archive Coverage Over {config["num_generations"]} Generations\n' +
                        f'(Grid: {config["grid_resolution"]}^{config["behavior_dims"]} = ' +
                        f'{config["grid_resolution"]**config["behavior_dims"]:,} cells)',
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Archive size
    axes[0, 1].plot(generations, red_sizes, 'r-s', label='Red', linewidth=2, markersize=6)
    axes[0, 1].plot(generations, blue_sizes, 'b-s', label='Blue', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Generation', fontsize=11)
    axes[0, 1].set_ylabel('Number of Policies', fontsize=11)
    axes[0, 1].set_title('Archive Size Growth', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance
    red_perf = [h['red_performance'] for h in history]
    blue_perf = [h['blue_performance'] for h in history]
    axes[1, 0].plot(generations, red_perf, 'r-o', label='Red', linewidth=2, markersize=6)
    axes[1, 0].plot(generations, blue_perf, 'b-o', label='Blue', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Generation', fontsize=11)
    axes[1, 0].set_ylabel('Performance (Avg Reward)', fontsize=11)
    axes[1, 0].set_title('Agent Performance Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Acceptance rates
    red_accept = [h['red_pool'].get('acceptance_rate', 0) * 100 for h in history]
    blue_accept = [h['blue_pool'].get('acceptance_rate', 0) * 100 for h in history]
    axes[1, 1].plot(generations, red_accept, 'r-^', label='Red', linewidth=2, markersize=6)
    axes[1, 1].plot(generations, blue_accept, 'b-^', label='Blue', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Generation', fontsize=11)
    axes[1, 1].set_ylabel('Acceptance Rate (%)', fontsize=11)
    axes[1, 1].set_title('Policy Acceptance Rate (New Behaviors Found)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_coverage_analysis.png', dpi=150)
    print("Saved: enhanced_coverage_analysis.png")
    plt.show()


def print_improvement_comparison():
    """Compare original vs enhanced results"""
    
    print("\n" + "="*70)
    print("IMPROVEMENT COMPARISON")
    print("="*70)
    
    # Load original results
    try:
        with open('logs/aceac_v2_qd_swap_rl.json', 'r') as f:
            original = json.load(f)
        original_loaded = True
    except:
        print("Original results not found (run aceac_v2_swap_rl.py first)")
        original_loaded = False
    
    # Load enhanced results
    try:
        with open('logs/aceac_v2_enhanced.json', 'r') as f:
            enhanced = json.load(f)
        enhanced_loaded = True
    except:
        print("Enhanced results not found (run enhanced version first)")
        enhanced_loaded = False
    
    if original_loaded and enhanced_loaded:
        print("\n" + "-"*70)
        print("ORIGINAL vs ENHANCED")
        print("-"*70)
        
        print(f"\nArchive Size:")
        print(f"  Original Red:  {original['red_pool_final']['size']} policies")
        print(f"  Enhanced Red:  {enhanced['red_pool_final']['size']} policies")
        print(f"  Improvement:   +{enhanced['red_pool_final']['size'] - original['red_pool_final']['size']} policies")
        
        print(f"\nCoverage:")
        print(f"  Original Red:  {original['red_pool_final']['coverage']:.2%}")
        print(f"  Enhanced Red:  {enhanced['red_pool_final']['coverage']:.2%}")
        print(f"  Improvement:   {enhanced['red_pool_final']['coverage'] - original['red_pool_final']['coverage']:.2%}")
        
        print(f"\nBest Fitness:")
        print(f"  Original Red:  {original['red_pool_final']['best']:.2f}")
        print(f"  Enhanced Red:  {enhanced['red_pool_final']['best']:.2f}")
        print(f"  Improvement:   {enhanced['red_pool_final']['best'] - original['red_pool_final']['best']:.2f}")
        
        print(f"\nBehavioral Dimensions:")
        print(f"  Original:      2D (Kill Chain + Diversity)")
        print(f"  Enhanced:      3D (+ Effectiveness)")
        
        print(f"\nTraining Duration:")
        print(f"  Original:      {original['duration_minutes']:.1f} minutes")
        print(f"  Enhanced:      {enhanced['duration_minutes']:.1f} minutes")
        print(f"  Difference:    {enhanced['duration_minutes'] - original['duration_minutes']:.1f} minutes")
        
    print("="*70)


if __name__ == "__main__":
    print("Generating enhanced visualizations...")
    print_improvement_comparison()
    plot_enhanced_coverage()
    plot_3d_behavior_space()
    print("\n✅ Visualizations complete!")
