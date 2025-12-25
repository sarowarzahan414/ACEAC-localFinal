"""
Visualization for QD-SWAP RL results
Shows behavioral diversity and training progress
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_qd_archive_coverage():
    """Plot QD archive grid coverage"""
    
    try:
        with open('logs/aceac_v2_qd_swap_rl.json', 'r') as f:
            log = json.load(f)
    except:
        print("Run training first: python aceac_v2_swap_rl.py")
        return
    
    history = log['history']
    
    # Extract coverage over generations
    generations = [h['generation'] for h in history]
    red_coverage = [h['red_pool']['coverage'] * 100 for h in history]
    blue_coverage = [h['blue_pool']['coverage'] * 100 for h in history]
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Archive coverage
    plt.subplot(1, 2, 1)
    plt.plot(generations, red_coverage, 'r-o', label='Red Archive', linewidth=2)
    plt.plot(generations, blue_coverage, 'b-o', label='Blue Archive', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Archive Coverage (%)')
    plt.title('QD Archive Coverage Over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Archive size
    plt.subplot(1, 2, 2)
    red_sizes = [h['red_pool']['size'] for h in history]
    blue_sizes = [h['blue_pool']['size'] for h in history]
    plt.plot(generations, red_sizes, 'r-s', label='Red Policies', linewidth=2)
    plt.plot(generations, blue_sizes, 'b-s', label='Blue Policies', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Number of Policies')
    plt.title('Archive Size Over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qd_archive_growth.png', dpi=150)
    print("Saved: qd_archive_growth.png")
    plt.show()


def plot_behavior_space():
    """Plot discovered behaviors in 2D space"""
    
    try:
        with open('logs/aceac_v2_qd_swap_rl.json', 'r') as f:
            log = json.load(f)
    except:
        print("Run training first: python aceac_v2_swap_rl.py")
        return
    
    history = log['history']
    
    # Extract behaviors
    red_behaviors = [h['red_behavior'] for h in history]
    blue_behaviors = [h['blue_behavior'] for h in history]
    
    plt.figure(figsize=(10, 10))
    
    # Plot grid
    resolution = log.get('grid_resolution', 10)
    for i in range(resolution):
        for j in range(resolution):
            rect = Rectangle((i/resolution, j/resolution), 1/resolution, 1/resolution,
                           linewidth=0.5, edgecolor='gray', facecolor='none', alpha=0.3)
            plt.gca().add_patch(rect)
    
    # Plot behaviors
    red_behaviors_array = np.array(red_behaviors)
    blue_behaviors_array = np.array(blue_behaviors)
    
    plt.scatter(red_behaviors_array[:, 0], red_behaviors_array[:, 1],
               c=range(len(red_behaviors)), cmap='Reds',
               s=100, alpha=0.7, label='Red Behaviors', edgecolors='darkred')
    
    plt.scatter(blue_behaviors_array[:, 0], blue_behaviors_array[:, 1],
               c=range(len(blue_behaviors)), cmap='Blues',
               s=100, alpha=0.7, label='Blue Behaviors', edgecolors='darkblue')
    
    plt.xlabel('Kill Chain Progression Rate', fontsize=12)
    plt.ylabel('Tool Diversity (Entropy)', fontsize=12)
    plt.title('Discovered Behaviors in QD Archive\n(Color = Generation)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qd_behavior_space_discovered.png', dpi=150)
    print("Saved: qd_behavior_space_discovered.png")
    plt.show()


if __name__ == "__main__":
    print("Generating QD visualizations...")
    plot_qd_archive_coverage()
    plot_behavior_space()
    print("\n✅ Visualizations complete!")
    
    
    
