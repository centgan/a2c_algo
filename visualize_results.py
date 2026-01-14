#!/usr/bin/env python3
"""
Visualize agent performance metrics from the results directory.
Usage: python visualize_results.py [agent_id]
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

def load_latest_metrics(agent_id):
    """Load the most recent metrics file for an agent"""
    results_dir = Path(f'./results/{agent_id}')
    
    if not results_dir.exists():
        print(f"No results directory found for agent {agent_id}")
        return None
    
    # Find all metrics files
    metrics_files = list(results_dir.glob('*.json'))
    
    if not metrics_files:
        print(f"No metrics files found for agent {agent_id}")
        return None
    
    # Get the most recent file
    latest_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading metrics from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_agent_performance(agent_id):
    """Create comprehensive performance plots for an agent"""
    metrics = load_latest_metrics(agent_id)
    
    if metrics is None:
        return
    
    # Convert timestamps to datetime
    timestamps = [datetime.fromtimestamp(ts) for ts in metrics['timestamps']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Agent {agent_id} Performance Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Balance over time
    axes[0].plot(timestamps, metrics['balance_history'], linewidth=1.5, color='blue')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    axes[0].set_ylabel('Balance ($)', fontsize=12)
    axes[0].set_title('Account Balance Over Time', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add final balance annotation
    final_balance = metrics['balance_history'][-1]
    axes[0].annotate(f'Final: ${final_balance:.2f}', 
                     xy=(timestamps[-1], final_balance),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     fontsize=10)
    
    # Plot 2: Realized rewards (moving average)
    reward_real_means = [np.mean(r) for r in metrics['reward_real_history']]
    window = min(50, len(reward_real_means) // 10)  # Adaptive window size
    if window > 1:
        moving_avg = np.convolve(reward_real_means, np.ones(window)/window, mode='valid')
        ma_timestamps = timestamps[window-1:]
        axes[1].plot(ma_timestamps, moving_avg, linewidth=2, color='green', label=f'{window}-step MA')
    axes[1].plot(timestamps, reward_real_means, alpha=0.3, color='green', label='Raw')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Avg Realized Reward', fontsize=12)
    axes[1].set_title('Realized Rewards (Moving Average)', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Unrealized rewards (moving average)
    reward_unreal_means = [np.mean(r) for r in metrics['reward_unreal_history']]
    if window > 1:
        moving_avg_unreal = np.convolve(reward_unreal_means, np.ones(window)/window, mode='valid')
        axes[2].plot(ma_timestamps, moving_avg_unreal, linewidth=2, color='orange', label=f'{window}-step MA')
    axes[2].plot(timestamps, reward_unreal_means, alpha=0.3, color='orange', label='Raw')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].set_ylabel('Avg Unrealized Reward', fontsize=12)
    axes[2].set_title('Unrealized Rewards (Moving Average)', fontsize=13)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Format x-axis for all plots
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'./results/{agent_id}/performance_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS - Agent {agent_id}")
    print("="*60)
    print(f"Total Steps: {len(metrics['balance_history'])}")
    print(f"Total Trades: {metrics['num_trades']}")
    print(f"Starting Balance: ${metrics['balance_history'][0]:.2f}")
    print(f"Final Balance: ${metrics['balance_history'][-1]:.2f}")
    print(f"Net P/L: ${metrics['balance_history'][-1] - metrics['balance_history'][0]:.2f}")
    print(f"Max Balance: ${max(metrics['balance_history']):.2f}")
    print(f"Min Balance: ${min(metrics['balance_history']):.2f}")
    print(f"Avg Realized Reward: {np.mean(reward_real_means):.4f}")
    print(f"Avg Unrealized Reward: {np.mean(reward_unreal_means):.4f}")
    print("="*60)
    
    plt.show()

def compare_all_agents():
    """Compare performance across all agents"""
    results_dir = Path('./results')
    
    if not results_dir.exists():
        print("No results directory found")
        return
    
    agent_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not agent_dirs:
        print("No agent directories found")
        return
    
    plt.figure(figsize=(14, 8))
    
    for agent_dir in sorted(agent_dirs):
        agent_id = agent_dir.name
        metrics = load_latest_metrics(agent_id)
        
        if metrics is None:
            continue
        
        timestamps = [datetime.fromtimestamp(ts) for ts in metrics['timestamps']]
        plt.plot(timestamps, metrics['balance_history'], label=f'Agent {agent_id}', linewidth=2)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Balance ($)', fontsize=12)
    plt.title('All Agents Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_file = './results/all_agents_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_file}")
    
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            compare_all_agents()
        else:
            agent_id = sys.argv[1]
            plot_agent_performance(agent_id)
    else:
        print("Usage:")
        print("  python visualize_results.py [agent_id]  - Plot specific agent")
        print("  python visualize_results.py all         - Compare all agents")
        print("\nExample: python visualize_results.py 0")
