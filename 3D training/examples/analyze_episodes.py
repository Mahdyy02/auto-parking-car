<<<<<<< HEAD
"""Analyze and compare multiple episodes to diagnose data quality issues.

This script:
1. Loads multiple episodes
2. Shows action distributions and patterns
3. Identifies if episodes are too similar/different
4. Helps diagnose why BC training might fail

Usage:
    python analyze_episodes.py --data_dir data/episodes --n_episodes 20
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


def analyze_episodes(data_dir, n_episodes=20, save_plots=True):
    """Analyze episode data quality."""
    
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if not files:
        print(f"Error: No episodes found in {data_dir}")
        return
    
    if n_episodes > len(files):
        n_episodes = len(files)
    
    files = files[:n_episodes]
    
    print("="*80)
    print(f"ANALYZING {n_episodes} EPISODES")
    print("="*80)
    print()
    
    # Collect statistics
    all_actions = []
    episode_stats = []
    
    for i, f in enumerate(files):
        data = np.load(f)
        actions = data['actions']
        rewards = data['rewards']
        
        stats = {
            'file': os.path.basename(f),
            'n_steps': len(actions),
            'total_reward': rewards.sum(),
            'final_reward': rewards[-1] if len(rewards) > 0 else 0,
            'steering_mean': actions[:, 0].mean(),
            'steering_std': actions[:, 0].std(),
            'steering_min': actions[:, 0].min(),
            'steering_max': actions[:, 0].max(),
            'throttle_mean': actions[:, 1].mean(),
            'throttle_std': actions[:, 1].std(),
            'brake_mean': actions[:, 2].mean(),
            'brake_std': actions[:, 2].std(),
        }
        
        episode_stats.append(stats)
        all_actions.append(actions)
    
    # Print episode-by-episode summary
    print("Episode-by-Episode Summary:")
    print("-"*80)
    print(f"{'Ep':<3} {'Steps':<6} {'Reward':<8} {'Steer(μ±σ)':<15} {'Steer Range':<15} {'Throttle(μ)':<12}")
    print("-"*80)
    
    for i, stats in enumerate(episode_stats):
        is_success = stats['final_reward'] > 5.0
        marker = "✓" if is_success else "✗"
        
        print(f"{i+1:2d}{marker} {stats['n_steps']:<6} {stats['total_reward']:7.2f}  "
              f"{stats['steering_mean']:+.3f}±{stats['steering_std']:.3f}      "
              f"[{stats['steering_min']:+.2f}, {stats['steering_max']:+.2f}]     "
              f"{stats['throttle_mean']:.3f}")
    
    print("-"*80)
    print()
    
    # Aggregate statistics
    all_actions_concat = np.concatenate(all_actions, axis=0)
    
    print("Aggregate Statistics Across All Episodes:")
    print("-"*80)
    print(f"Total timesteps: {len(all_actions_concat)}")
    print(f"Average episode length: {np.mean([s['n_steps'] for s in episode_stats]):.1f} steps")
    print()
    print("Steering:")
    print(f"  Mean:   {all_actions_concat[:, 0].mean():+.4f}")
    print(f"  Std:    {all_actions_concat[:, 0].std():.4f}")
    print(f"  Min:    {all_actions_concat[:, 0].min():+.4f}")
    print(f"  Max:    {all_actions_concat[:, 0].max():+.4f}")
    print(f"  Range:  {all_actions_concat[:, 0].max() - all_actions_concat[:, 0].min():.4f}")
    print()
    print("Throttle:")
    print(f"  Mean:   {all_actions_concat[:, 1].mean():.4f}")
    print(f"  Std:    {all_actions_concat[:, 1].std():.4f}")
    print(f"  Min:    {all_actions_concat[:, 1].min():.4f}")
    print(f"  Max:    {all_actions_concat[:, 1].max():.4f}")
    print()
    print("Brake:")
    print(f"  Mean:   {all_actions_concat[:, 2].mean():.4f}")
    print(f"  Std:    {all_actions_concat[:, 2].std():.4f}")
    print()
    
    # Diagnosis
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    issues = []
    
    # Check steering variance
    steering_std = all_actions_concat[:, 0].std()
    if steering_std < 0.1:
        issues.append("❌ CRITICAL: Steering variance is too low (< 0.1)")
        issues.append("   → Car is driving almost straight in all episodes")
        issues.append("   → BC cannot learn meaningful steering behavior")
        issues.append("   → Solution: Collect episodes with more turning/maneuvering")
    elif steering_std < 0.2:
        issues.append("⚠️  WARNING: Steering variance is low (< 0.2)")
        issues.append("   → Episodes might be too similar")
        issues.append("   → Consider more diverse trajectories")
    else:
        print("✓ Steering variance looks good")
    
    # Check episode length
    avg_steps = np.mean([s['n_steps'] for s in episode_stats])
    if avg_steps < 10:
        issues.append("❌ CRITICAL: Episodes are too short (< 10 steps)")
        issues.append("   → Not enough data per episode")
        issues.append("   → Task might be too easy (start very close to goal?)")
        issues.append("   → Solution: Check task setup, move car further from goal")
    elif avg_steps < 30:
        issues.append("⚠️  WARNING: Episodes are quite short (< 30 steps)")
        issues.append("   → Consider making task more challenging")
    else:
        print("✓ Episode length looks reasonable")
    
    # Check if actions are identical across episodes
    if len(episode_stats) > 1:
        steer_means = [s['steering_mean'] for s in episode_stats]
        if np.std(steer_means) < 0.05:
            issues.append("❌ CRITICAL: All episodes have nearly identical steering patterns")
            issues.append("   → Episodes are too similar")
            issues.append("   → BC will just learn to output the mean action")
    
    # Check success rate
    successes = sum(1 for s in episode_stats if s['final_reward'] > 5.0)
    success_rate = successes / len(episode_stats)
    if success_rate < 0.5:
        issues.append(f"⚠️  WARNING: Success rate is low ({success_rate*100:.0f}%)")
        issues.append("   → BC is learning from many failed episodes")
        issues.append("   → Consider filtering out failed episodes before training")
    else:
        print(f"✓ Success rate is good ({success_rate*100:.0f}%)")
    
    if issues:
        print()
        for issue in issues:
            print(issue)
    else:
        print("✓ All checks passed!")
    
    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if steering_std < 0.1:
        print("\n🎯 TOP PRIORITY: Fix steering variance")
        print("\nYour episodes show almost no steering movement. This means:")
        print("• The car is going straight")
        print("• BC cannot learn when/how to steer")
        print("• Training loss will plateau")
        print("\nHow to fix:")
        print("1. When collecting episodes manually, actively steer the car")
        print("2. Try different approach angles to the parking spot")
        print("3. Add obstacles that require maneuvering")
        print("4. Make the parking spot require a turn to enter")
        print("\nExample: Instead of straight-line approach:")
        print("  Start → → → → Goal")
        print("Try curved approaches:")
        print("  Start → ↘ → ↓ → ↙ Goal")
    
    if avg_steps < 10:
        print("\n🎯 Fix episode length")
        print("\nYour episodes are too short (< 10 steps). This suggests:")
        print("• Car starts very close to goal")
        print("• OR task is trivial")
        print("\nHow to fix:")
        print("1. Check starting distance to goal (should be 20-50 meters)")
        print("2. Reduce goal_threshold (make it harder to reach)")
        print("3. Add intermediate waypoints")
    
    # Generate plots
    if save_plots:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Steering distribution
        axes[0, 0].hist(all_actions_concat[:, 0], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Steering Distribution')
        axes[0, 0].set_xlabel('Steering')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='r', linestyle='--', label='Zero')
        axes[0, 0].legend()
        
        # Plot 2: Throttle distribution
        axes[0, 1].hist(all_actions_concat[:, 1], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_title('Throttle Distribution')
        axes[0, 1].set_xlabel('Throttle')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Steering over time (first 3 episodes)
        for i in range(min(3, len(all_actions))):
            axes[1, 0].plot(all_actions[i][:, 0], label=f'Episode {i+1}', alpha=0.7)
        axes[1, 0].set_title('Steering Over Time (First 3 Episodes)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Steering')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Episode lengths
        steps_list = [s['n_steps'] for s in episode_stats]
        axes[1, 1].bar(range(len(steps_list)), steps_list, alpha=0.7)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].axhline(np.mean(steps_list), color='r', linestyle='--', label=f'Mean: {np.mean(steps_list):.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'episode_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved analysis plots to: {plot_path}")
        print("  Open this file to visualize the action distributions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/episodes',
                        help='Directory containing episode files')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes to analyze')
    parser.add_argument('--no_plots', action='store_true',
                        help='Do not generate plots')
    args = parser.parse_args()
    
    analyze_episodes(args.data_dir, n_episodes=args.n_episodes, save_plots=not args.no_plots)


if __name__ == '__main__':
    main()
=======
"""Analyze and compare multiple episodes to diagnose data quality issues.

This script:
1. Loads multiple episodes
2. Shows action distributions and patterns
3. Identifies if episodes are too similar/different
4. Helps diagnose why BC training might fail

Usage:
    python analyze_episodes.py --data_dir data/episodes --n_episodes 20
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


def analyze_episodes(data_dir, n_episodes=20, save_plots=True):
    """Analyze episode data quality."""
    
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if not files:
        print(f"Error: No episodes found in {data_dir}")
        return
    
    if n_episodes > len(files):
        n_episodes = len(files)
    
    files = files[:n_episodes]
    
    print("="*80)
    print(f"ANALYZING {n_episodes} EPISODES")
    print("="*80)
    print()
    
    # Collect statistics
    all_actions = []
    episode_stats = []
    
    for i, f in enumerate(files):
        data = np.load(f)
        actions = data['actions']
        rewards = data['rewards']
        
        stats = {
            'file': os.path.basename(f),
            'n_steps': len(actions),
            'total_reward': rewards.sum(),
            'final_reward': rewards[-1] if len(rewards) > 0 else 0,
            'steering_mean': actions[:, 0].mean(),
            'steering_std': actions[:, 0].std(),
            'steering_min': actions[:, 0].min(),
            'steering_max': actions[:, 0].max(),
            'throttle_mean': actions[:, 1].mean(),
            'throttle_std': actions[:, 1].std(),
            'brake_mean': actions[:, 2].mean(),
            'brake_std': actions[:, 2].std(),
        }
        
        episode_stats.append(stats)
        all_actions.append(actions)
    
    # Print episode-by-episode summary
    print("Episode-by-Episode Summary:")
    print("-"*80)
    print(f"{'Ep':<3} {'Steps':<6} {'Reward':<8} {'Steer(μ±σ)':<15} {'Steer Range':<15} {'Throttle(μ)':<12}")
    print("-"*80)
    
    for i, stats in enumerate(episode_stats):
        is_success = stats['final_reward'] > 5.0
        marker = "✓" if is_success else "✗"
        
        print(f"{i+1:2d}{marker} {stats['n_steps']:<6} {stats['total_reward']:7.2f}  "
              f"{stats['steering_mean']:+.3f}±{stats['steering_std']:.3f}      "
              f"[{stats['steering_min']:+.2f}, {stats['steering_max']:+.2f}]     "
              f"{stats['throttle_mean']:.3f}")
    
    print("-"*80)
    print()
    
    # Aggregate statistics
    all_actions_concat = np.concatenate(all_actions, axis=0)
    
    print("Aggregate Statistics Across All Episodes:")
    print("-"*80)
    print(f"Total timesteps: {len(all_actions_concat)}")
    print(f"Average episode length: {np.mean([s['n_steps'] for s in episode_stats]):.1f} steps")
    print()
    print("Steering:")
    print(f"  Mean:   {all_actions_concat[:, 0].mean():+.4f}")
    print(f"  Std:    {all_actions_concat[:, 0].std():.4f}")
    print(f"  Min:    {all_actions_concat[:, 0].min():+.4f}")
    print(f"  Max:    {all_actions_concat[:, 0].max():+.4f}")
    print(f"  Range:  {all_actions_concat[:, 0].max() - all_actions_concat[:, 0].min():.4f}")
    print()
    print("Throttle:")
    print(f"  Mean:   {all_actions_concat[:, 1].mean():.4f}")
    print(f"  Std:    {all_actions_concat[:, 1].std():.4f}")
    print(f"  Min:    {all_actions_concat[:, 1].min():.4f}")
    print(f"  Max:    {all_actions_concat[:, 1].max():.4f}")
    print()
    print("Brake:")
    print(f"  Mean:   {all_actions_concat[:, 2].mean():.4f}")
    print(f"  Std:    {all_actions_concat[:, 2].std():.4f}")
    print()
    
    # Diagnosis
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    issues = []
    
    # Check steering variance
    steering_std = all_actions_concat[:, 0].std()
    if steering_std < 0.1:
        issues.append("❌ CRITICAL: Steering variance is too low (< 0.1)")
        issues.append("   → Car is driving almost straight in all episodes")
        issues.append("   → BC cannot learn meaningful steering behavior")
        issues.append("   → Solution: Collect episodes with more turning/maneuvering")
    elif steering_std < 0.2:
        issues.append("⚠️  WARNING: Steering variance is low (< 0.2)")
        issues.append("   → Episodes might be too similar")
        issues.append("   → Consider more diverse trajectories")
    else:
        print("✓ Steering variance looks good")
    
    # Check episode length
    avg_steps = np.mean([s['n_steps'] for s in episode_stats])
    if avg_steps < 10:
        issues.append("❌ CRITICAL: Episodes are too short (< 10 steps)")
        issues.append("   → Not enough data per episode")
        issues.append("   → Task might be too easy (start very close to goal?)")
        issues.append("   → Solution: Check task setup, move car further from goal")
    elif avg_steps < 30:
        issues.append("⚠️  WARNING: Episodes are quite short (< 30 steps)")
        issues.append("   → Consider making task more challenging")
    else:
        print("✓ Episode length looks reasonable")
    
    # Check if actions are identical across episodes
    if len(episode_stats) > 1:
        steer_means = [s['steering_mean'] for s in episode_stats]
        if np.std(steer_means) < 0.05:
            issues.append("❌ CRITICAL: All episodes have nearly identical steering patterns")
            issues.append("   → Episodes are too similar")
            issues.append("   → BC will just learn to output the mean action")
    
    # Check success rate
    successes = sum(1 for s in episode_stats if s['final_reward'] > 5.0)
    success_rate = successes / len(episode_stats)
    if success_rate < 0.5:
        issues.append(f"⚠️  WARNING: Success rate is low ({success_rate*100:.0f}%)")
        issues.append("   → BC is learning from many failed episodes")
        issues.append("   → Consider filtering out failed episodes before training")
    else:
        print(f"✓ Success rate is good ({success_rate*100:.0f}%)")
    
    if issues:
        print()
        for issue in issues:
            print(issue)
    else:
        print("✓ All checks passed!")
    
    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if steering_std < 0.1:
        print("\n🎯 TOP PRIORITY: Fix steering variance")
        print("\nYour episodes show almost no steering movement. This means:")
        print("• The car is going straight")
        print("• BC cannot learn when/how to steer")
        print("• Training loss will plateau")
        print("\nHow to fix:")
        print("1. When collecting episodes manually, actively steer the car")
        print("2. Try different approach angles to the parking spot")
        print("3. Add obstacles that require maneuvering")
        print("4. Make the parking spot require a turn to enter")
        print("\nExample: Instead of straight-line approach:")
        print("  Start → → → → Goal")
        print("Try curved approaches:")
        print("  Start → ↘ → ↓ → ↙ Goal")
    
    if avg_steps < 10:
        print("\n🎯 Fix episode length")
        print("\nYour episodes are too short (< 10 steps). This suggests:")
        print("• Car starts very close to goal")
        print("• OR task is trivial")
        print("\nHow to fix:")
        print("1. Check starting distance to goal (should be 20-50 meters)")
        print("2. Reduce goal_threshold (make it harder to reach)")
        print("3. Add intermediate waypoints")
    
    # Generate plots
    if save_plots:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Steering distribution
        axes[0, 0].hist(all_actions_concat[:, 0], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Steering Distribution')
        axes[0, 0].set_xlabel('Steering')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='r', linestyle='--', label='Zero')
        axes[0, 0].legend()
        
        # Plot 2: Throttle distribution
        axes[0, 1].hist(all_actions_concat[:, 1], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_title('Throttle Distribution')
        axes[0, 1].set_xlabel('Throttle')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Steering over time (first 3 episodes)
        for i in range(min(3, len(all_actions))):
            axes[1, 0].plot(all_actions[i][:, 0], label=f'Episode {i+1}', alpha=0.7)
        axes[1, 0].set_title('Steering Over Time (First 3 Episodes)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Steering')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Episode lengths
        steps_list = [s['n_steps'] for s in episode_stats]
        axes[1, 1].bar(range(len(steps_list)), steps_list, alpha=0.7)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].axhline(np.mean(steps_list), color='r', linestyle='--', label=f'Mean: {np.mean(steps_list):.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'episode_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved analysis plots to: {plot_path}")
        print("  Open this file to visualize the action distributions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/episodes',
                        help='Directory containing episode files')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes to analyze')
    parser.add_argument('--no_plots', action='store_true',
                        help='Do not generate plots')
    args = parser.parse_args()
    
    analyze_episodes(args.data_dir, n_episodes=args.n_episodes, save_plots=not args.no_plots)


if __name__ == '__main__':
    main()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
