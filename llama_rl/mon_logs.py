#!/usr/bin/env python3
"""
LLM Training Monitor - Real-time Log Analysis Tool

This script monitors a training log file for LLM fine-tuning, parses log entries,
calculates statistics, and generates visualizations in real-time as new updates appear.
"""

import os
import re
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
from pathlib import Path
import scipy.stats as stats


class LLMTrainingMonitor:
    """Monitors LLM training logs and provides real-time analysis and visualization."""

    def __init__(self, log_file_path: str, output_dir: str = None, update_interval: int = 5):
        """
        Initialize the training monitor.

        Args:
            log_file_path: Path to the log file to monitor
            output_dir: Directory to save output charts (default: same directory as log file)
            update_interval: How often to check for updates (in seconds)
        """
        self.log_file_path = log_file_path

        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(log_file_path), "training_monitor_output")
        else:
            self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.update_interval = update_interval
        self.last_position = 0
        self.updates_data = []
        self.df = pd.DataFrame()

        # Define the expected metrics
        self.expected_metrics = [
            'total_loss', 'actor_loss', 'critic_loss', 'kl_loss',
            'supervision_loss', 'llm_loss', 'llm_valid_pct',
            'reward_orig', 'advantage', 'value',
            'lambda_ppo', 'lambda_sup'
        ]

        # Regex pattern for update logs
        self.update_pattern = re.compile(
            r'\[Updt (\d+)\|(\d+\.\d+)s\] Loss\(Tot\):(\d+\.\d+) Act:([-]?\d+\.\d+) '
            r'Crit:(\d+\.\d+) KL:(\d+\.\d+) Sup:(\d+\.\d+) LLM:(\d+\.\d+) LLMValid:(\d+\.\d+)% '
            r'\| Rew\(orig\):([-]?\d+\.\d+) Adv:([-]?\d+\.\d+) Val:([-]?\d+\.\d+) '
            r'\| Lamb\(PPO\):(\d+\.\d+) Lamb\(Sup\):(\d+\.\d+)'
        )

        # Configuration for plots
        self.plot_config = {
            'style': 'seaborn-v0_8-darkgrid',
            'figsize': (15, 12),
            'dpi': 100,
            'cmap': 'viridis'
        }

        # Initialize plots dictionary to keep track of created plots
        self.plots = {}

        print(f"Monitoring log file: {self.log_file_path}")
        print(f"Output directory: {self.output_dir}")

    def parse_update_log(self, line: str) -> Optional[Dict]:
        """
        Parse a single update log line and extract metrics.

        Args:
            line: Log line to parse

        Returns:
            Dictionary of extracted metrics or None if not an update log
        """
        match = self.update_pattern.search(line)
        if not match:
            return None

        # Extract values
        update_num = int(match.group(1))
        update_time = float(match.group(2))
        total_loss = float(match.group(3))
        actor_loss = float(match.group(4))
        critic_loss = float(match.group(5))
        kl_loss = float(match.group(6))
        supervision_loss = float(match.group(7))
        llm_loss = float(match.group(8))
        llm_valid_pct = float(match.group(9))
        reward_orig = float(match.group(10))
        advantage = float(match.group(11))
        value = float(match.group(12))
        lambda_ppo = float(match.group(13))
        lambda_sup = float(match.group(14))

        # Create update data dictionary
        update_data = {
            'update': update_num,
            'time': update_time,
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'kl_loss': kl_loss,
            'supervision_loss': supervision_loss,
            'llm_loss': llm_loss,
            'llm_valid_pct': llm_valid_pct,
            'reward_orig': reward_orig,
            'advantage': advantage,
            'value': value,
            'lambda_ppo': lambda_ppo,
            'lambda_sup': lambda_sup,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        return update_data

    def read_new_logs(self) -> List[Dict]:
        """
        Read new log entries since the last check.

        Returns:
            List of new update data dictionaries
        """
        new_updates = []

        try:
            with open(self.log_file_path, 'r') as f:
                # Move to the last position we read
                f.seek(self.last_position)

                # Read new lines
                for line in f:
                    update_data = self.parse_update_log(line)
                    if update_data:
                        new_updates.append(update_data)

                # Update the position for next read
                self.last_position = f.tell()

        except Exception as e:
            print(f"Error reading log file: {e}")

        return new_updates

    def update_data(self) -> bool:
        """
        Check for new updates and process them.

        Returns:
            True if new updates were found, False otherwise
        """
        new_updates = self.read_new_logs()

        if not new_updates:
            return False

        # Add new updates to our data
        self.updates_data.extend(new_updates)

        # Update the dataframe
        self.df = pd.DataFrame(self.updates_data)

        # Print info about new updates
        for update in new_updates:
            print(f"New update detected: #{update['update']} - Total Loss: {update['total_loss']:.4f}, "
                  f"LLM Valid: {update['llm_valid_pct']:.1f}%, Reward: {update['reward_orig']:.4f}")

        return True

    def calculate_statistics(self) -> Dict:
        """
        Calculate statistics based on the current data.

        Returns:
            Dictionary containing calculated statistics
        """
        if self.df.empty:
            return {}

        stats = {}

        # Calculate basic statistics for key metrics
        for metric in ['total_loss', 'actor_loss', 'critic_loss', 'kl_loss',
                       'llm_loss', 'llm_valid_pct', 'reward_orig']:
            metric_data = self.df[metric].values
            stats[metric] = {
                'min': np.min(metric_data),
                'max': np.max(metric_data),
                'mean': np.mean(metric_data),
                'std': np.std(metric_data),
                'latest': metric_data[-1] if len(metric_data) > 0 else None,
                'range': np.max(metric_data) - np.min(metric_data)
            }

        # Add trend analysis
        recent_window = min(5, len(self.df))
        for metric in ['total_loss', 'llm_valid_pct', 'reward_orig']:
            if len(self.df) >= 2:
                # Calculate if metric is improving
                # For loss, lower is better; for valid percentage, higher is better; for reward, higher is better
                improving_direction = -1 if metric in ['total_loss'] else 1
                first_avg = self.df[metric].iloc[:2].mean()
                last_avg = self.df[metric].iloc[-2:].mean()
                stats[f"{metric}_improving"] = (last_avg - first_avg) * improving_direction > 0

                # Calculate trend direction based on recent updates
                recent_values = self.df[metric].iloc[-recent_window:].values
                if len(recent_values) >= 3:
                    # Get differences between consecutive elements
                    diffs = np.diff(recent_values)
                    # Check if all differences are positive or negative
                    all_positive = np.all(diffs > 0)
                    all_negative = np.all(diffs < 0)

                    if all_positive:
                        stats[f"{metric}_trend"] = "↑" if improving_direction == 1 else "↓"
                    elif all_negative:
                        stats[f"{metric}_trend"] = "↓" if improving_direction == 1 else "↑"
                    elif diffs[-1] > 0:
                        stats[f"{metric}_trend"] = "↗" if improving_direction == 1 else "↘"
                    elif diffs[-1] < 0:
                        stats[f"{metric}_trend"] = "↘" if improving_direction == 1 else "↗"
                    else:
                        stats[f"{metric}_trend"] = "⟷"

        # Check for oscillation in key metrics
        for metric in ['total_loss', 'actor_loss', 'critic_loss', 'reward_orig']:
            if len(self.df) >= 4:
                values = self.df[metric].values
                diffs = np.diff(values)
                # Count sign changes in differences
                sign_changes = np.sum(np.diff(np.signbit(diffs)) != 0)
                # If we have at least 2 sign changes, that's oscillation
                stats[f"{metric}_oscillating"] = sign_changes >= 2

        # Calculate actor loss sign changes
        if len(self.df) >= 2:
            actor_loss_values = self.df['actor_loss'].values
            sign_changes = np.sum(np.diff(np.signbit(actor_loss_values)) != 0)
            stats['actor_loss_sign_changes'] = int(sign_changes)

        # Count KL divergence spikes
        stats['kl_spikes'] = int(np.sum(self.df['kl_loss'] > 0.5))

        # Check if lambdas are annealing
        if len(self.df) >= 2:
            ppo_unique = len(self.df['lambda_ppo'].unique())
            sup_unique = len(self.df['lambda_sup'].unique())
            stats['lambdas_annealing'] = ppo_unique > 1 or sup_unique > 1

        # Calculate correlations between key metrics
        if len(self.df) >= 3:
            # Correlation matrix
            corr_metrics = ['total_loss', 'actor_loss', 'critic_loss', 'llm_loss',
                           'llm_valid_pct', 'reward_orig']
            corr_matrix = self.df[corr_metrics].corr()

            # Extract key correlations
            stats['corr_llm_valid_reward'] = corr_matrix.loc['llm_valid_pct', 'reward_orig']
            stats['corr_total_loss_reward'] = corr_matrix.loc['total_loss', 'reward_orig']
            stats['corr_llm_loss_valid'] = corr_matrix.loc['llm_loss', 'llm_valid_pct']

            # Flag concerning correlations
            stats['concerning_corr_llm_valid_reward'] = stats['corr_llm_valid_reward'] < 0

        # Training health assessment (simple heuristic)
        if len(self.df) >= 3:
            health_score = 0

            # Total loss improving?
            if stats.get('total_loss_improving', False):
                health_score += 1

            # LLM valid percentage improving?
            if stats.get('llm_valid_pct_improving', False):
                health_score += 1

            # Rewards improving?
            if stats.get('reward_orig_improving', False):
                health_score += 1

            # KL divergence stable?
            recent_kl = self.df['kl_loss'].iloc[-3:].values
            if np.all(recent_kl < 0.1):
                health_score += 1

            # LLM loss improving?
            llm_first_avg = self.df['llm_loss'].iloc[:2].mean()
            llm_last_avg = self.df['llm_loss'].iloc[-2:].mean()
            if llm_last_avg < llm_first_avg:
                health_score += 1

            # Is supervision loss stable?
            sup_recent = self.df['supervision_loss'].iloc[-3:].values
            sup_range = np.max(sup_recent) - np.min(sup_recent)
            if sup_range < 0.2:
                health_score += 1

            # Non-oscillating behavior in main metrics?
            if not stats.get('total_loss_oscillating', True) and not stats.get('reward_orig_oscillating', True):
                health_score += 1

            # Map score to assessment
            if health_score >= 6:
                stats['training_health'] = "Excellent"
            elif health_score >= 4:
                stats['training_health'] = "Good"
            elif health_score >= 2:
                stats['training_health'] = "Concerning"
            else:
                stats['training_health'] = "Poor"

            stats['health_score'] = health_score

        # Analyze reward configuration issues based on observed data
        if len(self.df) >= 5:
            # Check for specific patterns that might suggest reward config issues
            stats['reward_config_suggestions'] = self._analyze_reward_config()

        return stats

    def _analyze_reward_config(self) -> List[str]:
        """
        Analyze the training data to identify potential reward configuration issues.

        Returns:
            List of suggestions for reward configuration improvements
        """
        suggestions = []

        # Calculate correlations for reward analysis
        corr_valid_reward = self.df['llm_valid_pct'].corr(self.df['reward_orig'])

        # Check for negative correlation between LLM valid % and rewards
        if corr_valid_reward < -0.3:
            suggestions.append(
                "CRITICAL: Negative correlation between LLM Valid % and rewards. "
                "Consider increasing answer_accuracy_weight and reducing think_critical_negative_words_weight."
            )
        elif corr_valid_reward < 0.1:
            suggestions.append(
                "Weak correlation between LLM Valid % and rewards. "
                "Consider increasing answer_accuracy_weight from 0.35 to 0.45."
            )

        # Check reward trends vs. LLM valid % trends
        # Get the top 3 updates by LLM valid %
        if len(self.df) >= 3:
            top_valid_updates = self.df.nlargest(3, 'llm_valid_pct')
            avg_reward_top_valid = top_valid_updates['reward_orig'].mean()

            # Get the bottom 3 updates by LLM valid %
            bottom_valid_updates = self.df.nsmallest(3, 'llm_valid_pct')
            avg_reward_bottom_valid = bottom_valid_updates['reward_orig'].mean()

            # If the average reward for top valid % is worse than for bottom valid %
            if avg_reward_top_valid < avg_reward_bottom_valid:
                suggestions.append(
                    "Updates with highest LLM Valid % have worse rewards than updates with lowest Valid %. "
                    "Critical issue with reward weighting - consider reducing think_critical_negative_words_weight "
                    "and increasing answer_accuracy_weight."
                )

        # Check for patterns in oscillating metrics
        total_loss_oscillating = False
        actor_loss_oscillating = False
        reward_oscillating = False

        if len(self.df) >= 4:
            # Simple oscillation detection
            total_loss_diffs = np.diff(self.df['total_loss'].values)
            sign_changes_total = np.sum(np.diff(np.signbit(total_loss_diffs)) != 0)
            total_loss_oscillating = sign_changes_total >= 2

            actor_loss_diffs = np.diff(self.df['actor_loss'].values)
            sign_changes_actor = np.sum(np.diff(np.signbit(actor_loss_diffs)) != 0)
            actor_loss_oscillating = sign_changes_actor >= 2

            reward_diffs = np.diff(self.df['reward_orig'].values)
            sign_changes_reward = np.sum(np.diff(np.signbit(reward_diffs)) != 0)
            reward_oscillating = sign_changes_reward >= 2

            # If multiple metrics are oscillating
            if sum([total_loss_oscillating, actor_loss_oscillating, reward_oscillating]) >= 2:
                suggestions.append(
                    "Multiple metrics showing oscillation patterns. Consider more balanced reward weights "
                    "and reducing critic learning rate to stabilize training."
                )

        # Check reward-related loss patterns
        if len(self.df) >= 5:
            recent_rewards = self.df['reward_orig'].iloc[-5:].values
            reward_improving = np.mean(recent_rewards[-2:]) > np.mean(recent_rewards[:2])

            if not reward_improving and total_loss_oscillating:
                suggestions.append(
                    "Rewards not improving despite oscillating total loss. Consider adjusting reward weights to "
                    "better align with training objectives: reduce think_critical_negative_words_weight (0.15→0.10) "
                    "and increase answer_accuracy_weight (0.35→0.40)."
                )

        # Check for overall negative rewards
        mean_reward = self.df['reward_orig'].mean()
        min_reward = self.df['reward_orig'].min()

        if mean_reward < -0.5 and min_reward < -0.8:
            suggestions.append(
                "Consistently negative rewards with low mean. Consider adjusting min_reward_clip "
                "or scaling reward components to provide more positive reinforcement signals."
            )

        # Check the relation between LLM loss and LLM valid %
        corr_llm_loss_valid = self.df['llm_loss'].corr(self.df['llm_valid_pct'])

        if corr_llm_loss_valid > 0.3:
            suggestions.append(
                "Positive correlation between LLM loss and LLM Valid %, which is counterintuitive. "
                "Consider rebalancing the reward components to better align with LLM learning objectives."
            )

        return suggestions

    def generate_plots(self) -> None:
        """Generate and save plots based on current training data."""
        if self.df.empty:
            print("No data to plot yet.")
            return

        # Set the plot style
        plt.style.use(self.plot_config['style'])

        # Create timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        latest_update = self.df['update'].max()

        # 1. Main metrics overview plot
        self._create_metrics_overview_plot(timestamp, latest_update)

        # 2. Loss components plot
        self._create_loss_components_plot(timestamp, latest_update)

        # 3. Actor-Critic behavior plot
        self._create_actor_critic_plot(timestamp, latest_update)

        # 4. Reward analysis plot
        self._create_reward_analysis_plot(timestamp, latest_update)

        # 5. LLM performance plot
        self._create_llm_performance_plot(timestamp, latest_update)

        # 6. Training health dashboard
        self._create_training_health_dashboard(timestamp, latest_update)

        # 7. Correlation matrix plot
        self._create_correlation_matrix_plot(timestamp, latest_update)

    def _create_metrics_overview_plot(self, timestamp: str, latest_update: int) -> None:
        """Create overview plot showing main training metrics."""
        fig, axes = plt.subplots(3, 1, figsize=self.plot_config['figsize'],
                                 dpi=self.plot_config['dpi'], sharex=True)

        # Total loss
        self.df.plot(x='update', y='total_loss', ax=axes[0],
                     marker='o', color='blue', linewidth=2)
        axes[0].set_title('Total Loss', fontsize=14)
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        # LLM Valid Percentage
        self.df.plot(x='update', y='llm_valid_pct', ax=axes[1],
                     marker='o', color='green', linewidth=2)
        axes[1].set_title('LLM Valid Percentage', fontsize=14)
        axes[1].set_ylabel('Valid %')
        axes[1].grid(True)

        # Original Reward
        self.df.plot(x='update', y='reward_orig', ax=axes[2],
                     marker='o', color='red', linewidth=2)
        axes[2].set_title('Original Reward', fontsize=14)
        axes[2].set_xlabel('Update')
        axes[2].set_ylabel('Reward')
        axes[2].grid(True)

        # Ensure x-axis shows integer ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save the figure
        filename = f'metrics_overview_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved metrics overview plot to: {filepath}")

    def _create_loss_components_plot(self, timestamp: str, latest_update: int) -> None:
        """Create plot showing the different loss components."""
        fig, axes = plt.subplots(3, 2, figsize=self.plot_config['figsize'],
                                 dpi=self.plot_config['dpi'], sharex=True)

        # Actor Loss
        self.df.plot(x='update', y='actor_loss', ax=axes[0, 0],
                     marker='o', color='blue', linewidth=2)
        axes[0, 0].set_title('Actor Loss', fontsize=14)
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Critic Loss
        self.df.plot(x='update', y='critic_loss', ax=axes[0, 1],
                     marker='o', color='green', linewidth=2)
        axes[0, 1].set_title('Critic Loss', fontsize=14)
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)

        # KL Loss
        self.df.plot(x='update', y='kl_loss', ax=axes[1, 0],
                     marker='o', color='red', linewidth=2)
        axes[1, 0].set_title('KL Divergence', fontsize=14)
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)

        # Supervision Loss
        self.df.plot(x='update', y='supervision_loss', ax=axes[1, 1],
                     marker='o', color='purple', linewidth=2)
        axes[1, 1].set_title('Supervision Loss', fontsize=14)
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)

        # LLM Loss
        self.df.plot(x='update', y='llm_loss', ax=axes[2, 0],
                     marker='o', color='orange', linewidth=2)
        axes[2, 0].set_title('LLM Loss', fontsize=14)
        axes[2, 0].set_xlabel('Update')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].grid(True)

        # Loss component stacked area
        # Create a copy of key loss components
        loss_components = self.df[['update', 'actor_loss', 'critic_loss',
                                 'kl_loss', 'supervision_loss']].copy()

        # Ensure absolute values for the stacked area chart
        for col in loss_components.columns:
            if col != 'update':
                loss_components[col] = loss_components[col].abs()

        # Plot stacked area
        loss_components.plot.area(x='update', stacked=True, ax=axes[2, 1],
                               colormap=self.plot_config['cmap'], alpha=0.7)
        axes[2, 1].set_title('Loss Components Distribution', fontsize=14)
        axes[2, 1].set_xlabel('Update')
        axes[2, 1].set_ylabel('Absolute Loss')
        axes[2, 1].grid(True)

        # Ensure x-axis shows integer ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save the figure
        filename = f'loss_components_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved loss components plot to: {filepath}")

    def _create_actor_critic_plot(self, timestamp: str, latest_update: int) -> None:
        """Create plot showing actor-critic behavior."""
        fig, axes = plt.subplots(2, 2, figsize=self.plot_config['figsize'],
                                 dpi=self.plot_config['dpi'], sharex=True)

        # Actor Loss vs Advantage
        ax1 = axes[0, 0]
        ax1.plot(self.df['update'], self.df['actor_loss'], 'bo-', label='Actor Loss')
        ax1.plot(self.df['update'], self.df['advantage'], 'go-', label='Advantage')
        ax1.set_title('Actor Loss vs Advantage', fontsize=14)
        ax1.set_ylabel('Value')
        ax1.grid(True)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.legend()

        # Critic Loss vs Value
        ax2 = axes[0, 1]
        ax2.plot(self.df['update'], self.df['critic_loss'], 'ro-', label='Critic Loss')
        ax2.plot(self.df['update'], self.df['value'], 'mo-', label='Value Estimate')
        ax2.set_title('Critic Loss vs Value Estimate', fontsize=14)
        ax2.set_ylabel('Value')
        ax2.grid(True)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend()

        # Actor Loss Sign Changes
        ax3 = axes[1, 0]
        actor_loss = self.df['actor_loss'].values
        actor_signs = np.sign(actor_loss)
        sign_changes = np.abs(np.diff(actor_signs))
        sign_changes = np.insert(sign_changes, 0, 0)  # Insert 0 at the beginning for alignment

        ax3.plot(self.df['update'], actor_loss, 'bo-', label='Actor Loss')
        ax3.set_title('Actor Loss Sign Changes', fontsize=14)
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Actor Loss')
        ax3.grid(True)

        # Highlight sign changes
        for i, (update, sign_change) in enumerate(zip(self.df['update'], sign_changes)):
            if sign_change > 0:
                ax3.axvline(x=update, color='red', linestyle='--', alpha=0.5)
                ax3.text(update, actor_loss[i], 'Sign\nChange',
                        ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Critic Loss Pattern
        ax4 = axes[1, 1]
        critic_loss = self.df['critic_loss'].values
        ax4.plot(self.df['update'], critic_loss, 'ro-', label='Critic Loss')

        # Identify high vs low pattern
        threshold = 0.5
        for i, (update, loss) in enumerate(zip(self.df['update'], critic_loss)):
            pattern = "High" if loss > threshold else "Low"
            color = "red" if pattern == "High" else "blue"
            ax4.text(update, loss + 0.05, pattern,
                    ha='center', va='bottom', fontsize=10, color=color)

        ax4.set_title('Critic Loss Pattern', fontsize=14)
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Critic Loss')
        ax4.grid(True)

        # Ensure x-axis shows integer ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save the figure
        filename = f'actor_critic_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved actor-critic plot to: {filepath}")

    def _create_reward_analysis_plot(self, timestamp: str, latest_update: int) -> None:
        """Create plot analyzing reward trends."""
        fig, axes = plt.subplots(2, 2, figsize=self.plot_config['figsize'],
                                 dpi=self.plot_config['dpi'])

        # Original Reward Trend
        ax1 = axes[0, 0]
        reward_data = self.df['reward_orig'].values
        updates = self.df['update'].values

        ax1.plot(updates, reward_data, 'ro-', linewidth=2)
        ax1.set_title('Original Reward Trend', fontsize=14)
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Reward')
        ax1.grid(True)

        # Mark best and worst rewards
        best_idx = np.argmax(reward_data)
        worst_idx = np.argmin(reward_data)

        ax1.plot(updates[best_idx], reward_data[best_idx], 'go', markersize=10)
        ax1.text(updates[best_idx], reward_data[best_idx], f'Best: {reward_data[best_idx]:.3f}',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax1.plot(updates[worst_idx], reward_data[worst_idx], 'ro', markersize=10)
        ax1.text(updates[worst_idx], reward_data[worst_idx], f'Worst: {reward_data[worst_idx]:.3f}',
                ha='center', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        # Reward Rolling Mean and Trend Line
        ax2 = axes[0, 1]

        # Plot actual rewards
        ax2.plot(updates, reward_data, 'ro-', alpha=0.5, label='Actual Reward')

        # Calculate and plot rolling mean if we have enough data
        if len(reward_data) >= 3:
            window_size = min(3, len(reward_data))
            rolling_mean = pd.Series(reward_data).rolling(window=window_size).mean().values
            ax2.plot(updates, rolling_mean, 'b-', linewidth=2, label='Rolling Mean')

            # Calculate and plot trend line
            if len(updates) >= 2:
                z = np.polyfit(updates, reward_data, 1)
                p = np.poly1d(z)
                trend_line = p(updates)
                ax2.plot(updates, trend_line, 'g--', linewidth=2, label='Trend Line')

                # Show trend slope
                trend_slope = z[0]
                trend_direction = "increasing" if trend_slope > 0 else "decreasing"
                ax2.text(0.05, 0.05, f'Trend: {trend_direction} ({trend_slope:.4f})',
                        transform=ax2.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax2.set_title('Reward Trend Analysis', fontsize=14)
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)

        # Reward Histogram
        ax3 = axes[1, 0]
        ax3.hist(reward_data, bins=min(10, len(reward_data)), alpha=0.7, color='green')
        ax3.axvline(x=np.mean(reward_data), color='r', linestyle='--',
                    label=f'Mean: {np.mean(reward_data):.3f}')
        ax3.set_title('Reward Distribution', fontsize=14)
        ax3.set_xlabel('Reward Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)

        # Reward correlation with LLM Valid Percentage
        ax4 = axes[1, 1]
        ax4.scatter(self.df['llm_valid_pct'], self.df['reward_orig'],
                   s=80, alpha=0.7, c=self.df['update'], cmap='viridis')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                 norm=plt.Normalize(self.df['update'].min(),
                                                  self.df['update'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax4)
        cbar.set_label('Update')

        # Add correlation coefficient if we have enough data
        if len(self.df) >= 3:
            corr = np.corrcoef(self.df['llm_valid_pct'], self.df['reward_orig'])[0, 1]
            correlation_text = f'Correlation: {corr:.3f}'
            ax4.text(0.05, 0.95, correlation_text, transform=ax4.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax4.set_title('Reward vs LLM Valid Percentage', fontsize=14)
        ax4.set_xlabel('LLM Valid Percentage')
        ax4.set_ylabel('Reward')
        ax4.grid(True)

        # Ensure x-axis shows integer ticks for update plots
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save the figure
        filename = f'reward_analysis_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved reward analysis plot to: {filepath}")

    def _create_llm_performance_plot(self, timestamp: str, latest_update: int) -> None:
        """Create plot analyzing LLM performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=self.plot_config['figsize'],
                                 dpi=self.plot_config['dpi'])

        # LLM Valid Percentage
        ax1 = axes[0, 0]
        valid_pct = self.df['llm_valid_pct'].values
        updates = self.df['update'].values

        ax1.plot(updates, valid_pct, 'go-', linewidth=2)
        ax1.set_title('LLM Valid Percentage', fontsize=14)
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Valid %')
        ax1.grid(True)

        # Mark best and worst valid %
        best_idx = np.argmax(valid_pct)
        worst_idx = np.argmin(valid_pct)

        ax1.plot(updates[best_idx], valid_pct[best_idx], 'go', markersize=10)
        ax1.text(updates[best_idx], valid_pct[best_idx], f'Best: {valid_pct[best_idx]:.1f}%',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax1.plot(updates[worst_idx], valid_pct[worst_idx], 'ro', markersize=10)
        ax1.text(updates[worst_idx], valid_pct[worst_idx], f'Worst: {valid_pct[worst_idx]:.1f}%',
                ha='center', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        # LLM Loss
        ax2 = axes[0, 1]
        llm_loss = self.df['llm_loss'].values

        ax2.plot(updates, llm_loss, 'bo-', linewidth=2)
        ax2.set_title('LLM Loss', fontsize=14)
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        # Calculate and plot trend line for LLM loss
        if len(updates) >= 2:
            z = np.polyfit(updates, llm_loss, 1)
            p = np.poly1d(z)
            trend_line = p(updates)
            ax2.plot(updates, trend_line, 'r--', linewidth=2, label='Trend Line')

            # Show trend slope
            trend_slope = z[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
            ax2.text(0.05, 0.95, f'Trend: {trend_direction} ({trend_slope:.4f})',
                    transform=ax2.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            ax2.legend()

        # Valid % Rolling Mean and Prediction
        ax3 = axes[1, 0]

        # Plot actual valid %
        ax3.plot(updates, valid_pct, 'go-', alpha=0.5, label='LLM Valid %')

        # Calculate and plot rolling mean if we have enough data
        if len(valid_pct) >= 3:
            window_size = min(3, len(valid_pct))
            rolling_mean = pd.Series(valid_pct).rolling(window=window_size).mean().values
            ax3.plot(updates, rolling_mean, 'b-', linewidth=2, label='Rolling Mean')

            # Add future prediction if we have enough data
            if len(updates) >= 5:
                # Use last 5 points for prediction
                last_5_updates = updates[-5:]
                last_5_valid = valid_pct[-5:]

                z = np.polyfit(last_5_updates, last_5_valid, 1)
                p = np.poly1d(z)

                # Predict next 3 updates
                future_updates = np.array([updates[-1] + i + 1 for i in range(3)])
                predictions = p(future_updates)

                # Plot prediction
                ax3.plot(future_updates, predictions, 'm--', linewidth=2, label='Prediction')
                ax3.plot(future_updates, predictions, 'mo', markersize=8)

                # Add prediction values
                for i, (update, pred) in enumerate(zip(future_updates, predictions)):
                    ax3.text(update, pred, f'{pred:.1f}%',
                           ha='center', va='bottom', fontsize=9,
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax3.set_title('LLM Valid % Trend & Prediction', fontsize=14)
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Valid %')
        ax3.legend()
        ax3.grid(True)

        # LLM Loss vs Valid % Correlation
        ax4 = axes[1, 1]
        ax4.scatter(llm_loss, valid_pct, s=80, alpha=0.7, c=updates, cmap='viridis')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(updates.min(), updates.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax4)
        cbar.set_label('Update')

        # Add correlation coefficient if we have enough data
        if len(self.df) >= 3:
            corr = np.corrcoef(llm_loss, valid_pct)[0, 1]
            correlation_text = f'Correlation: {corr:.3f}'
            ax4.text(0.05, 0.95, correlation_text, transform=ax4.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        ax4.set_title('LLM Loss vs Valid Percentage', fontsize=14)
        ax4.set_xlabel('LLM Loss')
        ax4.set_ylabel('Valid %')
        ax4.grid(True)

        # Ensure x-axis shows integer ticks for update plots
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save the figure
        filename = f'llm_performance_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved LLM performance plot to: {filepath}")

    def _1create_training_health_dashboard(self, timestamp: str, latest_update: int) -> None:
        """Create a dashboard summarizing overall training health."""
        # Calculate statistics
        stats = self.calculate_statistics()

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10), dpi=self.plot_config['dpi'])
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 1. Metrics trend summary (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['total_loss', 'llm_valid_pct', 'reward_orig']
        colors = ['blue', 'green', 'red']

        # Calculate normalized metrics for comparison
        normalized_metrics = pd.DataFrame()
        for metric in metrics:
            values = self.df[metric].values
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized_metrics[metric] = (values - min_val) / range_val

        # Plot normalized metrics
        for metric, color in zip(metrics, colors):
            ax1.plot(self.df['update'], normalized_metrics[metric],
                   f'{color}-', linewidth=2, label=metric)

        ax1.set_title('Normalized Metrics Trends', fontsize=12)
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Normalized Value')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 2. Key statistics table (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')

        # Create table data
        table_data = []
        table_data.append(['Metric', 'Min', 'Max', 'Mean', 'Latest', 'Trend'])

        for metric in metrics:
            if metric in stats:
                trend_symbol = stats.get(f"{metric}_trend", "⟷")
                improving = stats.get(f"{metric}_improving", False)
                trend_color = 'green' if improving else 'red'

                row = [
                    metric.replace('_', ' ').title(),
                    f"{stats[metric]['min']:.3f}",
                    f"{stats[metric]['max']:.3f}",
                    f"{stats[metric]['mean']:.3f}",
                    f"{stats[metric]['latest']:.3f}",
                    trend_symbol
                ]
                table_data.append(row)

        # Create table
        table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Color the trend symbols
        for i, metric in enumerate(metrics, 1):
            if metric in stats:
                improving = stats.get(f"{metric}_improving", False)
                color = 'green' if improving else 'red'
                table[(i, 5)].get_text().set_color(color)

        ax2.set_title('Key Metrics Statistics', fontsize=12)

        # 3. Training health summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        # Health score visualization
        health_score = stats.get('health_score', 0)
        max_health_score = 7
        health_colors = ['#FF0000', '#FF4000', '#FF8000', '#FFBF00', '#FFFF00', '#BFFF00', '#00FF00']

        # Create horizontal bar for health score
        for i in range(max_health_score):
            color = health_colors[i] if i < len(health_colors) else health_colors[-1]
            alpha = 1.0 if i < health_score else 0.3
            ax3.barh(0, 1, left=i, color=color, alpha=alpha, edgecolor='black')

        # Add health assessment text
        health_assessment = stats.get('training_health', 'Unknown')
        ax3.text(max_health_score/2, -0.5, f"Health: {health_assessment} ({health_score}/{max_health_score})",
                ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        # Add key observations
        observations = []

        # Add lambda annealing observation
        lambdas_annealing = stats.get('lambdas_annealing', False)
        observations.append(f"Lambda annealing: {'Active' if lambdas_annealing else 'Not yet active'}")

        # Add KL divergence observation
        kl_spikes = stats.get('kl_spikes', 0)
        observations.append(f"KL divergence spikes: {kl_spikes}")

        # Add oscillation observations
        for metric in ['total_loss', 'actor_loss', 'critic_loss', 'reward_orig']:
            if f"{metric}_oscillating" in stats and stats[f"{metric}_oscillating"]:
                observations.append(f"{metric.replace('_', ' ').title()} shows oscillation")

        # Add actor loss sign changes
        actor_sign_changes = stats.get('actor_loss_sign_changes', 0)
        observations.append(f"Actor loss sign changes: {actor_sign_changes}")

        # Add correlation observations
        if 'corr_llm_valid_reward' in stats:
            corr = stats['corr_llm_valid_reward']
            if corr < -0.3:
                observations.append(f"⚠️ Negative correlation between Valid % and Reward: {corr:.3f}")
            elif corr > 0.3:
                observations.append(f"Positive correlation between Valid % and Reward: {corr:.3f}")

        # Add reward config suggestions if available
        if 'reward_config_suggestions' in stats and stats['reward_config_suggestions']:
            for i, suggestion in enumerate(stats['reward_config_suggestions'][:2]):  # Limit to top 2 suggestions
                observations.append(f"⚠️ {suggestion}")

        # Display observations
        y_pos = 2
        ax3.text(0, y_pos, "Key Observations:", fontsize=12, fontweight='bold')
        y_pos += 0.5

        for obs in observations:
            ax3.text(0, y_pos, f"• {obs}", fontsize=10, va='center')
            y_pos += 0.5

        ax3.set_xlim(-0.5, max_health_score + 0.5)
        ax3.set_ylim(-1, y_pos)
        ax3.set_title('Training Health Assessment', fontsize=12)

        # 4. Oscillation analysis (middle left)
        ax4 = fig.add_subplot(gs[1, 0])

        # Calculate oscillation metrics
        oscillation_data = {}
        for metric in ['total_loss', 'actor_loss', 'critic_loss', 'reward_orig']:
            if len(self.df) >= 4:
                values = self.df[metric].values
                diffs = np.diff(values)
                sign_changes = np.sum(np.diff(np.signbit(diffs)) != 0)
                oscillation_data[metric] = sign_changes

        if oscillation_data:
            metrics = list(oscillation_data.keys())
            values = list(oscillation_data.values())

            y_pos = np.arange(len(metrics))
            bars = ax4.barh(y_pos, values, align='center')

            # Color bars based on threshold for concern
            for i, bar in enumerate(bars):
                if values[i] >= 2:  # Consider 2+ sign changes as oscillation
                    bar.set_color('red')
                else:
                    bar.set_color('green')

            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
            ax4.set_xlabel('Sign Changes in Differences')
            ax4.set_title('Oscillation Analysis', fontsize=12)
            ax4.grid(True, axis='x')

            # Add threshold line
            ax4.axvline(x=2, color='red', linestyle='--', alpha=0.7)
            ax4.text(2, -0.5, 'Oscillation\nThreshold', ha='center', va='top',
                    fontsize=9, color='red')

        # 5. Correlation matrix (middle center)
        ax5 = fig.add_subplot(gs[1, 1])

        # Calculate correlation matrix
        corr_metrics = ['total_loss', 'actor_loss', 'critic_loss', 'llm_loss',
                       'llm_valid_pct', 'reward_orig']
        corr_matrix = self.df[corr_metrics].corr()

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                   linewidths=0.5, ax=ax5, vmin=-1, vmax=1)

        ax5.set_title('Correlation Matrix', fontsize=12)

        # 6. Reward config analysis (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # Define default reward config weights
        reward_config = {
            'reward_overall_format_weight': 0.10,
            'reward_tag_presence_weight': 0.05,
            'think_non_repetition_weight': 0.10,
            'think_conciseness_weight': 0.10,
            'think_positive_words_weight': 0.10,
            'think_negative_words_weight': 0.05,
            'think_critical_negative_words_weight': 0.15,
            'answer_accuracy_weight': 0.35
        }

        # Group weights by category
        categories = {
            'Format': ['reward_overall_format_weight', 'reward_tag_presence_weight'],
            'Thinking': ['think_non_repetition_weight', 'think_conciseness_weight',
                       'think_positive_words_weight', 'think_negative_words_weight',
                       'think_critical_negative_words_weight'],
            'Answer': ['answer_accuracy_weight']
        }

        category_values = {}
        for category, weights in categories.items():
            category_values[category] = sum(reward_config[w] for w in weights)

        # Create pie chart for category distribution
        cat_labels = list(category_values.keys())
        cat_sizes = list(category_values.values())
        cat_colors = ['#FF9999', '#66B2FF', '#99FF99']
        cat_explode = (0.1, 0, 0)  # explode format slice

        ax6_pie = fig.add_subplot(gs[2, 0])
        wedges, texts, autotexts = ax6_pie.pie(cat_sizes, explode=cat_explode, labels=cat_labels,
                                             autopct='%1.1f%%', shadow=True, startangle=90,
                                             colors=cat_colors)

        # Make text easier to see
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')

        ax6_pie.set_title('Reward Configuration by Category', fontsize=12)

        # Suggestions for reward config improvement
        ax6.text(0, 0, "Reward Configuration Analysis:", fontsize=12, fontweight='bold')

        # Start with current config summary
        format_weight = category_values['Format']
        thinking_weight = category_values['Thinking']
        answer_weight = category_values['Answer']

        ax6.text(0, 0.5, f"Current weights: Format {format_weight:.2f}, Thinking {thinking_weight:.2f}, Answer {answer_weight:.2f}", fontsize=10)

        # Add suggestions based on data analysis
        if 'corr_llm_valid_reward' in stats:
            corr = stats['corr_llm_valid_reward']
            if corr < -0.2:
                ax6.text(0, 1.0, "⚠️ WARNING: Negative correlation between Valid % and Reward",
                        fontsize=11, color='red', fontweight='bold')
                ax6.text(0, 1.5, "Suggested changes:", fontsize=10, fontweight='bold')
                ax6.text(0, 2.0, "- Reduce think_critical_negative_words_weight: 0.15 → 0.10", fontsize=10)
                ax6.text(0, 2.5, "- Increase answer_accuracy_weight: 0.35 → 0.45", fontsize=10)
                ax6.text(0, 3.0, "- Ensure answer_accuracy calculation rewards Valid %", fontsize=10)
            elif thinking_weight > 0.45 and answer_weight < 0.4:
                ax6.text(0, 1.0, "Suggested improvements:", fontsize=10, fontweight='bold')
                ax6.text(0, 1.5, "- Rebalance weights to increase answer_accuracy_weight", fontsize=10)
                ax6.text(0, 2.0, "- Suggested ratio: Thinking 0.45, Answer 0.40, Format 0.15", fontsize=10)
            else:
                ax6.text(0, 1.0, "Current weight distribution appears reasonable.", fontsize=10)
                ax6.text(0, 1.5, "Monitor correlation between Valid % and Reward.", fontsize=10)
        else:
            ax6.text(0, 1.0, "Insufficient data to analyze reward configuration.", fontsize=10)
            ax6.text(0, 1.5, "Continue monitoring as more updates are collected.", fontsize=10)

        # 7. Reward weight recommendations (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.axis('off')

        # Based on the data analysis, provide specific reward weight recommendations
        ax7.text(0.5, 0, "Reward Weight Recommendations", fontsize=12, fontweight='bold', ha='center')

        # Define current and recommended weights
        weight_data = []

        # Start with current weights
        for key, value in reward_config.items():
            weight_data.append({
                'Component': key.replace('_weight', '').replace('_', ' ').title(),
                'Current': value,
                'Recommended': value  # Initialize with current
            })

        # Modify recommendations based on analysis
        if 'corr_llm_valid_reward' in stats and stats['corr_llm_valid_reward'] < -0.2:
            # If negative correlation between LLM Valid % and reward, adjust weights
            for item in weight_data:
                if item['Component'] == 'Think Critical Negative Words':
                    item['Recommended'] = 0.10
                elif item['Component'] == 'Answer Accuracy':
                    item['Recommended'] = 0.45

                # When reducing think_critical_negative_words_weight and increasing answer_accuracy_weight,
                # we need to adjust other weights to maintain sum = 1.0
                # This is a simple adjustment to keep the ratio of other weights unchanged
                initial_other_sum = sum(w['Current'] for w in weight_data
                                     if w['Component'] not in ['Think Critical Negative Words', 'Answer Accuracy'])
                target_other_sum = 1.0 - 0.10 - 0.45  # 1.0 - new critical weight - new accuracy weight
                adjustment_factor = target_other_sum / initial_other_sum if initial_other_sum > 0 else 1.0

                if item['Component'] not in ['Think Critical Negative Words', 'Answer Accuracy']:
                    item['Recommended'] = round(item['Current'] * adjustment_factor, 2)

        # Create a table for recommendations
        table_data = [['Component', 'Current', 'Recommended', 'Change']]
        for item in weight_data:
            change = item['Recommended'] - item['Current']
            change_str = f"{change:+.2f}" if abs(change) > 0.001 else "0.00"

            table_data.append([
                item['Component'],
                f"{item['Current']:.2f}",
                f"{item['Recommended']:.2f}",
                change_str
            ])

        # Create table
        recommendation_table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
        recommendation_table.auto_set_font_size(False)
        recommendation_table.set_fontsize(9)
        recommendation_table.scale(1, 1.5)

        # Color code changes
        for i in range(1, len(table_data)):
            change_value = float(table_data[i][3])
            if change_value > 0:
                recommendation_table[(i, 3)].get_text().set_color('green')
            elif change_value < 0:
                recommendation_table[(i, 3)].get_text().set_color('red')

        # 8. Final recommendations (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        ax8.text(0.5, 0, "Training Recommendations", fontsize=12, fontweight='bold', ha='center')

        # Generate recommendations based on the current training data
        recommendations = []

        # Reward configuration recommendations
        if 'corr_llm_valid_reward' in stats and stats['corr_llm_valid_reward'] < -0.2:
            recommendations.append("1. Adjust reward weights: reduce critical negative words penalty and increase answer accuracy reward")

        # Actor-critic recommendations
        if 'critic_loss_oscillating' in stats and stats['critic_loss_oscillating']:
            recommendations.append("2. Reduce critic learning rate to stabilize oscillations")

        # KL divergence recommendations
        if stats.get('kl_spikes', 0) > 1:
            recommendations.append("3. Decrease policy learning rate to prevent KL divergence spikes")

        # Lambda annealing recommendations
        if not stats.get('lambdas_annealing', False) and len(self.df) > 5:
            recommendations.append("4. Decrease annealing_steps to activate lambda annealing sooner")

        # LLM valid % recommendations
        if 'llm_valid_pct_improving' in stats and not stats['llm_valid_pct_improving']:
            recommendations.append("5. Review the answer evaluation logic and consider modifying the accuracy assessment")

        # Add default recommendations if none were generated
        if not recommendations:
            if stats.get('health_score', 0) >= 5:
                recommendations.append("Current training appears healthy. Continue monitoring.")
            else:
                recommendations.append("1. Collect more training data before making significant changes")
                recommendations.append("2. Monitor correlation between LLM Valid % and rewards")

        # Display recommendations
        y_pos = 1
        for rec in recommendations:
            ax8.text(0, y_pos, rec, fontsize=10, va='center')
            y_pos += 0.5

        plt.tight_layout()

        # Save the figure
        filename = f'training_health_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved training health dashboard to: {filepath}")


    def _create_training_health_dashboard(self, timestamp: str, latest_update: int) -> None:
        """Create a dashboard summarizing overall training health."""
        # Calculate statistics
        stats = self.calculate_statistics()

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10), dpi=self.plot_config['dpi'])
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 1. Metrics trend summary (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = [m for m in ['total_loss', 'llm_valid_pct', 'reward_orig'] if m in self.df.columns]
        colors = ['b', 'g', 'r']  # Use single letter colors, not color names

        # Calculate normalized metrics for comparison
        normalized_metrics = pd.DataFrame()
        for metric in metrics:
            values = self.df[metric].values
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized_metrics[metric] = (values - min_val) / range_val

        # Plot normalized metrics
        for idx, (metric, color) in enumerate(zip(metrics, colors[:len(metrics)])):
            ax1.plot(self.df['update'], normalized_metrics[metric],
                   color=color, linestyle='-', linewidth=2, label=metric)

        ax1.set_title('Normalized Metrics Trends', fontsize=12)
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Normalized Value')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 2. Key statistics table (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')

        # Create table data
        table_data = []
        table_data.append(['Metric', 'Min', 'Max', 'Mean', 'Latest', 'Trend'])

        for metric in metrics:
            if metric in stats:
                trend_symbol = stats.get(f"{metric}_trend", "⟷")
                improving = stats.get(f"{metric}_improving", False)

                row = [
                    metric.replace('_', ' ').title(),
                    f"{stats[metric]['min']:.3f}",
                    f"{stats[metric]['max']:.3f}",
                    f"{stats[metric]['mean']:.3f}",
                    f"{stats[metric]['latest']:.3f}",
                    trend_symbol
                ]
                table_data.append(row)

        # Create table
        table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Color the trend symbols
        for i, metric in enumerate(metrics, 1):
            if metric in stats:
                improving = stats.get(f"{metric}_improving", False)
                color = 'green' if improving else 'red'
                table[(i, 5)].get_text().set_color(color)

        ax2.set_title('Key Metrics Statistics', fontsize=12)

        # 3. Training health summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        # Health score visualization
        health_score = stats.get('health_score', 0)
        max_health_score = 7
        health_colors = ['#FF0000', '#FF4000', '#FF8000', '#FFBF00', '#FFFF00', '#BFFF00', '#00FF00']

        # Create horizontal bar for health score
        for i in range(max_health_score):
            color = health_colors[i] if i < len(health_colors) else health_colors[-1]
            alpha = 1.0 if i < health_score else 0.3
            ax3.barh(0, 1, left=i, color=color, alpha=alpha, edgecolor='black')

        # Add health assessment text
        health_assessment = stats.get('training_health', 'Unknown')
        ax3.text(max_health_score/2, -0.5, f"Health: {health_assessment} ({health_score}/{max_health_score})",
                ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        # Add key observations
        observations = []

        # Add lambda annealing observation
        lambdas_annealing = stats.get('lambdas_annealing', False)
        observations.append(f"Lambda annealing: {'Active' if lambdas_annealing else 'Not yet active'}")

        # Add KL divergence observation
        kl_spikes = stats.get('kl_spikes', 0)
        observations.append(f"KL divergence spikes: {kl_spikes}")

        # Add oscillation observations
        for metric in ['total_loss', 'actor_loss', 'critic_loss', 'reward_orig']:
            if f"{metric}_oscillating" in stats and stats[f"{metric}_oscillating"]:
                observations.append(f"{metric.replace('_', ' ').title()} shows oscillation")

        # Add actor loss sign changes
        actor_sign_changes = stats.get('actor_loss_sign_changes', 0)
        observations.append(f"Actor loss sign changes: {actor_sign_changes}")

        # Add correlation observations
        if 'corr_llm_valid_reward' in stats:
            corr = stats['corr_llm_valid_reward']
            if corr < -0.3:
                observations.append(f"⚠️ Negative correlation between Valid % and Reward: {corr:.3f}")
            elif corr > 0.3:
                observations.append(f"Positive correlation between Valid % and Reward: {corr:.3f}")

        # Add reward config suggestions if available
        if 'reward_config_suggestions' in stats and stats['reward_config_suggestions']:
            for i, suggestion in enumerate(stats['reward_config_suggestions'][:2]):  # Limit to top 2 suggestions
                observations.append(f"⚠️ {suggestion}")

        # Display observations
        y_pos = 2
        ax3.text(0, y_pos, "Key Observations:", fontsize=12, fontweight='bold')
        y_pos += 0.5

        for obs in observations:
            ax3.text(0, y_pos, f"• {obs}", fontsize=10, va='center')
            y_pos += 0.5

        ax3.set_xlim(-0.5, max_health_score + 0.5)
        ax3.set_ylim(-1, y_pos)
        ax3.set_title('Training Health Assessment', fontsize=12)

        # 4. Oscillation analysis (middle left)
        ax4 = fig.add_subplot(gs[1, 0])

        # Calculate oscillation metrics
        oscillation_data = {}
        for metric in ['total_loss', 'actor_loss', 'critic_loss', 'reward_orig']:
            if metric not in self.df.columns:
                continue

            if len(self.df) >= 4:
                values = self.df[metric].values
                diffs = np.diff(values)
                sign_changes = np.sum(np.diff(np.signbit(diffs)) != 0)
                oscillation_data[metric] = sign_changes

        if oscillation_data:
            metrics_list = list(oscillation_data.keys())
            values = list(oscillation_data.values())

            y_pos = np.arange(len(metrics_list))
            bars = ax4.barh(y_pos, values, align='center')

            # Color bars based on threshold for concern
            for i, bar in enumerate(bars):
                if values[i] >= 2:  # Consider 2+ sign changes as oscillation
                    bar.set_color('red')
                else:
                    bar.set_color('green')

            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([m.replace('_', ' ').title() for m in metrics_list])
            ax4.set_xlabel('Sign Changes in Differences')
            ax4.set_title('Oscillation Analysis', fontsize=12)
            ax4.grid(True, axis='x')

            # Add threshold line
            ax4.axvline(x=2, color='red', linestyle='--', alpha=0.7)
            ax4.text(2, -0.5, 'Oscillation\nThreshold', ha='center', va='top',
                    fontsize=9, color='red')

        # 5. Correlation matrix (middle center)
        ax5 = fig.add_subplot(gs[1, 1])

        # Calculate correlation matrix
        corr_metrics = [m for m in ['total_loss', 'actor_loss', 'critic_loss', 'llm_loss',
                                   'llm_valid_pct', 'reward_orig'] if m in self.df.columns]

        if len(corr_metrics) >= 2:
            corr_matrix = self.df[corr_metrics].corr()

            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                       linewidths=0.5, ax=ax5, vmin=-1, vmax=1)

            ax5.set_title('Correlation Matrix', fontsize=12)
        else:
            ax5.text(0.5, 0.5, "Insufficient data\nfor correlation matrix",
                   ha='center', va='center', fontsize=12)

        # 6. Reward config analysis (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # Define default reward config weights
        reward_config = {
            'reward_overall_format_weight': 0.10,
            'reward_tag_presence_weight': 0.05,
            'think_non_repetition_weight': 0.10,
            'think_conciseness_weight': 0.10,
            'think_positive_words_weight': 0.10,
            'think_negative_words_weight': 0.05,
            'think_critical_negative_words_weight': 0.15,
            'answer_accuracy_weight': 0.35
        }

        # Group weights by category
        categories = {
            'Format': ['reward_overall_format_weight', 'reward_tag_presence_weight'],
            'Thinking': ['think_non_repetition_weight', 'think_conciseness_weight',
                       'think_positive_words_weight', 'think_negative_words_weight',
                       'think_critical_negative_words_weight'],
            'Answer': ['answer_accuracy_weight']
        }

        category_values = {}
        for category, weights in categories.items():
            category_values[category] = sum(reward_config[w] for w in weights)

        # Create pie chart for category distribution
        cat_labels = list(category_values.keys())
        cat_sizes = list(category_values.values())
        cat_colors = ['#FF9999', '#66B2FF', '#99FF99']
        cat_explode = (0.1, 0, 0)  # explode format slice

        ax6_pie = fig.add_subplot(gs[2, 0])
        wedges, texts, autotexts = ax6_pie.pie(cat_sizes, explode=cat_explode, labels=cat_labels,
                                             autopct='%1.1f%%', shadow=True, startangle=90,
                                             colors=cat_colors)

        # Make text easier to see
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')

        ax6_pie.set_title('Reward Configuration by Category', fontsize=12)

        # Suggestions for reward config improvement
        ax6.text(0, 0, "Reward Configuration Analysis:", fontsize=12, fontweight='bold')

        # Start with current config summary
        format_weight = category_values['Format']
        thinking_weight = category_values['Thinking']
        answer_weight = category_values['Answer']

        ax6.text(0, 0.5, f"Current weights: Format {format_weight:.2f}, Thinking {thinking_weight:.2f}, Answer {answer_weight:.2f}", fontsize=10)

        # Add suggestions based on data analysis
        if 'corr_llm_valid_reward' in stats:
            corr = stats['corr_llm_valid_reward']
            if corr < -0.2:
                ax6.text(0, 1.0, "⚠️ WARNING: Negative correlation between Valid % and Reward",
                        fontsize=11, color='red', fontweight='bold')
                ax6.text(0, 1.5, "Suggested changes:", fontsize=10, fontweight='bold')
                ax6.text(0, 2.0, "- Reduce think_critical_negative_words_weight: 0.15 → 0.10", fontsize=10)
                ax6.text(0, 2.5, "- Increase answer_accuracy_weight: 0.35 → 0.45", fontsize=10)
                ax6.text(0, 3.0, "- Ensure answer_accuracy calculation rewards Valid %", fontsize=10)
            elif thinking_weight > 0.45 and answer_weight < 0.4:
                ax6.text(0, 1.0, "Suggested improvements:", fontsize=10, fontweight='bold')
                ax6.text(0, 1.5, "- Rebalance weights to increase answer_accuracy_weight", fontsize=10)
                ax6.text(0, 2.0, "- Suggested ratio: Thinking 0.45, Answer 0.40, Format 0.15", fontsize=10)
            else:
                ax6.text(0, 1.0, "Current weight distribution appears reasonable.", fontsize=10)
                ax6.text(0, 1.5, "Monitor correlation between Valid % and Reward.", fontsize=10)
        else:
            ax6.text(0, 1.0, "Insufficient data to analyze reward configuration.", fontsize=10)
            ax6.text(0, 1.5, "Continue monitoring as more updates are collected.", fontsize=10)

        # 7. Reward weight recommendations (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.axis('off')

        # Based on the data analysis, provide specific reward weight recommendations
        ax7.text(0.5, 0, "Reward Weight Recommendations", fontsize=12, fontweight='bold', ha='center')

        # Define current and recommended weights
        weight_data = []

        # Start with current weights
        for key, value in reward_config.items():
            weight_data.append({
                'Component': key.replace('_weight', '').replace('_', ' ').title(),
                'Current': value,
                'Recommended': value  # Initialize with current
            })

        # Modify recommendations based on analysis
        if 'corr_llm_valid_reward' in stats and stats['corr_llm_valid_reward'] < -0.2:
            # If negative correlation between LLM Valid % and reward, adjust weights
            for item in weight_data:
                if item['Component'] == 'Think Critical Negative Words':
                    item['Recommended'] = 0.10
                elif item['Component'] == 'Answer Accuracy':
                    item['Recommended'] = 0.45

                # When reducing think_critical_negative_words_weight and increasing answer_accuracy_weight,
                # we need to adjust other weights to maintain sum = 1.0
                # This is a simple adjustment to keep the ratio of other weights unchanged
                initial_other_sum = sum(w['Current'] for w in weight_data
                                     if w['Component'] not in ['Think Critical Negative Words', 'Answer Accuracy'])
                target_other_sum = 1.0 - 0.10 - 0.45  # 1.0 - new critical weight - new accuracy weight
                adjustment_factor = target_other_sum / initial_other_sum if initial_other_sum > 0 else 1.0

                if item['Component'] not in ['Think Critical Negative Words', 'Answer Accuracy']:
                    item['Recommended'] = round(item['Current'] * adjustment_factor, 2)

        # Create a table for recommendations
        table_data = [['Component', 'Current', 'Recommended', 'Change']]
        for item in weight_data:
            change = item['Recommended'] - item['Current']
            change_str = f"{change:+.2f}" if abs(change) > 0.001 else "0.00"

            table_data.append([
                item['Component'],
                f"{item['Current']:.2f}",
                f"{item['Recommended']:.2f}",
                change_str
            ])

        # Create table
        recommendation_table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
        recommendation_table.auto_set_font_size(False)
        recommendation_table.set_fontsize(9)
        recommendation_table.scale(1, 1.5)

        # Color code changes
        for i in range(1, len(table_data)):
            change_value = float(table_data[i][3])
            if change_value > 0:
                recommendation_table[(i, 3)].get_text().set_color('green')
            elif change_value < 0:
                recommendation_table[(i, 3)].get_text().set_color('red')

        # 8. Final recommendations (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        ax8.text(0.5, 0, "Training Recommendations", fontsize=12, fontweight='bold', ha='center')

        # Generate recommendations based on the current training data
        recommendations = []

        # Reward configuration recommendations
        if 'corr_llm_valid_reward' in stats and stats['corr_llm_valid_reward'] < -0.2:
            recommendations.append("1. Adjust reward weights: reduce critical negative words penalty and increase answer accuracy reward")

        # Actor-critic recommendations
        if 'critic_loss_oscillating' in stats and stats['critic_loss_oscillating']:
            recommendations.append("2. Reduce critic learning rate to stabilize oscillations")

        # KL divergence recommendations
        if stats.get('kl_spikes', 0) > 1:
            recommendations.append("3. Decrease policy learning rate to prevent KL divergence spikes")

        # Lambda annealing recommendations
        if not stats.get('lambdas_annealing', False) and len(self.df) > 5:
            recommendations.append("4. Decrease annealing_steps to activate lambda annealing sooner")

        # LLM valid % recommendations
        if 'llm_valid_pct_improving' in stats and not stats['llm_valid_pct_improving']:
            recommendations.append("5. Review the answer evaluation logic and consider modifying the accuracy assessment")

        # Add default recommendations if none were generated
        if not recommendations:
            if stats.get('health_score', 0) >= 5:
                recommendations.append("Current training appears healthy. Continue monitoring.")
            else:
                recommendations.append("1. Collect more training data before making significant changes")
                recommendations.append("2. Monitor correlation between LLM Valid % and rewards")

        # Display recommendations
        y_pos = 1
        for rec in recommendations:
            ax8.text(0, y_pos, rec, fontsize=10, va='center')
            y_pos += 0.5

        plt.tight_layout()

        # Save the figure
        filename = f'training_health_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved training health dashboard to: {filepath}")

    def _create_correlation_matrix_plot(self, timestamp: str, latest_update: int) -> None:
        """Create a detailed correlation matrix visualization."""
        if len(self.df) < 3:
            print("Not enough data for correlation analysis yet.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=self.plot_config['dpi'])

        # Select metrics for correlation analysis
        metrics = ['total_loss', 'actor_loss', 'critic_loss', 'kl_loss',
                  'supervision_loss', 'llm_loss', 'llm_valid_pct', 'reward_orig']

        # Calculate correlation matrix
        corr_matrix = self.df[metrics].corr()

        # Plot correlation heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                   linewidths=0.5, ax=axes[0], vmin=-1, vmax=1, annot_kws={"size": 9})

        axes[0].set_title('Metric Correlation Matrix', fontsize=14)

        # Create scatter plot for most interesting correlation
        # Focus on the correlation between LLM Valid % and Reward
        sns.scatterplot(x='llm_valid_pct', y='reward_orig', data=self.df,
                      ax=axes[1], s=100, hue='update', palette='viridis')

        # Add regression line
        sns.regplot(x='llm_valid_pct', y='reward_orig', data=self.df,
                  ax=axes[1], scatter=False, color='red')

        # Add correlation coefficient
        corr = self.df['llm_valid_pct'].corr(self.df['reward_orig'])
        axes[1].text(0.05, 0.05, f"Correlation: {corr:.3f}", transform=axes[1].transAxes,
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        # Color code based on whether correlation is negative
        if corr < 0:
            warning_txt = "WARNING: Negative correlation suggests\nreward config issue!"
            axes[1].text(0.05, 0.15, warning_txt, transform=axes[1].transAxes,
                       fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.7))

        axes[1].set_title('LLM Valid % vs. Reward', fontsize=14)
        axes[1].set_xlabel('LLM Valid Percentage')
        axes[1].set_ylabel('Original Reward')
        axes[1].legend(title='Update', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)

        plt.tight_layout()

        # Save the figure
        filename = f'correlation_matrix_update_{latest_update}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

        print(f"Saved correlation matrix plot to: {filepath}")

    def run(self) -> None:
        """
        Start monitoring the log file and generating plots.

        This method runs in an infinite loop, periodically checking for new updates
        and generating plots when new data is found.
        """
        print(f"Starting monitoring of {self.log_file_path}...")
        print(f"Press Ctrl+C to stop monitoring")

        try:
            while True:
                # Check for new updates
                new_data = self.update_data()

                # If new data was found, calculate statistics and generate plots
                if new_data and not self.df.empty:
                    stats = self.calculate_statistics()
                    print("\n=== Training Statistics ===")
                    # Print key statistics
                    if 'total_loss' in stats:
                        print(f"Total Loss: {stats['total_loss']['latest']:.4f} (avg: {stats['total_loss']['mean']:.4f})")
                    if 'llm_valid_pct' in stats:
                        print(f"LLM Valid %: {stats['llm_valid_pct']['latest']:.1f}% (avg: {stats['llm_valid_pct']['mean']:.1f}%)")
                    if 'reward_orig' in stats:
                        print(f"Reward: {stats['reward_orig']['latest']:.4f} (avg: {stats['reward_orig']['mean']:.4f})")

                    # Print health assessment
                    if 'training_health' in stats:
                        print(f"Training Health: {stats['training_health']} (Score: {stats['health_score']}/7)")

                    # Print key correlations
                    if 'corr_llm_valid_reward' in stats:
                        corr = stats['corr_llm_valid_reward']
                        corr_msg = f"Correlation between LLM Valid % and Reward: {corr:.3f}"
                        if corr < -0.2:
                            corr_msg += " ⚠️ WARNING: Negative correlation!"
                        print(corr_msg)

                    # Print reward config suggestions
                    if 'reward_config_suggestions' in stats and stats['reward_config_suggestions']:
                        print("\n=== Reward Configuration Suggestions ===")
                        for suggestion in stats['reward_config_suggestions']:
                            print(f"• {suggestion}")

                    # Generate plots
                    self.generate_plots()

                    print("\nWaiting for new updates...")

                # Wait before checking again
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"Error during monitoring: {e}")
            raise


def main():
    """Main function to parse arguments and start the monitor."""
    parser = argparse.ArgumentParser(description='LLM Training Monitor')
    parser.add_argument('log_file', type=str, help='Path to the log file to monitor')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output charts (default: create directory next to log file)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Interval in seconds between checks for updates (default: 5)')

    args = parser.parse_args()

    # Create and run the monitor
    monitor = LLMTrainingMonitor(args.log_file, args.output_dir, args.interval)
    monitor.run()


if __name__ == '__main__':
    main()
