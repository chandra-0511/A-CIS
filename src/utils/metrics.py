import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter

class MetricsTracker:
    """Track and visualize cache performance metrics."""
    
    def __init__(self, log_dir: str = 'runs'):
        """Initialize metrics tracker.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)
        self.reset()
        
    def reset(self):
        """Reset accumulated metrics."""
        self.episode_rewards = []
        self.hit_rates = []
        self.energy_rates = []
        self.latencies = []
        self.mtta_records = []
        self.phg_values = []
        
    def update(self, 
               episode: int,
               rewards: List[float],
               metrics: Dict[str, float]):
        """Update metrics for one episode."""
        # Store episode metrics
        self.episode_rewards.append(np.mean(rewards))
        self.hit_rates.append(metrics['hit_rate'])
        self.energy_rates.append(metrics['energy_consumed'])
        self.latencies.append(metrics['avg_latency'])
        
        # Log to TensorBoard
        self.writer.add_scalar('Reward/episode', np.mean(rewards), episode)
        self.writer.add_scalar('Performance/hit_rate', metrics['hit_rate'], episode)
        self.writer.add_scalar('Performance/energy', metrics['energy_consumed'], episode)
        self.writer.add_scalar('Performance/latency', metrics['avg_latency'], episode)
        
    def compute_mtta(self, 
                    pre_change_rate: float,
                    recovery_threshold: float = 0.9
                   ) -> int:
        """Compute Mean Time to Adaptation after workload change.
        
        Args:
            pre_change_rate: Hit rate before workload change
            recovery_threshold: Fraction of original rate to consider recovered
            
        Returns:
            Number of episodes to recover
        """
        target = pre_change_rate * recovery_threshold
        for i, rate in enumerate(self.hit_rates):
            if rate >= target:
                self.mtta_records.append(i)
                return i
        return len(self.hit_rates)
    
    def compute_phg(self,
                   baseline_rates: List[float],
                   prediction_rates: List[float]
                  ) -> float:
        """Compute Prediction-aided Hit Gain.
        
        Args:
            baseline_rates: Hit rates without prediction
            prediction_rates: Hit rates with prediction
            
        Returns:
            Relative improvement in hit rate
        """
        if not baseline_rates or not prediction_rates:
            return 0.0
            
        baseline_avg = np.mean(baseline_rates)
        prediction_avg = np.mean(prediction_rates)
        
        if baseline_avg == 0:
            return float('inf') if prediction_avg > 0 else 0.0
            
        gain = (prediction_avg - baseline_avg) / baseline_avg
        self.phg_values.append(gain)
        return gain
    
    def plot_training_curves(self, 
                           save_path: str = None,
                           show: bool = True):
        """Plot training metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards)
        ax1.set_title('Average Episode Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Hit rate
        ax2.plot(episodes, self.hit_rates)
        ax2.set_title('Cache Hit Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Hit Rate')
        ax2.grid(True)
        
        # Energy consumption
        ax3.plot(episodes, self.energy_rates)
        ax3.set_title('Energy Consumption')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Energy Units')
        ax3.grid(True)
        
        # Average latency
        ax4.plot(episodes, self.latencies)
        ax4.set_title('Average Latency')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Latency Units')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_pareto_front(self,
                         save_path: str = None,
                         show: bool = True):
        """Plot Pareto front of energy vs performance."""
        # Compute CPI (cycles per instruction) and EPI (energy per instruction)
        cpi = np.array(self.latencies)
        epi = np.array(self.energy_rates)
        
        # Find Pareto optimal points
        pareto_points = []
        for i in range(len(cpi)):
            dominated = False
            for j in range(len(cpi)):
                if i != j:
                    if cpi[j] <= cpi[i] and epi[j] <= epi[i]:
                        if cpi[j] < cpi[i] or epi[j] < epi[i]:
                            dominated = True
                            break
            if not dominated:
                pareto_points.append((cpi[i], epi[i]))
                
        # Sort points for plotting
        pareto_points.sort()
        pareto_x, pareto_y = zip(*pareto_points)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(cpi, epi, alpha=0.5, label='Operating Points')
        plt.plot(pareto_x, pareto_y, 'r-', label='Pareto Front')
        plt.title('Energy-Performance Trade-off')
        plt.xlabel('Cycles Per Instruction (CPI)')
        plt.ylabel('Energy Per Instruction (EPI)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
            
    def close(self):
        """Clean up resources."""
        self.writer.close()