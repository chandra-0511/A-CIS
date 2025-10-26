import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def plot_results(self, hit_rates, baselines, title="Hit Rate"):
        """Plot agent hit rates and one or more baseline values.

        baselines: dict mapping label -> baseline_value (0..1)
        """
        episodes = list(range(1, len(hit_rates) + 1))
        # moving average window
        window = 10
        moving_avg = []
        for i in range(len(hit_rates)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(hit_rates[start_idx:i+1]))

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, hit_rates, 'b-', alpha=0.25, label='DQN Agent (raw)')
        plt.plot(episodes, moving_avg, 'b-', linewidth=2, label=f'DQN Agent ({window}-ep avg)')

        # plot each baseline line
        max_baseline = 0.0
        for label, val in (baselines or {}).items():
            plt.hlines(val, 1, max(1, len(hit_rates)), linestyles='--', linewidth=2, label=label)
            if val is not None:
                max_baseline = max(max_baseline, float(val))

        # y limit: either 1.0 or a little above max of agent or baselines
        ymax = 1.0
        if len(hit_rates) > 0:
            ymax = max(ymax, max(hit_rates) * 1.1, max_baseline * 1.1)

        plt.xlabel('Episode')
        plt.ylabel('Hit Rate')
        plt.ylim(0, min(1.0, ymax))
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
