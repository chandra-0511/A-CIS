"""Main script to run simulations and training for A-CIS cache simulator (DQN-based).

Run this script after installing dependencies from requirements.txt.
"""

from config import Config
from utils.workload_loader import WorkloadLoader
from environment.cache_env import CacheEnvironment
from agent.rl_agent import RLAgent
from utils.plotter import Plotter


def run():
    cfg = Config()

    # Load workload
    loader = WorkloadLoader()
    trace = loader.load_trace(cfg.trace_path)

    # Optionally override with a synthetic/adversarial trace for demonstration
    if getattr(cfg, 'synthetic', False):
        trace = []
        # create an adversarial trace: hotset blocks all mapping to the SAME cache set
        # this forces more evictions when hotset > associativity
        hot = cfg.adversarial_hotset if getattr(cfg, 'adversarial', False) else cfg.synthetic_hotset
        length = cfg.synthetic_length
        cache_sets = max(1, cfg.cache_size // cfg.associativity)
        target_set = 0
        # generate 'hot' distinct block numbers that all map to target_set
        hot_blocks = [ (i * cache_sets + target_set) for i in range(hot) ]
        # convert blocks to addresses (block * block_size) and repeat pattern
        blocks = []
        for i in range(length):
            b = hot_blocks[i % len(hot_blocks)]
            blocks.append(b)
        trace = [b * cfg.block_size for b in blocks]
        print(f"Using adversarial synthetic trace: length={len(trace)}, hot={hot}, target_set={target_set}, cache_sets={cache_sets}")

    # Precompute occurrence lists (block -> list of positions) for offline lookahead
    block_list = [addr // cfg.block_size for addr in trace]
    occ_map = {}
    for idx, b in enumerate(block_list):
        occ_map.setdefault(b, []).append(idx)

    # Create environment (pass occ_map for lookahead info)
    env = CacheEnvironment(cache_size=cfg.cache_size,
                           ways=cfg.associativity,
                           block_size=cfg.block_size,
                           trace=trace,
                           occ_map=occ_map,
                           trace_length=len(trace),
                           future_penalty_threshold=cfg.future_penalty_threshold,
                           future_penalty=cfg.future_penalty,
                           miss_reward=cfg.miss_reward)

    # Create agent
    agent = RLAgent(state_dim=env.state_size,
                    action_dim=env.action_size,
                    lr=cfg.lr,
                    gamma=cfg.gamma,
                    epsilon_start=cfg.epsilon_start,
                    epsilon_end=cfg.epsilon_end,
                    epsilon_decay=cfg.epsilon_decay,
                    batch_size=cfg.batch_size,
                    buffer_size=cfg.buffer_size,
                    target_update=cfg.target_update)

    hit_rates = []
    episode_rewards = []

    for ep in range(cfg.episodes):
        state = env.reset()
        total_reward = 0
        hits = 0
        steps = 0
        # iterate by index so environment can use pos for lookahead information
        for pos, addr in enumerate(trace):
            # Check if access is a hit first. Only select/store/learn on misses
            if env.is_hit(addr):
                # perform step without agent action (updates LRU)
                reward, next_state, done, info = env.step(None, addr, pos)
                state = next_state
                total_reward += reward
                hits += 1
                steps += 1
            else:
                # miss: agent must choose which way to evict
                action = agent.select_action(state)
                reward, next_state, done, info = env.step(action, addr, pos)

                # store and learn only for miss-driven decisions
                agent.store_transition(state, action, reward, next_state, done)
                agent.learn()

                state = next_state
                total_reward += reward
                steps += 1

        hit_rate = hits / steps if steps > 0 else 0.0
        hit_rates.append(hit_rate)
        episode_rewards.append(total_reward)

        print(f"Episode {ep+1}/{cfg.episodes} - Reward: {total_reward:.1f}, Hit rate: {hit_rate:.3f}, Epsilon: {agent.epsilon:.3f}")

    # Compute LRU baseline
    lru_set_assoc = env.simulate_lru(trace)
    lru_fully_assoc = env.simulate_fully_associative_lru(trace)
    lru_page = env.simulate_lru_page_cache(trace, page_size=getattr(cfg, 'page_size', 4096))

    print(f"LRU (set-assoc) hit rate: {lru_set_assoc:.3f}")
    print(f"LRU (fully-assoc) hit rate: {lru_fully_assoc:.3f}")
    print(f"LRU (page-cache, page_size={getattr(cfg, 'page_size', 4096)}) hit rate: {lru_page:.3f}")

    # Plot results with multiple baselines
    plotter = Plotter()
    baselines = {
        'LRU (set-assoc)': lru_set_assoc,
        'LRU (fully-assoc)': lru_fully_assoc,
        f'LRU (page {getattr(cfg, "page_size", 4096)})': lru_page
    }
    plotter.plot_results(hit_rates, baselines, title="DQN Cache Hit Rate vs LRU Baselines")


if __name__ == '__main__':
    run()
