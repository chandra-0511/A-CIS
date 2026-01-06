"""Quick CLI demo to compare LRU vs simple AI (farthest-next-use) without GUI.

Run from project root:
    python compare_demo.py
"""
import numpy as np
from acis.environment.cache_env import CacheEnvironment
from acis.utils.trace_generator import generate_sample_trace


def run_demo(trace_size=2000, cache_size=256, ways=4, block_size=64):
    trace = generate_sample_trace(size=trace_size, pattern_type='mixed')
    # build occ_map
    occ_map = {}
    for pos, a in enumerate(trace):
        block = int(a) // block_size
        occ_map.setdefault(block, []).append(pos)

    total = len(trace)
    lru = CacheEnvironment(cache_size=cache_size, ways=ways, block_size=block_size,
                           occ_map=occ_map, trace=trace, trace_length=total)
    agent = CacheEnvironment(cache_size=cache_size, ways=ways, block_size=block_size,
                             occ_map=occ_map, trace=trace, trace_length=total)

    lru_hits = 0
    agent_hits = 0

    for i, addr in enumerate(trace):
        # LRU
        if lru.is_hit(addr):
            lru_hits += 1
        lru.step(None, addr, pos=i)

        # Agent: on miss use farthest-next-use heuristic
        if agent.is_hit(addr):
            agent_hits += 1
            agent.step(None, addr, pos=i)
        else:
            state = agent._state_from_address(addr)
            next_use_norms = state[2::3]
            action = int(np.argmax(next_use_norms)) if len(next_use_norms) > 0 else None
            agent.step(action, addr, pos=i)

    print(f"Trace size: {total}")
    print(f"LRU hits: {lru_hits}, hit rate: {lru_hits/total*100:.2f}%")
    print(f"Agent hits: {agent_hits}, hit rate: {agent_hits/total*100:.2f}%")


if __name__ == '__main__':
    run_demo()
