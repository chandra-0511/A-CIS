import random
import numpy as np


class CacheEnvironment:
    """A simple set-associative cache environment.

    - cache_size: number of blocks total (used to derive number of sets)
    - ways: associativity (number of ways per set)
    - block_size: in bytes (affects tag computation)

    State representation: For the current address, we expose the set's tags as a fixed-size vector of length `ways`.
    Each entry is normalized to [0,1] using modulo and division to keep values small.

    Action space: integer in [0, ways-1] indicating which way to evict on a miss.
    """

    def __init__(self, cache_size=1024, ways=4, block_size=64, trace=None,
                 occ_map=None, trace_length=None,
                 future_penalty_threshold=8, future_penalty=1.0,
                 miss_reward=0.0):
        self.cache_size = cache_size
        self.ways = ways
        self.block_size = block_size
        self.trace = trace or []
        self.occ_map = occ_map or {}
        self.trace_length = trace_length or (len(self.trace) if self.trace is not None else 0)
        self.future_penalty_threshold = future_penalty_threshold
        self.future_penalty = future_penalty
        self.miss_reward = miss_reward

        # derive number of sets from cache_size and ways
        self.num_sets = max(1, cache_size // ways)
        # state/action sizes for agent wiring
        # state will include per-way (tag_norm, age_norm, next_use_norm) => 3 values per way
        self.state_size = self.ways * 3
        self.action_size = self.ways

        # cache data structure: list of sets, each set is list of ways (tag or None)
        self.reset()

    def reset(self):
        # empty cache: each set contains tuples (tag, lru_counter)
        self.sets = [ [(None, 0) for _ in range(self.ways)] for _ in range(self.num_sets) ]
        self.time = 0
        self.pos = 0
        return self._state_from_address(0)  # initial dummy state

    def _addr_to_set_tag(self, addr):
        block = addr // self.block_size
        set_idx = block % self.num_sets
        tag = block // self.num_sets
        return set_idx, tag

    def _state_from_address(self, addr):
        """Return state vector for the set containing addr.

        Per-way features: (tag_norm, age_norm, next_use_norm)
        - tag_norm: normalized tag or -1 for empty
        - age_norm: recency normalized to [0,1]
        - next_use_norm: normalized distance to next use (0 = immediate, 1 = far/never)
        """
        set_idx, tag = self._addr_to_set_tag(addr)
        ways = self.sets[set_idx]
        vec = []
        current = max(1, self.time)
        # current position used for lookahead
        pos = getattr(self, 'pos', 0)
        for (t, lru) in ways:
            if t is None:
                tag_norm = -1.0
                age_norm = 1.0
                next_use_norm = 1.0
            else:
                tag_norm = (t % 1024) / 1024.0
                age = current - lru
                age_norm = min(1.0, age / current)
                # compute next use distance (in positions) for this cached block
                # reconstruct block number from tag and set_idx
                block_num = int(t * self.num_sets + set_idx)
                occs = self.occ_map.get(block_num, [])
                # binary search for first occ > pos
                next_pos = None
                for o in occs:
                    if o > pos:
                        next_pos = o
                        break
                if next_pos is None:
                    next_use_norm = 1.0
                else:
                    dist = next_pos - pos
                    # normalize by trace_length
                    next_use_norm = min(1.0, dist / max(1, self.trace_length))
            vec.append(tag_norm)
            vec.append(age_norm)
            vec.append(next_use_norm)
        return np.array(vec, dtype=np.float32)

    def is_hit(self, addr):
        """Return True if the given address would hit in the current cache state."""
        set_idx, tag = self._addr_to_set_tag(addr)
        ways = self.sets[set_idx]
        for (t, lru) in ways:
            if t == tag:
                return True
        return False

    def step(self, action, addr, pos=None):
        """Apply agent's action for the incoming address.

        action: chosen way index to evict on a miss (0..ways-1)
        addr: memory address (int)

        Returns: reward, next_state, done, info
        """
        # update time and position
        self.time += 1
        if pos is not None:
            self.pos = pos
        set_idx, tag = self._addr_to_set_tag(addr)
        ways = self.sets[set_idx]

        # check for hit
        for i, (t, lru) in enumerate(ways):
            if t == tag:
                # hit: update LRU counters
                ways[i] = (t, self.time)
                next_state = self._state_from_address(addr)
                return 1.0, next_state, False, {'hit': True}

        # miss: use agent action to select eviction
        # if there's empty way, prefer that
        empty_idx = None
        for i, (t, lru) in enumerate(ways):
            if t is None:
                empty_idx = i
                break

        if empty_idx is not None:
            use_idx = empty_idx
            evicted_block = None
        else:
            # if no empty and no action given, evict the LRU way (oldest lru)
            if action is None:
                # choose way with smallest lru (oldest)
                lru_vals = [l for (t, l) in ways]
                use_idx = int(np.argmin(lru_vals))
            else:
                # use the agent-selected action (wrap if out of range)
                use_idx = int(action % self.ways)
            # compute evicted block number for reward shaping
            ev_tag, ev_lru = ways[use_idx]
            if ev_tag is None:
                evicted_block = None
            else:
                evicted_block = int(ev_tag * self.num_sets + set_idx)

        # place the new tag
        ways[use_idx] = (tag, self.time)

        # reward shaping: hit handled earlier; on miss return miss_reward but penalize evicting soon-to-be-used blocks
        reward = float(self.miss_reward)
        info = {'hit': False, 'evicted_way': use_idx}
        if evicted_block is not None:
            # determine next use distance for evicted block from current pos
            occs = self.occ_map.get(evicted_block, [])
            next_pos = None
            pos_now = getattr(self, 'pos', 0)
            for o in occs:
                if o > pos_now:
                    next_pos = o
                    break
            if next_pos is not None:
                dist = next_pos - pos_now
                if dist <= self.future_penalty_threshold:
                    reward = -float(self.future_penalty)

        next_state = self._state_from_address(addr)
        return reward, next_state, False, info

    def simulate_lru(self, trace):
        """Simulate LRU policy over the given trace and return overall hit rate."""
        # reset an LRU-only cache
        sets = [ [] for _ in range(self.num_sets) ]  # each is list of tags, front = most recent
        hits = 0
        for addr in trace:
            set_idx, tag = self._addr_to_set_tag(addr)
            s = sets[set_idx]
            found = False
            for i, t in enumerate(s):
                if t == tag:
                    # hit: move to front
                    s.pop(i)
                    s.insert(0, tag)
                    hits += 1
                    found = True
                    break
            if not found:
                # miss: insert at front, evict last if needed
                s.insert(0, tag)
                if len(s) > self.ways:
                    s.pop()
        total = len(trace)
        return hits / total if total > 0 else 0.0

    def simulate_fully_associative_lru(self, trace):
        """Simulate a fully-associative LRU cache (no sets). Capacity = number of blocks (self.cache_size).

        This avoids set collisions entirely and gives a best-case LRU baseline when only capacity matters.
        """
        capacity = max(1, int(self.cache_size))
        # maintain list of block ids (front = most recent)
        lru = []
        hits = 0
        for addr in trace:
            block = addr // self.block_size
            if block in lru:
                # hit: move to front
                lru.remove(block)
                lru.insert(0, block)
                hits += 1
            else:
                # miss: insert and evict if needed
                lru.insert(0, block)
                if len(lru) > capacity:
                    lru.pop()
        total = len(trace)
        return hits / total if total > 0 else 0.0

    def simulate_lru_page_cache(self, trace, page_size=4096):
        """Simulate an LRU page-cache (coarser granularity) and return hit rate.

        - page_size: bytes per page (default 4096)
        - capacity (pages) is derived from total cache bytes = cache_size * block_size
        """
        total_cache_bytes = int(self.cache_size) * int(self.block_size)
        capacity_pages = max(1, total_cache_bytes // int(page_size))
        lru = []  # list of page_ids, front = most recent
        hits = 0
        for addr in trace:
            page_id = addr // page_size
            if page_id in lru:
                lru.remove(page_id)
                lru.insert(0, page_id)
                hits += 1
            else:
                lru.insert(0, page_id)
                if len(lru) > capacity_pages:
                    lru.pop()
        total = len(trace)
        return hits / total if total > 0 else 0.0
