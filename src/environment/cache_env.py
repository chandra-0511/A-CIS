import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional

class CacheEnvironment(gym.Env):
    """Advanced Cache Intelligence Simulator Environment.
    
    Implements a set-associative cache with:
    - Energy-aware state and reward modeling
    - Dual-agent support (admission and replacement)
    - LSTM-based access pattern prediction
    - Hardware constraints and multi-objective optimization
    """
    
    def __init__(self, 
                 cache_size: int = 1024,
                 ways: int = 4,
                 block_size: int = 64,
                 trace: Optional[List[int]] = None,
                 occ_map: Optional[Dict[int, List[int]]] = None,
                 energy_weight: float = 0.3,
                 latency_weight: float = 0.2,
                 enable_prediction: bool = True):
        """Initialize cache environment with energy and latency awareness.
        
        Args:
            cache_size: Total number of blocks in cache
            ways: Number of ways per set (associativity)
            block_size: Size of each block in bytes
            trace: Optional memory access trace
            occ_map: Optional block->positions mapping for lookahead
            energy_weight: Weight of energy cost in reward (α)
            latency_weight: Weight of latency in reward (β)
            enable_prediction: Whether to use LSTM prediction
        """
        super().__init__()
        
        # Cache parameters
        self.cache_size = cache_size
        self.ways = ways
        self.block_size = block_size
        self.num_sets = max(1, cache_size // ways)
        
        # Workload data
        self.trace = trace or []
        self.occ_map = occ_map or {}
        self.trace_length = len(self.trace) if trace is not None else 0
        
        # Reward weighting
        self.energy_weight = energy_weight
        self.latency_weight = latency_weight
        self.enable_prediction = enable_prediction
        
        # State/action spaces
        # State: [tag_norm, age_norm, next_use_norm, energy_norm, pred_reuse] per way
        self.features_per_way = 5
        self.state_size = self.ways * self.features_per_way
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.state_size,), 
            dtype=np.float32
        )
        
        # Actions: way index to evict (replacement) or whether to admit (admission)
        self.action_space = spaces.Discrete(self.ways)
        
        # Runtime state
        self.reset()
        
        # Performance metrics
        self.hits = 0
        self.accesses = 0
        self.energy_consumed = 0.0
        self.total_latency = 0.0
        
    def reset(self):
        """Reset cache state and metrics."""
        # Initialize empty cache: (tag, lru_counter, energy_cost)
        self.sets = [[(None, 0, 0.0) for _ in range(self.ways)] 
                    for _ in range(self.num_sets)]
        
        self.time = 0
        self.pos = 0
        self.hits = 0
        self.accesses = 0
        self.energy_consumed = 0.0
        self.total_latency = 0.0
        
        # Return initial state
        return self._state_from_address(0)
    
    def _addr_to_set_tag(self, addr: int) -> Tuple[int, int]:
        """Convert address to (set_index, tag)."""
        block = addr // self.block_size
        set_idx = block % self.num_sets
        tag = block // self.num_sets
        return set_idx, tag
    
    def _compute_energy_cost(self, is_hit: bool, way_index: int) -> float:
        """Compute energy cost for cache operation."""
        # Simple energy model (can be enhanced)
        if is_hit:
            return 1.0  # Base hit energy
        return 3.0 + 0.5 * way_index  # Miss energy + way-dependent cost
    
    def _compute_latency(self, is_hit: bool) -> float:
        """Compute latency for cache operation."""
        return 1.0 if is_hit else 10.0  # Simple hit/miss latency model
    
    def _predict_reuse(self, block_num: int, current_pos: int) -> float:
        """Predict probability of near-future reuse (placeholder for LSTM)."""
        # For now, use simple next-use distance, later replace with LSTM
        occs = self.occ_map.get(block_num, [])
        next_pos = None
        for o in occs:
            if o > current_pos:
                next_pos = o
                break
        if next_pos is None:
            return 0.0  # No predicted reuse
        dist = next_pos - current_pos
        return 1.0 - min(1.0, dist / max(1, self.trace_length))
    
    def _state_from_address(self, addr: int) -> np.ndarray:
        """Generate state vector for current address.
        
        State features per way:
        - tag_norm: Normalized tag value
        - age_norm: Normalized recency (LRU timestamp)
        - next_use_norm: Normalized distance to next use
        - energy_norm: Normalized cumulative energy cost
        - pred_reuse: Predicted reuse probability
        """
        set_idx, tag = self._addr_to_set_tag(addr)
        ways = self.sets[set_idx]
        vec = []
        
        current = max(1, self.time)
        block_num = addr // self.block_size
        
        for t, lru, energy in ways:
            if t is None:
                # Empty way
                vec.extend([-1.0, 1.0, 1.0, 0.0, 0.0])
            else:
                # Compute normalized features
                tag_norm = (t % 1024) / 1024.0
                age = current - lru
                age_norm = min(1.0, age / current)
                
                # Next use distance
                next_use_norm = 1.0
                way_block = int(t * self.num_sets + set_idx)
                occs = self.occ_map.get(way_block, [])
                next_pos = None
                pos_now = getattr(self, 'pos', 0)
                for o in occs:
                    if o > pos_now:
                        next_pos = o
                        break
                if next_pos is not None:
                    dist = next_pos - pos_now
                    next_use_norm = min(1.0, dist / max(1, self.trace_length))
                
                # Energy normalization (simple linear scale)
                energy_norm = min(1.0, energy / 100.0)
                
                # Prediction (if enabled)
                pred_reuse = self._predict_reuse(way_block, pos_now) if self.enable_prediction else 0.0
                
                vec.extend([tag_norm, age_norm, next_use_norm, energy_norm, pred_reuse])
        
        return np.array(vec, dtype=np.float32)
    
    def step(self, action: int, addr: int, pos: Optional[int] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one cache access step.
        
        Args:
            action: Way to evict (if replacement needed)
            addr: Memory address to access
            pos: Current position in trace (for prediction)
        
        Returns:
            (next_state, reward, done, info)
        """
        self.time += 1
        self.accesses += 1
        if pos is not None:
            self.pos = pos
            
        set_idx, tag = self._addr_to_set_tag(addr)
        ways = self.sets[set_idx]
        
        # Check for hit
        hit_way = None
        for i, (t, lru, energy) in enumerate(ways):
            if t == tag:
                hit_way = i
                break
                
        if hit_way is not None:
            # Hit: update LRU and energy
            self.hits += 1
            t, _, e = ways[hit_way]
            energy_cost = self._compute_energy_cost(True, hit_way)
            latency = self._compute_latency(True)
            ways[hit_way] = (t, self.time, e + energy_cost)
            
            self.energy_consumed += energy_cost
            self.total_latency += latency
            
            # Multi-objective reward
            reward = 1.0 - self.energy_weight * energy_cost - self.latency_weight * latency
            
            next_state = self._state_from_address(addr)
            return next_state, reward, False, {'hit': True, 'energy': energy_cost}
            
        # Miss: handle replacement
        empty_idx = None
        for i, (t, _, _) in enumerate(ways):
            if t is None:
                empty_idx = i
                break
                
        # Choose way for replacement
        if empty_idx is not None:
            use_idx = empty_idx
            evicted_block = None
        else:
            use_idx = action % self.ways
            t, _, _ = ways[use_idx]
            evicted_block = int(t * self.num_sets + set_idx) if t is not None else None
            
        # Place new block
        energy_cost = self._compute_energy_cost(False, use_idx)
        latency = self._compute_latency(False)
        ways[use_idx] = (tag, self.time, energy_cost)
        
        self.energy_consumed += energy_cost
        self.total_latency += latency
        
        # Multi-objective reward with prediction penalty
        reward = -self.energy_weight * energy_cost - self.latency_weight * latency
        
        if evicted_block is not None and self.enable_prediction:
            # Add prediction-based penalty
            pred_reuse = self._predict_reuse(evicted_block, self.pos)
            reward -= pred_reuse  # Penalty for evicting likely-to-be-used block
            
        next_state = self._state_from_address(addr)
        return next_state, reward, False, {
            'hit': False,
            'energy': energy_cost,
            'evicted': evicted_block
        }
    
    def get_metrics(self) -> dict:
        """Return current performance metrics."""
        hit_rate = self.hits / max(1, self.accesses)
        return {
            'hit_rate': hit_rate,
            'energy_consumed': self.energy_consumed,
            'avg_latency': self.total_latency / max(1, self.accesses),
            'accesses': self.accesses
        }
    
    # Baseline policies for comparison
    def simulate_lru(self, trace: List[int]) -> Dict[str, float]:
        """Simulate LRU policy and return metrics."""
        self.reset()
        for addr in trace:
            set_idx, tag = self._addr_to_set_tag(addr)
            s = self.sets[set_idx]
            found = False
            for i, (t, _, _) in enumerate(s):
                if t == tag:
                    # Hit: move to front
                    energy = self._compute_energy_cost(True, i)
                    latency = self._compute_latency(True)
                    s[i] = (t, self.time, energy)
                    self.hits += 1
                    self.energy_consumed += energy
                    self.total_latency += latency
                    found = True
                    break
            if not found:
                # Miss: insert at front, evict LRU
                lru_idx = min(range(len(s)), key=lambda i: s[i][1])
                energy = self._compute_energy_cost(False, lru_idx)
                latency = self._compute_latency(False)
                s[lru_idx] = (tag, self.time, energy)
                self.energy_consumed += energy
                self.total_latency += latency
            self.time += 1
            self.accesses += 1
            
        return self.get_metrics()