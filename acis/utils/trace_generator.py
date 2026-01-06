import numpy as np

def generate_sample_trace(size=10000, pattern_type='mixed'):
    """Generate a sample memory access trace for testing"""
    if pattern_type == 'sequential':
        # Sequential access pattern
        return np.arange(size) * 64  # 64-byte blocks
    
    elif pattern_type == 'random':
        # Random access pattern
        return np.random.randint(0, size * 64, size)
    
    elif pattern_type == 'mixed':
        # Mix of sequential and random
        trace = []
        
        # Add sequential pattern
        trace.extend(np.arange(size//2) * 64)
        
        # Add random accesses
        trace.extend(np.random.randint(0, size * 64, size//2))
        
        # Shuffle to mix patterns
        np.random.shuffle(trace)
        return trace
    
    elif pattern_type == 'loop':
        # Loop pattern (simulating program loops)
        base_pattern = np.arange(100) * 64
        repeats = size // 100
        trace = np.tile(base_pattern, repeats)
        return trace
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

def save_trace(trace, filename):
    """Save trace to a file"""
    np.save(filename, trace)

def load_trace(filename):
    """Load trace from a file"""
    return np.load(filename)