class Config:
    # Cache configuration
    cache_size = 1024          # total number of blocks the cache can hold (conceptual)
    associativity = 4          # number of ways per set
    block_size = 64            # bytes per block
    page_size = 4096           # bytes per page for page-cache baseline

    # RL / training
    lr = 1e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    # epsilon decay per step (larger value for faster demo decay)
    epsilon_decay = 1e-3
    batch_size = 64
    buffer_size = 10000
    target_update = 100        # steps between target network updates

    episodes = 20
    # Training/trace options
    episodes = 200

    # Use synthetic trace generation for demo experiments when True
    synthetic = True
    synthetic_length = 500     # number of accesses in the trace
    synthetic_hotset = 8       # size of the hot working set
    synthetic_stride = 16      # stride to generate conflicts

    # Trace
    trace_path = 'traces/sample_trace.txt'
    # Adversarial / reward shaping
    adversarial = True
    # If True, hotset will be greater than associativity to force evictions
    adversarial_hotset = 16
    # reward shaping: penalty when evicting a block that will be used within this many steps
    future_penalty_threshold = 8
    future_penalty = 1.0
    # Miss base reward (0 = no reward for miss)
    miss_reward = 0.0
