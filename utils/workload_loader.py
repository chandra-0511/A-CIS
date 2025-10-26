class WorkloadLoader:
    def load_trace(self, path):
        # simple loader: each non-empty line contains an integer address
        addrs = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    try:
                        addrs.append(int(s, 0))
                    except ValueError:
                        # try hex like 0x...
                        try:
                            addrs.append(int(s, 16))
                        except Exception:
                            continue
        except FileNotFoundError:
            raise
        return addrs
