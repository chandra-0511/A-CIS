import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['s', 'a', 'r', 's2', 'done'])

    def push(self, s, a, r, s2, done):
        self.buffer.append(self.experience(s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s = np.vstack([e.s for e in batch])
        a = np.array([e.a for e in batch])
        r = np.array([e.r for e in batch], dtype=np.float32)
        s2 = np.vstack([e.s2 for e in batch])
        done = np.array([e.done for e in batch], dtype=np.uint8)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1e-4,
                 batch_size=64, buffer_size=10000, target_update=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_step = 0
        self.target_update = target_update

    def select_action(self, state):
        # state: numpy array
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            # decay epsilon per step
            self._decay_epsilon()
            return action
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(state_v)
        action = int(q.argmax(dim=1).item())
        # decay epsilon per step
        self._decay_epsilon()
        return action

    def store_transition(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def learn(self):
        if len(self.buffer) < max(32, self.batch_size // 2):
            # wait until buffer has some samples
            return

        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s_v = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_v = torch.tensor(a, dtype=torch.long, device=self.device)
        r_v = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2_v = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_mask = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_vals = self.policy_net(s_v)
        q_val = q_vals.gather(1, a_v.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(s2_v)
            q_next_max = q_next.max(1)[0]
            q_target = r_v + self.gamma * q_next_max * (1.0 - done_mask)

        loss = self.loss_fn(q_val, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def _decay_epsilon(self):
        # Linear decay towards epsilon_end
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
