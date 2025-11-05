import os
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR, TOWN7


class DuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DuelingDQnetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR + '/' + TOWN7, model)

        self.Linear1 = nn.Sequential(
            nn.Linear(95 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.V = nn.Linear(64, 1)
        self.A = nn.Linear(64, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        fc = self.Linear1(x)
        V = self.V(fc)
        A = self.A(fc)
        return V, A

    def save_checkpoint(self):
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        # Tạo thư mục (và các thư mục cha) nếu nó chưa tồn tại
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

