import torch
import torch.optim as optim
from torch.distributions import Categorical
from network import PolicyGradientNetwork

class PolicyGradientAgent:
    def __init__(self, ckpt_dir):
        self.network = PolicyGradientNetwork()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01)
        self.checkpoint_dir = ckpt_dir

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def save_models(self):
        self.network.save_checkpoint(self.checkpoint_dir)
    
    def load_models(self):
        self.network.load_checkpoint(self.checkpoint_dir)
