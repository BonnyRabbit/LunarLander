import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1 = nn.Linear(8,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
    
    def save_checkpoint(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))  