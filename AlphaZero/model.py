import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class ResBlock(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

        self.conv1 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_hidden)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_hidden)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = F.relu(x)
        return x


class ResNet_C4(nn.Module):
    def __init__(self, board_dim, n_actions, n_res, n_hidden, device):
        super().__init__()
        self.device = device

        self.inputBlock = nn.Sequential(
            nn.Conv2d(3, n_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU()
        )

        self.resblocks = nn.ModuleList([ResBlock(n_hidden) for i in range(n_res)])

        self.valueHead = nn.Sequential(
            nn.Conv2d(n_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_dim[0] * board_dim[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(n_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_dim[0] * board_dim[1], n_actions)
        )

        self.to(device)
    
    def forward(self, x):
        x = self.inputBlock(x)
        for resblock in self.resblocks:
            x = resblock(x)
        value = self.valueHead(x)
        policy = self.policyHead(x)
        return value, policy


if __name__=="__main__":
    from torchinfo import summary
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    model = ResNet_C4((6,7), 14, 10, 64, device)
    architecture = summary(model, input_size=(16, 3, 6, 7), verbose=0)
    print(architecture)