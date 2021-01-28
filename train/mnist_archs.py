import torch.nn as nn
import torch.nn.functional as F

# F.Chollet architecture (Keras) but added two dropout layers (see Intrinsic Dimension paper)

class mnistConvNet(nn.Module):
    def __init__(self):
        super(mnistConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)  # added Dropout
        self.fc1 = nn.Linear(1600, 128)
        self.dropout2 = nn.Dropout2d(0.5)  # added Dropout
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout1(x)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def extract(self, x, verbose=False):
        out1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        out2 = F.relu(F.max_pool2d(self.conv2(out1), 2))
        t = self.dropout1(out2)   # added Dropout
        t = t.view(-1, 1600)
        out3 = F.relu(self.fc1(t))
        t = self.dropout2(out3)  # added Dropout
        t = self.fc2(t)
        out4 = F.log_softmax(t, dim=1)

        if verbose == True:
            print(out1.size())
            print(out2.size())
            print(out3.size())
            print(out4.size())

        return out1, out2, out3, out4

    def extract_all(self, x, verbose=False):
        out1 = self.conv1(x)
        out2 = F.relu(F.max_pool2d(out1, 2))
        out3 = self.conv2(out2)
        out4 = F.relu(F.max_pool2d(out3, 2))
        t = self.dropout1(out4)
        t = t.view(-1, 1600)
        out5 = F.relu(self.fc1(t))
        t = self.dropout2(out5)
        t = self.fc2(t)
        out6 = F.log_softmax(t, dim=1)

        if verbose == True:
            print(out1.size())
            print(out2.size())
            print(out3.size())
            print(out4.size())
            print(out5.size())
            print(out6.size())
        return out1, out2, out3, out4, out5, out6


class mnistConvNet5class(mnistConvNet):
    def __init__(self):
        super(mnistConvNet5class, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1600, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 5)  # 5 output layers


# class mnistConvNet2class(mnistConvNet):
#     def __init__(self):
#         super(mnistConvNet2class, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.fc1 = nn.Linear(1600, 128)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc2 = nn.Linear(128, 2)
