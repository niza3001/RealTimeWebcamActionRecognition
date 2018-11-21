from torch import nn
from non_local_simple_version import NONLocalBlock2D
from torchsummary import summary
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NLN_CNN(nn.Module):
    def __init__(self):
        super(NLN_CNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=2048, out_features=101)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(1, 3, 224, 224))
    net = NLN_CNN().cpu()
    output = net.forward(img)
    print output.shape
    # pass
    # summary(net, (3, 224, 244))

