import torch
import torch.nn as nn
from collections import OrderedDict

class ECGNet(nn.Module):
  def __init__(self, embedding_size=264, dropout=False, num_layers=2):
    super(ECGNet, self).__init__()


    self.num_layers = num_layers
    self.dropout = dropout

    #layer1
    self.layer1_conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 25), stride=(1, 2), bias=True)


    #layer2
    self.layer2_conv2d = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(1, 15), stride=(1, 1), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(1, 15), stride=(1, 2),  bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(1, 15), stride=(1, 1), bias=True)),
    ]))
    self.layer2_seModule = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    #layer3
    self.layer3_conv2d_block1 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True)),
    ]))
    self.layer3_seModule_block1 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer3_conv2d_block2 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0), bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(5, 1), padding=(2, 0), bias=True)),
    ]))
    self.layer3_seModule_block2 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer3_conv2d_block3 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0), bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(7, 1), padding=(3, 0), bias=True)),
    ]))
    self.layer3_seModule_block3 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    #layer4
    self.layer4_conv1d_short_block1 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 384, kernel_size=3, stride=9, bias=True)),
    ]))

    self.layer4_conv1d_block1 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 768, kernel_size=3, stride=2, bias=True)),
        ("bn2", nn.BatchNorm1d(num_features=768)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv1d(768, 768, kernel_size=3, stride=1, bias=True)),
        ("bn3", nn.BatchNorm1d(num_features=768)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv1d(768, 1536, kernel_size=3, stride=2, bias=True)),
        ("bn4", nn.BatchNorm1d(num_features=1536)),
        ("act4", nn.ReLU()),
        ("cn4", nn.Conv1d(1536, 384, kernel_size=3, stride=2, bias=True)),
    ]))
    self.layer4_seModule_block1 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv1d(384, 48, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv1d(48, 384, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer4_conv1d_short_block2 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 384, kernel_size=5, stride=9, bias=True)),
    ]))

    self.layer4_conv1d_block2 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 768, kernel_size=5, stride=2, padding=2, bias=True)),
        ("bn2", nn.BatchNorm1d(num_features=768)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv1d(768, 768, kernel_size=5, stride=2, padding=1, bias=True)),
        ("bn3", nn.BatchNorm1d(num_features=768)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv1d(768, 1536, kernel_size=5, stride=1, padding=2, bias=True)),
        ("bn4", nn.BatchNorm1d(num_features=1536)),
        ("act4", nn.ReLU()),
        ("cn4", nn.Conv1d(1536, 384, kernel_size=5, stride=2, padding=1, bias=True)),
    ]))
    self.layer4_seModule_block2 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv1d(384, 48, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv1d(48, 384, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer4_conv1d_short_block3 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 384, kernel_size=7, stride=9, bias=True)),
    ]))

    self.layer4_conv1d_block3 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 768, kernel_size=7, stride=2, padding=2, bias=True)),
        ("bn2", nn.BatchNorm1d(num_features=768)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv1d(768, 768, kernel_size=7, stride=2, padding=1, bias=True)),
        ("bn3", nn.BatchNorm1d(num_features=768)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv1d(768, 1536, kernel_size=7, stride=1, padding=3, bias=True)),
        ("bn4", nn.BatchNorm1d(num_features=1536)),
        ("act4", nn.ReLU()),
        ("cn4", nn.Conv1d(1536, 384, kernel_size=7, stride=2, padding=2, bias=True)),
    ]))
    self.layer4_seModule_block3 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv1d(384, 48, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv1d(48, 384, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer5_avg_pool1 = nn.AvgPool1d(kernel_size=10)
    self.layer5_avg_pool2 = nn.AvgPool1d(kernel_size=10)
    self.layer5_avg_pool3 = nn.AvgPool1d(kernel_size=10)

    cur_hidden_dim = 1152
    fc_layres = []
    for i in range(num_layers - 1):
      fc_layres.append((f"ln{i+1}", nn.Linear(cur_hidden_dim, embedding_size)))
      cur_hidden_dim = embedding_size
      fc_layres.append((f"act{i+1}", nn.ReLU()))
      if dropout and i % 2 == 0:
        fc_layres.append((f"dp{i // 2}", nn.Dropout(p=dropout)))

    fc_layres.append((f"ln{num_layers}", nn.Linear(cur_hidden_dim, 1)))
    fc_layres.append((f"sigmoid", nn.Sigmoid()))

    self.fc = nn.Sequential(OrderedDict(fc_layres))
  def forward(self, x):
    #layer1
    x = self.layer1_conv2d(x)

    #layer2
    x = self.layer2_conv2d(x)
    u = x
    x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
    x = self.layer2_seModule(x)
    x = u * x

    #layer3
    x1 = self.layer3_conv2d_block1(x)
    u1 = x1
    x1 = x1.view(x1.size(0), x1.size(1), -1).mean(-1).view(x1.size(0), x1.size(1), 1, 1)
    x1 = self.layer3_seModule_block1(x1)
    x1 = u1 * x1

    x2 = self.layer3_conv2d_block2(x)
    u2 = x2
    x2 = x2.view(x2.size(0), x2.size(1), -1).mean(-1).view(x2.size(0), x2.size(1), 1, 1)
    x2 = self.layer3_seModule_block2(x2)
    x2 = u2 * x2

    x3 = self.layer3_conv2d_block3(x)
    u3 = x3
    x3 = x3.view(x3.size(0), x3.size(1), -1).mean(-1).view(x3.size(0), x3.size(1), 1, 1)
    x3 = self.layer3_seModule_block3(x3)
    x3 = u3 * x3

    #layer4
    x1 = torch.flatten(x1, start_dim=1, end_dim=2)
    x2 = torch.flatten(x2, start_dim=1, end_dim=2)
    x3 = torch.flatten(x3, start_dim=1, end_dim=2)

    x1_short = self.layer4_conv1d_short_block1(x1)

    x1 = self.layer4_conv1d_block1(x1)
    u1 = x1
    x1 = x1.view(x1.size(0), x1.size(1), -1).mean(-1).view(x1.size(0), x1.size(1), 1, 1).flatten(2, 3)
    x1 = self.layer4_seModule_block1(x1)
    x1 = u1 * x1
    x1 = x1 + x1_short

    x2_short = self.layer4_conv1d_short_block2(x2)

    x2 = self.layer4_conv1d_block2(x2)
    u2 = x2
    x2 = x2.view(x2.size(0), x2.size(1), -1).mean(-1).view(x2.size(0), x2.size(1), 1, 1).flatten(2, 3)
    x2 = self.layer4_seModule_block2(x2)
    x2 = u2 * x2
    x2 = x2 + x2_short

    x3_short = self.layer4_conv1d_short_block3(x3)

    x3 = self.layer4_conv1d_block3(x3)
    u3 = x3
    x3 = x3.view(x3.size(0), x3.size(1), -1).mean(-1).view(x3.size(0), x3.size(1), 1, 1).flatten(2, 3)
    x3 = self.layer4_seModule_block3(x3)
    x3 = u3 * x3
    x3 = x3 + x3_short

    x1 = self.layer5_avg_pool1(x1)
    x2 = self.layer5_avg_pool2(x2)
    x3 = self.layer5_avg_pool3(x3)

    x = torch.cat((x1, x2, x3), dim=1).flatten(1)

    x = self.fc(x)

    return x
  def embed(self, x):
    #layer1
    x = self.layer1_conv2d(x)

    #layer2
    x = self.layer2_conv2d(x)
    u = x
    x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
    x = self.layer2_seModule(x)
    x = u * x

    #layer3
    x1 = self.layer3_conv2d_block1(x)
    u1 = x1
    x1 = x1.view(x1.size(0), x1.size(1), -1).mean(-1).view(x1.size(0), x1.size(1), 1, 1)
    x1 = self.layer3_seModule_block1(x1)
    x1 = u1 * x1

    x2 = self.layer3_conv2d_block2(x)
    u2 = x2
    x2 = x2.view(x2.size(0), x2.size(1), -1).mean(-1).view(x2.size(0), x2.size(1), 1, 1)
    x2 = self.layer3_seModule_block2(x2)
    x2 = u2 * x2

    x3 = self.layer3_conv2d_block3(x)
    u3 = x3
    x3 = x3.view(x3.size(0), x3.size(1), -1).mean(-1).view(x3.size(0), x3.size(1), 1, 1)
    x3 = self.layer3_seModule_block3(x3)
    x3 = u3 * x3

    #layer4
    x1 = torch.flatten(x1, start_dim=1, end_dim=2)
    x2 = torch.flatten(x2, start_dim=1, end_dim=2)
    x3 = torch.flatten(x3, start_dim=1, end_dim=2)

    x1_short = self.layer4_conv1d_short_block1(x1)

    x1 = self.layer4_conv1d_block1(x1)
    u1 = x1
    x1 = x1.view(x1.size(0), x1.size(1), -1).mean(-1).view(x1.size(0), x1.size(1), 1, 1).flatten(2, 3)
    x1 = self.layer4_seModule_block1(x1)
    x1 = u1 * x1
    x1 = x1 + x1_short

    x2_short = self.layer4_conv1d_short_block2(x2)

    x2 = self.layer4_conv1d_block2(x2)
    u2 = x2
    x2 = x2.view(x2.size(0), x2.size(1), -1).mean(-1).view(x2.size(0), x2.size(1), 1, 1).flatten(2, 3)
    x2 = self.layer4_seModule_block2(x2)
    x2 = u2 * x2
    x2 = x2 + x2_short

    x3_short = self.layer4_conv1d_short_block3(x3)

    x3 = self.layer4_conv1d_block3(x3)
    u3 = x3
    x3 = x3.view(x3.size(0), x3.size(1), -1).mean(-1).view(x3.size(0), x3.size(1), 1, 1).flatten(2, 3)
    x3 = self.layer4_seModule_block3(x3)
    x3 = u3 * x3
    x3 = x3 + x3_short

    x1 = self.layer5_avg_pool1(x1)
    x2 = self.layer5_avg_pool2(x2)
    x3 = self.layer5_avg_pool3(x3)

    x = torch.cat((x1, x2, x3), dim=1).flatten(1)

    for i in range(self.num_layers - 1):
      x = getattr(self.fc, f"ln{i+1}")(x)
      x = getattr(self.fc, f"act{i+1}")(x)

    return x