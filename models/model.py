import torch
import torch.nn as nn
import torchvision.models as models

def find_pts(x, k):

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_feature(x, k=20, idx=None, dim9=False):

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if not dim9:
            idx = find_pts(x[:, 0:3], k=k)
        else:
            idx = find_pts(x[:, 6:], k=k)

    device = torch.device('cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class Model(nn.Module):
    def __init__(self, k, emb_dims, dropout):

        super(Model, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        # self.resnet = models.resnet50(pretrained=True)
        # self.resnet.fc = nn.Linear(2048, self.emb_dims)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.layer1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=self.dropout)
        self.layer9 = nn.Conv1d(256, 4, kernel_size=1, bias=False)

    def forward(self, x):

        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_feature(x, k=self.k, dim9=False)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_feature(x1, k=self.k)
        x = self.layer3(x)
        x = self.layer4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_feature(x2, k=self.k)
        x = self.layer5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.layer6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.dp1(x)
        x = self.layer9(x)
        
        return x
