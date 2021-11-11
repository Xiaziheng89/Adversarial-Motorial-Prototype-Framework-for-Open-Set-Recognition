import torch
import torch.nn as nn
import torch.nn.functional as f
from loss.Dist import Dist


class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(RPLoss, self).__init__()
        self.weight_pl = float(options['weight_pl'])
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
        self.radius = 1
        self.radius = nn.Parameter(torch.Tensor(self.radius))
        self.radius.data.fill_(0)

    def forward(self, x, y, labels=None):
        dist = self.Dist(x)
        logits = f.softmax(dist, dim=1)
        if labels is None:
            return logits, 0
        loss_main = f.cross_entropy(dist, labels)

        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = f.mse_loss(_dis, self.radius)
        loss = loss_main + self.weight_pl * loss_r

        return logits, loss
    

