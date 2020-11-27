import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from argparse_utils import from_argparse_args
from metric import tnr_at_tpr95, auroc


class GODIN(pl.LightningModule):
    def __init__(self,
                 *args,
                 learning_rate=None,
                 weight_decay=None,
                 warmup_epochs=None,
                 max_epochs=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34',
                                  pretrained=True)
        self.net.classifier = nn.Identity()
        num_features = 1000
        n_classes = 10
        self.bn_g = nn.BatchNorm1d(1)
        self.linear_h = nn.Linear(num_features, n_classes, bias=False)
        self.linear_g = nn.Linear(num_features, 1)

    def get_gh(self, x):
        t = self.net(x)
        g = torch.sigmoid(self.bn_g(self.linear_g(t)))

        w_norm = self.linear_h.weight.norm(dim=1).unsqueeze(0)
        t_norm = t.norm(p=2, dim=1).unsqueeze(1)

        norm = t_norm * w_norm
        eps = 1e-3
        norm = torch.where(norm < eps, torch.tensor(eps, device=norm.device), norm)
        h = self.linear_h(t) / norm

        return h, g

    def forward(self, x):
        h, g = self.get_gh(x)
        return h / g

    def calc_loss(self, x, label):
        y = F.log_softmax(self(x), dim=1)
        loss = F.nll_loss(y, label)

        return loss

    def score(self, x):
        h, _ = self.get_gh(x)
        return h.max(dim=1).values

    def training_step(self, batch, batch_idx):
        x, label = batch
        loss = self.calc_loss(x, label)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, label = batch

        with torch.no_grad():
            s = self.score(x)

        return {'score': s, 'label': label}

    def validation_epoch_end(self, outputs):
        score = torch.cat([o['score'] for o in outputs]).cpu().numpy()
        label = torch.cat([o['label'] for o in outputs]).cpu().numpy()

        mode = 'auroc'

        if mode == 'auroc':
            logs = {
                'val_acc': torch.tensor(auroc(score, label))
            }
        else:
            crit, fitness = tnr_at_tpr95(score, label)
            logs = {
                'val_acc': torch.tensor(fitness),
                'criteria': torch.tensor(crit)
            }

        results = {'log': logs}

        return results

    def configure_optimizers(self):
        params = [
            {'params': self.net.parameters()},
            {'params': self.linear_g.parameters()},
            {'params': self.bn_g.parameters()},
            {'params': self.linear_h.parameters(), 'weight_decay': 0}
        ]
        optimizer = SGD(params,
                        lr=self.hparams.learning_rate,
                        momentum=0.9,
                        weight_decay=self.hparams.weight_decay)

        return optimizer

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.0005)
        parser.add_argument('--warmup_epochs', type=int, default=10)

        return parser
