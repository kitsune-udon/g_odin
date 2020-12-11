import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from argparse_utils import from_argparse_args
from metric import auroc


class GODIN(pl.LightningModule):
    def __init__(self,
                 *args,
                 similarity=None,
                 use_dropout=None,
                 use_preprocessing=None,
                 perturbation_magnitude=None,
                 learning_rate=None,
                 weight_decay=None,
                 warmup_epochs=None,
                 max_epochs=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.net = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121',
                                  pretrained=False)
        num_features = 1000
        n_classes = 10

        self.similarity = similarity
        self.use_dropout = use_dropout
        self.use_preprocessing = use_preprocessing
        self.perturbation_magnitude = perturbation_magnitude

        if self.similarity == 'cosine':
            self.linear_h = nn.Linear(num_features, n_classes, bias=False)
            kaiming_normal_(self.linear_h.weight)
        elif self.similarity == 'iprod':
            self.linear_h = nn.Linear(num_features, n_classes, bias=True)
            kaiming_normal_(self.linear_h.weight)
        elif self.similarity == 'plain':
            self.linear_h = nn.Linear(num_features, n_classes, bias=True)
            kaiming_normal_(self.linear_h.weight)
        else:
            raise ValueError("invalid similarity mode")

        self.linear_g = nn.Linear(num_features, 1)
        self.bn_g = nn.BatchNorm1d(1)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.7)

    def get_gh(self, x):
        t = self.net(x)

        if self.use_dropout:
            t = self.dropout(t)

        if self.similarity == 'plain':
            g = 1
        else:
            g = torch.sigmoid(self.bn_g(self.linear_g(t)))

        if self.similarity == 'cosine':
            w_norm = self.linear_h.weight.norm(dim=1).unsqueeze(0)
            t_norm = t.norm(p=None, dim=1).unsqueeze(1)

            norm = t_norm * w_norm
            eps = 1e-9
            norm = torch.where(norm < eps, torch.tensor(
                eps, device=norm.device), norm)
            h = self.linear_h(t) / norm
        elif self.similarity == 'iprod':
            h = self.linear_h(t)
        elif self.similarity == 'plain':
            h = self.linear_h(t)
        else:
            raise ValueError("invalid similarity mode")

        return h, g

    def forward(self, x):
        h, g = self.get_gh(x)
        return h / g

    def calc_loss(self, x, label):
        loss = F.cross_entropy(self(x), label)

        return loss

    def score(self, x):
        def by_hmax(x):
            h, _ = self.get_gh(x)
            return h.max(dim=1).values

        def by_g(x):
            _, g = self.get_gh(x)
            return g

        return by_hmax(x)

    def training_step(self, batch, batch_idx):
        x, label = batch
        loss = self.calc_loss(x, label)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        if not self.use_preprocessing:
            x, label = batch

            with torch.no_grad():
                s = self.score(x)

            return {'score': s, 'label': label}
        else:
            torch.set_grad_enabled(True)
            x, label = batch
            x = x.requires_grad_()

            s = self.score(x).sum()
            grad = torch.autograd.grad(s, x)[0]
            x_ = x + self.perturbation_magnitude * torch.sign(grad)

            with torch.no_grad():
                s = self.score(x_)

            return {'score': s, 'label': label}

    def validation_epoch_end(self, outputs):
        score = torch.cat([o['score'] for o in outputs]).cpu().numpy()
        label = torch.cat([o['label'] for o in outputs]).cpu().numpy()

        auroc_v = auroc(score, label)
        np.savetxt('score', score)
        np.savetxt('label', label)

        logs = {
            'val_acc': torch.tensor(auroc_v),
            'auroc': torch.tensor(auroc_v),
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
        optimizer = AdamW(params,
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        scheduler = StepLR(optimizer, 5, gamma=0.7)

        return [optimizer], [scheduler]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--warmup_epochs', type=int, default=10)
        parser.add_argument('--similarity', type=str, default='cosine')
        parser.add_argument('--use_dropout', action='store_true')
        parser.add_argument('--use_preprocessing', action='store_true')
        parser.add_argument('--perturbation_magnitude',
                            type=float, default=0.0025)

        return parser
