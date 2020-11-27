import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers import LARSWrapper, LinearWarmupCosineAnnealingLR
from torch.optim import SGD
from torchvision.models.densenet import densenet121

from argparse_utils import from_argparse_args
from metric import tnr_at_tpr95


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
        self.net = densenet121()
        self.net.classifier = nn.Identity()
        num_features = 1024
        self.bn_g = nn.BatchNorm1d(1)
        self.linear_h = nn.Linear(num_features, 100, bias=False)
        self.linear_g = nn.Linear(num_features, 1)

    def get_gh(self, x):
        t = self.net(x)
        g = torch.sigmoid(self.bn_g(self.linear_g(t)))

        w_norm = self.linear_h.weight.norm(dim=1).unsqueeze(0)
        t_norm = t.norm(p=2, dim=1).unsqueeze(1)

        norm = t_norm * w_norm
        h = self.linear_h(t) / norm

        return h, g

    def forward(self, x):
        h, g = self.get_gh(x)
        return h / g

    def calc_loss(self, x, label):
        y = F.log_softmax(self(x), dim=1)
        loss = F.nll_loss(y, label).mean()

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
        logs = {'val_acc': torch.tensor(tnr_at_tpr95(score, label))}
        results = {'log': logs}

        return results

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.hparams.learning_rate,
                        momentum=0.9,
                        weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0.001
        )

        return [optimizer], [scheduler]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--warmup_epochs', type=int, default=10)

        return parser
