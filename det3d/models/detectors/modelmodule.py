import torch
from lightning import LightningModule
from collections import OrderedDict


class CPModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        loss = self.model(batch, return_loss=True)
        loss, _  = self.parse_second_losses(loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=0, weight_decay=0.01, betas=(0.9, 0.99), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.001, total_steps=self.trainer.estimated_stepping_batches, max_momentum=0.95, base_momentum=0.85, div_factor=10.0, pct_start=0.4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'frequency': 1,
                'interval': 'step'},
        }

    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses['loss'])
        for loss_name, loss_value in losses.items():
            if loss_name == 'loc_loss_elem':
                log_vars[loss_name] = [[i.item() for i in j]
                                       for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars
