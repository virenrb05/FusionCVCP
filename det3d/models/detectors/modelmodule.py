import torch
from lightning import LightningModule
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from math import sqrt
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory
from nuscenes import NuScenes
from math import pi

from .AveragePrecision import AveragePrecision


class CPModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.avg_precision = AveragePrecision()
        self.preds = []

    def training_step(self, batch, batch_idx):
        loss = self.model(batch, return_loss=True)
        loss, second_losses = self.parse_second_losses(loss)
        self.log("train_loss", loss)
        self.log("train_loss_hm", second_losses['hm_loss'][0])
        self.log("train_loss_loc", second_losses['loc_loss'][0])
        self.log("train_loss_loc_car", second_losses['loc_loss_elem'][0])
        return loss

    def test_step(self, batch, batch_idx):
        preds = self.model(batch, return_loss=False)
        preds_boxes = preds[0]['box3d_lidar']
        # TODO: display confidence score in visualization next to corresponding box
        label_boxes = batch['gt_boxes_and_cls'][..., :-1].squeeze(0)
        # remove any rows with all zeros
        label_boxes = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]
        self.visualize(preds_boxes, label_boxes, batch['metadata'][0]['token'])
        
        self.avg_precision.update(preds_boxes, preds[0]['scores'], label_boxes)
        self.avg_precision.compute()
        self.avg_precision.reset()

        # return boxes

        # for i in range(preds_boxes.shape[0]):
        #     attr = 'vehicle.moving' if sqrt(
        #         preds_boxes[i, 6] ** 2 + preds_boxes[i, 7] ** 2) > 0.2 else 'vehicle.parked'
        #     pred = {
        #         'sample_token': batch['metadata'][0]['token'],
        #         'translation': preds_boxes[i, :3].tolist(),
        #         'size': preds_boxes[i, 3:6].tolist(),
        #         'rotation': R.from_euler('z', preds_boxes[i, -1].item(), degrees=False).as_quat().tolist(),
        #         'velocity': preds_boxes[i, 6:8].tolist(),
        #         'detection_name': 'car',
        #         'detection_score': preds[0]['scores'][i].item(),
        #         'attribute_name': attr
        #     }
        #     self.preds.append(pred)

    def on_test_epoch_end(self):
        # with open(os.path.join(self.logger.log_dir, "prediction.pkl"), "wb") as f:
        #     pickle.dump(self.preds, f)

        # nusc = NuScenes(version='v1.0-trainval', dataroot='/home/vxm240030/nuscenes', verbose=True)

        # eval = DetectionEval(nusc, config_factory("cvpr_2019"), os.path.join(self.logger.log_dir, "prediction.pkl"), 'val', self.logger.log_dir, verbose=True)

        # eval.main()
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=0, weight_decay=0.02, betas=(0.9, 0.99), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.0001, total_steps=self.trainer.estimated_stepping_batches, max_momentum=0.95, base_momentum=0.85, div_factor=10.0, pct_start=0.4)
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

    def visualize(self, pred_boxes_3d, label_boxes_3d, token):
       # Create a plot
        fig, ax = plt.subplots()
        
        for box in pred_boxes_3d:
            x, y, _, w, l, _, _, _, yaw = box
            lower_left_x = x - w / 2
            lower_left_y = y - l / 2
            w = w.item()
            l = l.item()
            rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item(
            )), w, l, yaw * 180 / pi, linewidth=1, edgecolor='r', facecolor='none', rotation_point='xy')
            ax.add_patch(rect)
        
        for box in label_boxes_3d:
            x, y, _, w, l, _, _, _, yaw = box
            lower_left_x = x - w / 2
            lower_left_y = y - l / 2
            w = w.item()
            l = l.item()
            rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item(
            )), l, w, yaw * 180 / pi, linewidth=1, edgecolor='g', facecolor='none', rotation_point='xy')
            ax.add_patch(rect)
        
        # Plot the ego vehicle at the origin
        ax.plot(0, 0, 'bo')  # blue dot

        # Set plot limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        # Set labels and title
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('2D Bounding Boxes Relative to Ego Vehicle')

        # Save the plot as an image
        path = Path(self.logger.log_dir) / 'bounding_boxes_plots'
        os.makedirs(path, exist_ok=True)
        plt.savefig(path / f'bounding_boxes_plot_{token}.png')
        plt.close()
