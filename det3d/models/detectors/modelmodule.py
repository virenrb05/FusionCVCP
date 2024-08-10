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
import numpy as np
from .AveragePrecision import AveragePrecision
from lightning.pytorch.loggers import TensorBoardLogger


class CPModel(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        # self.avg_precision = AveragePrecision()
        # self.avg_prec = 0
        # self.preds = []

    def training_step(self, batch, batch_idx):
        loss, orig_preds, boxes = self.model(batch)
        loss, second_losses = self.parse_second_losses(loss)
        self.log("train_loss", loss)
        self.log("train_loss_hm", second_losses['hm_loss'][0])
        self.log("train_loss_loc", second_losses['loc_loss'][0])
        self.log("train_loss_loc_car", sum(second_losses['loc_loss_elem'][0]))
        # for i, box in enumerate(boxes):
        #     preds = box['box3d_lidar']
        #     scores = box['scores']
        #     label_boxes = batch['gt_boxes_and_cls'][i, :, :-1].squeeze(0)
        #     # remove any rows with all zeros
        #     label_boxes = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]
        #     self.avg_prec += self.avg_precision(preds, scores, label_boxes)
        #     self.avg_precision.reset()

        if batch_idx % 500 == 0:
            plt.matshow(np.array(batch['hm'][0][0].squeeze().detach().cpu()))
            plt.savefig('label.png')
            plt.close()
            plt.matshow(
                np.array(orig_preds[0]['hm'][0].squeeze().detach().cpu()))
            plt.savefig('pred.png')
            plt.close()
            preds_boxes = boxes[0]['box3d_lidar']
            # filter out boxes with confidence score less than 0.5
            preds_boxes = preds_boxes[boxes[0]['scores'] > 0.5]
            # TODO: display confidence score in visualization next to corresponding box
            # remove any rows with all zeros
            label_boxes = batch['gt_boxes_and_cls'][0, :, :-1].squeeze(0)
            token = batch['metadata'][0]['token']
            self.visualize(preds_boxes, label_boxes,
                           f'{self.current_epoch}-{batch_idx}-{token}')

        return loss

    # def on_train_epoch_end(self):
    #     self.log('train_avg_prec', self.avg_prec / self.trainer.num_training_batches)
    #     self.avg_prec = 0

    # def log_tb_images(self, image, idx, token) -> None:
    #     # Get tensorboard logger
    #     tb_logger = None
    #     for logger in self.trainer.loggers:
    #         if isinstance(logger, TensorBoardLogger):
    #             tb_logger = logger.experiment
    #             break

    #     if tb_logger is None:
    #         raise ValueError('TensorBoard Logger not found')

    #     # Log the images (Give them different names)
    #     tb_logger.add_image(f"Image/{idx}_{token}", image)

    def test_step(self, batch, batch_idx):
        _, orig_preds, boxes = self.model(batch)
        plt.matshow(np.array(batch['hm'][0][0].squeeze().detach().cpu()))
        plt.savefig('label.png')
        plt.close()
        plt.matshow(
            np.array(orig_preds[0]['hm'][0].squeeze().detach().cpu()))
        plt.savefig('pred.png')
        plt.close()
        
        # self.preds.append(boxes)
        preds_boxes = boxes[0]['box3d_lidar']
        # TODO: display confidence score in visualization next to corresponding box
        label_boxes = batch['gt_boxes_and_cls'][..., :-1].squeeze(0)
        # remove any rows with all zeros
        label_boxes = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]
        token = batch['metadata'][0]['token']
        self.visualize(preds_boxes, label_boxes,
                       f'{token}')
        # self.avg_precision.update(preds_boxes, preds[0]['scores'], label_boxes)
        # print(self.avg_precision.compute().item(), token)
        # self.avg_precision.reset()
        return boxes


    #     # for i in range(preds_boxes.shape[0]):
    #     #     attr = 'vehicle.moving' if sqrt(
    #     #         preds_boxes[i, 6] ** 2 + preds_boxes[i, 7] ** 2) > 0.2 else 'vehicle.parked'
    #     #     pred = {
    #     #         'sample_token': batch['metadata'][0]['token'],
    #     #         'translation': preds_boxes[i, :3].tolist(),
    #     #         'size': preds_boxes[i, 3:6].tolist(),
    #     #         'rotation': R.from_euler('z', preds_boxes[i, -1].item(), degrees=False).as_quat().tolist(),
    #     #         'velocity': preds_boxes[i, 6:8].tolist(),
    #     #         'detection_name': 'car',
    #     #         'detection_score': preds[0]['scores'][i].item(),
    #     #         'attribute_name': attr
    #     #     }
    #     #     self.preds.append(pred)

    # def on_test_epoch_end(self):
    #     scores = []
    #     for pred in self.preds:
    #         scores.append(pred[0]['scores'])
    #     scores = torch.cat(scores)
    #     plt.hist(scores.cpu().numpy(), bins=50)
    #     plt.savefig(os.path.join(self.logger.log_dir, "scores.png"))
    #     plt.close()

    #     # with open(os.path.join(self.logger.log_dir, "prediction.pkl"), "wb") as f:
    #     #     pickle.dump(self.preds, f)

    #     # nusc = NuScenes(version='v1.0-trainval', dataroot='/home/vxm240030/nuscenes', verbose=True)

    #     # eval = DetectionEval(nusc, config_factory("cvpr_2019"), os.path.join(self.logger.log_dir, "prediction.pkl"), 'val', self.logger.log_dir, verbose=True)

    #     # eval.main()
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=self.cfg.optimizer.amsgrad, weight_decay=self.cfg.optimizer.wd, betas=(0.9, 0.99), lr=self.cfg.lr_config.lr_max)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.cfg.lr_config.lr_max, total_steps=self.trainer.estimated_stepping_batches, max_momentum=self.cfg.lr_config.moms[0], base_momentum=self.cfg.lr_config.moms[1], div_factor=self.cfg.lr_config.div_factor, pct_start=self.cfg.lr_config.pct_start)
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

    def visualize(self, pred_boxes_3d, label_boxes_3d, token, rotate=True):
       # Create a plot
        fig, ax = plt.subplots()

        for box in pred_boxes_3d:
            x, y, _, w, l, _, _, _, yaw = box

            # flip y axis
            y = -y

            lower_left_x = x - w / 2
            lower_left_y = y - l / 2
            w = w.item()
            l = l.item()
            yaw = yaw * 180 / pi
            rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item(
            )), w, l, yaw, linewidth=1, edgecolor='r', facecolor='none', rotation_point='center')
            ax.add_patch(rect)
            # ax.text(x, y, f'{int(yaw * 180 / pi)}', color='r')

        for box in label_boxes_3d:
            x, y, _, w, l, _, _, _, yaw = box
            # flip y axis
            y = -y
            lower_left_x = x - w / 2
            lower_left_y = y - l / 2
            w = w.item()
            l = l.item()
            yaw = yaw * 180 / pi
            if rotate: yaw += 90
            rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item(
            )), w, l, yaw, linewidth=1, edgecolor='g', facecolor='none', rotation_point='center')
            ax.add_patch(rect)
            # if (_yaw := 90 + yaw * 180 / pi) != 0:
            #     print(_yaw, yaw)
            # ax.text(x, y, f'{int(yaw * 180 / pi)}', color='g')

        # Plot the ego vehicle at the origin
        ax.plot(0, 0, 'bo')  # blue dot

        ax.legend(handles=[patches.Patch(color='r', label='Prediction'), patches.Patch(
            color='g', label='Ground Truth')])

        # Set plot limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        # Set labels and title
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('2D Bounding Boxes Relative to Ego Vehicle')

        # Save the plot as an image
        path = Path(self.logger.log_dir) / 'images'

        os.makedirs(path, exist_ok=True)
        plt.savefig(path / f'{token}.png')
        plt.close()
