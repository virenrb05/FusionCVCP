import torch
from lightning import LightningModule
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi
import numpy as np
from .AveragePrecision import AveragePrecision


class CPModel(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.avg_precision = AveragePrecision()

    def training_step(self, batch, batch_idx):
        loss, orig_preds, boxes = self.model(batch)
        loss, second_losses = self.parse_second_losses(loss)
        self.log("train_loss", loss)
        self.log("train_loss_hm", second_losses["hm_loss"][0])
        self.log("train_loss_loc", second_losses["loc_loss"][0])
        self.log("train_loss_loc_car", sum(second_losses["loc_loss_elem"][0]))

        if batch_idx % 400 == 0:
            preds_boxes = boxes[0]["box3d_lidar"]
            # filter out boxes with confidence score less than 0.5
            # preds_boxes = preds_boxes[boxes[0]['scores'] > 0.5]

            # visualize the predictions and ground truths for first batch only
            self.visualize(batch, pred_boxes_3d=preds_boxes, orig_preds=orig_preds, idx=self.trainer.global_step)

            # for AP calculation, use all batches for both preds and labels
            # for i in range(len(boxes)):
            #     preds_boxes = boxes[i]['box3d_lidar']
            #     scores = boxes[i]['scores']
            #     labels = batch['gt_boxes_and_cls'][i, :, :-1]
            #     labels = labels[~torch.all(labels == 0.0, dim=1)]
            #     self.avg_precision.update(preds_boxes, scores, labels)

            # ap = self.avg_precision.compute()
            # self.log('train_ap', ap, batch_size=self.cfg.data.samples_per_gpu,
            #          on_epoch=False, on_step=True)
            # self.avg_precision.reset()

        return loss

    def test_step(self, batch, batch_idx):
        self.logger.experiment.add_text("token", batch["metadata"][0]["token"], global_step=self.trainer.global_step)
        _, orig_preds, boxes = self.model(batch)
        preds_boxes = boxes[0]["box3d_lidar"]
        self.visualize(
            batch, pred_boxes_3d=preds_boxes, orig_preds=orig_preds, idx=batch_idx
        )

        labels = batch["gt_boxes_and_cls"][..., :7].squeeze(0)
        labels = labels[~torch.all(labels == 0.0, dim=1)]

        ap = self.avg_precision(preds_boxes, boxes[0]["scores"], labels)
        self.log("test_ap", ap, batch_size=1, on_epoch=False, on_step=True)
        self.avg_precision.reset()

        return boxes

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
            self.parameters(),
            amsgrad=self.cfg.optimizer.amsgrad,
            weight_decay=self.cfg.optimizer.wd,
            betas=(0.9, 0.99),
            lr=self.cfg.lr_config.lr_max,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr_config.lr_max,
            total_steps=self.trainer.estimated_stepping_batches,
            max_momentum=self.cfg.lr_config.moms[0],
            base_momentum=self.cfg.lr_config.moms[1],
            div_factor=self.cfg.lr_config.div_factor,
            pct_start=self.cfg.lr_config.pct_start,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 1,
                "interval": "step",
            },
        }

    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses["loss"])
        for loss_name, loss_value in losses.items():
            if loss_name == "loc_loss_elem":
                log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars

    def visualize(self, batch, pred_boxes_3d: torch.Tensor, orig_preds: dict, idx=0):
        writer = self.logger.experiment
        token = batch["metadata"][0]["token"]
        writer.add_text("token", token, global_step=idx)

        # visualize the ground truth and predicted heatmaps
        pred_hm = np.array(orig_preds[0]["hm"][0].squeeze().detach().cpu())
        gt_hm = np.array(batch["hm"][0][0].squeeze().detach().cpu())

        writer.add_image("pred_hm", pred_hm, global_step=idx, dataformats="HW")
        writer.add_image("gt_hm", gt_hm, global_step=idx, dataformats="HW")

        # TODO: display confidence score in visualization next to corresponding box
        label_boxes = batch["gt_boxes_and_cls"][0, :, :-1]
        # remove any rows with all zeros
        label_boxes_3d = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]

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
            rect = patches.Rectangle(
                xy=(lower_left_x.item(), lower_left_y.item()),
                width=w,
                height=l,
                angle=yaw,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                rotation_point="center",
            )
            ax.add_patch(rect)

        for box in label_boxes_3d:
            # NOTE: yaw is before velocity here, AssignLabel.py outputs this order
            x, y, _, w, l, _, yaw, _, _ = box
            # flip y axis
            y = -y
            lower_left_x = x - w / 2
            lower_left_y = y - l / 2
            w = w.item()
            l = l.item()
            yaw = yaw * 180 / pi
            rect = patches.Rectangle(
                xy=(lower_left_x.item(), lower_left_y.item()),
                width=w,
                height=l,
                angle=yaw,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
                rotation_point="center",
            )
            ax.add_patch(rect)

        # Plot the ego vehicle at the origin
        ax.plot(0, 0, "bo")  # blue dot

        ax.legend(
            handles=[
                patches.Patch(color="r", label="Prediction"),
                patches.Patch(color="g", label="Ground Truth"),
            ]
        )

        # Set plot limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        # Set labels and title
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_title("2D Bounding Boxes Relative to Ego Vehicle")

        # Now we can save it to a numpy array
        fig.canvas.draw()
        viz = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        viz = viz.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        writer.add_image("visual", viz, global_step=idx, dataformats="HWC")

        plt.close()
