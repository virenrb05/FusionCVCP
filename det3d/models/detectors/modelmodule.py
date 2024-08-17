import torch
from lightning import LightningModule
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi
import numpy as np
from .AveragePrecision import AveragePrecision
import json


class CPModel(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.avg_precision = AveragePrecision()
        self.preds = {}

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
        # NOTE: a box is of form xyz, wlh, yaw, vx, vy, class
        self.logger.experiment.add_text("token", batch["metadata"][0]["token"], global_step=self.trainer.global_step)
        _, orig_preds, boxes = self.model(batch)
        preds_boxes = boxes[0]["box3d_lidar"]
        self.visualize(batch, pred_boxes_3d=preds_boxes, orig_preds=orig_preds, idx=batch_idx)

        labels = batch["gt_boxes_and_cls"][..., :7].squeeze(0)
        labels = labels[~torch.all(labels == 0.0, dim=1)]
        
        # log histogram of confidence scores
        scores = boxes[0]["scores"]
        if scores.numel() > 0: self.logger.experiment.add_histogram("scores", scores, global_step=self.trainer.global_step)
        
        # log average precision
        ap = self.avg_precision(preds_boxes, boxes[0]["scores"], labels)
        self.logger.experiment.add_text("AP", str(ap), global_step=self.trainer.global_step)
        self.log("test_ap", ap, batch_size=1, on_epoch=False, on_step=True)
        self.avg_precision.reset()
        
        # add predictions to preds dict
        self.preds[batch["metadata"][0]["token"]] = (preds_boxes, scores)

        return boxes
    
    def on_test_end(self):
        # from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import DetectionEval, DetectionConfig
        # from nuscenes.eval.common.config import config_factory
        # import json
        # # cfg = config_factory('detection_cvpr_2019')
        # with open('cfg.json') as f:
        #     cfg = DetectionConfig.deserialize(json.load(f))
        # nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuScenes', verbose=True)
        # eval = DetectionEval(nusc, cfg, 'predictions.json', 'val', 'nusclogs')
        # eval.main(plot_examples=5)
        # pass
        
        # generate predictions.json for nuscenes evaluation
        # needs to be in specific format
        '''
        {
            "meta": {
                "use_camera": true
                "use_lidar": true
                "use_radar": false
                "use_map": false
                "use_external": false
            },
            "results": {
                sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
            }
        }
        '''
        
        # each sample_result is a dict with the following
        '''
        sample_result {
        "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
        "size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
        "rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
        "detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
        "attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
                                            See table below for valid attributes for each class, e.g. cycle.with_rider.
                                            Attributes are ignored for classes without attributes.
                                            There are a few cases (0.4%) where attributes are missing also for classes
                                            that should have them. We ignore the predicted attributes for these cases.
        }
        '''
        
        # construct predictions.json
        predictions = {
            "meta": {
                "use_camera": True,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False
            },
            'results': {}
        }
        
        for token, (boxes, scores) in self.preds.items():
            sample_result = []
            for box, score in zip(boxes, scores):
                attr = 'vehicle.moving' if torch.norm(box[7:9]) > 0.2 else 'vehicle.parked'
                pred = {
                    'sample_token': token,
                    'translation': box[:3].tolist(),
                    'size': box[3:6].tolist(),
                    'rotation': box[6].tolist(),
                    'velocity': box[7:9].tolist(),
                    'detection_name': 'car',
                    'detection_score': score.item(),
                    'attribute_name': attr
                }
                sample_result.append(pred)
            predictions['results'][token] = sample_result
        
        # write predictions to file
        with open('predictions.json', 'w') as f:
            json.dump(predictions, f)

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
