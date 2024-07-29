from torch import nn
from torchmetrics import Metric
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy
import torch
from .decoder import Decoder
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchmetrics import Metric
import torch
from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # self.f1 = F1Score3D()
        self.mlp = MLP(num_hidden=4)
        self.decoder = Decoder(dim=128, blocks=[128, 128, 64])
        self.decoder.load_state_dict(torch.load('./decoder.pth'))
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=384, kernel_size=(768, 3, 3), stride=(768, 1, 1), padding=(0, 1, 1)).cuda()

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        ctp_bev = x

        # fusion
        cvt_bev = torch.tensor([]).cuda()
        for batch in example['metadata']:
            bev_loaded_batch = torch.load(
                f"./predictions/{batch['token']}.pth").cuda()
            cvt_bev = torch.cat((cvt_bev, bev_loaded_batch), dim=0)
        cvt_bev = self.decoder(cvt_bev)

        # interpolate cvt_bev to match sizes
        cvt_bev = self.mlp(cvt_bev)
        import torch.nn.functional as F
        cvt_bev = F.interpolate(cvt_bev, size=(128, 128))

        # fuse
        bev_fused = torch.cat((cvt_bev, ctp_bev), dim=1).contiguous()
        bev_fused = bev_fused.unsqueeze(1) # Bx1xDxHxW
                
        # Apply the convolution
        bev_fused = self.conv3d(bev_fused) # Bx128x1xHxW

        # Remove the singleton dimension
        # Now the shape is (4, 128, 128, 128)
        bev_fused = bev_fused.squeeze(2)

        preds, _ = self.bbox_head(bev_fused)
        # preds, _ = self.bbox_head(x)

        
        # self.f1(boxes, example['gt_boxes_and_cls'])
        # boxes =  self.bbox_head.predict(example, preds, self.test_cfg)
        # visualize(boxes[0]['box3d_lidar'], example['gt_boxes_and_cls'][0][:, :-1])
        
        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            # visualize(boxes[0]['box3d_lidar'], example['gt_boxes_and_cls'][0][:, :-1])
            return boxes



    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        ctp_bev = x

        # fusion
        cvt_bev = torch.tensor([]).cuda()
        for batch in example['metadata']:
            bev_loaded_batch = torch.load(
                f"./predictions/{batch['token']}.pt").cuda()
            cvt_bev = torch.cat((cvt_bev, bev_loaded_batch), dim=0)

        # interpolate cvt_bev to match sizes
        cvt_bev = self.mlp(cvt_bev)
        import torch.nn. functional as F
        cvt_bev = F.interpolate(cvt_bev, size=(360, 360))

        # fuse
        bev_fused = torch.cat((cvt_bev, ctp_bev), dim=1).contiguous()

        # Define a 1x1 convolution to reduce the 8 channels back to 4 channels
        # conv1x1 = torch.nn.Conv2d(
        #     in_channels=bev_fused.shape[0], out_channels=exa, kernel_size=(1, 1, 1)).cuda()

        # Apply the convolution
        # bev_fused = conv1x1(bev_fused)

        # Remove the singleton dimension
        # Now the shape is (4, 384, 360, 360)
        bev_fused = bev_fused.squeeze(0)

        preds, _ = self.bbox_head(bev_fused)


        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        # self.f1_metric(boxes, example['gt_boxes_and_cls'])
        loss = self.bbox_head.loss(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_fused, loss
        else:
            return boxes, bev_fused, None


def visualize(pred_boxes_3d, label_boxes_3d):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    pred_boxes_2d = []
    label_boxes_2d = []
    for box in pred_boxes_3d:
        x, y, z, w, l, h, vel_x, vel_y, yaw = box
        pred_boxes_2d.append((x, y, w, l))
    for box in label_boxes_3d:
        x, y, z, w, l, h, vel_x, vel_y, yaw = box
        label_boxes_2d.append((x, y, w, l))
    # Create a plot
    fig, ax = plt.subplots()

    # Plot each pred bounding box
    for bbox in pred_boxes_2d:
        c_x, c_y, w, l = bbox
        lower_left_x = c_x - l / 2
        lower_left_y = c_y - w / 2
        w = w.item()
        l = l.item()
        rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item()), w, l, yaw, linewidth=1, edgecolor='r', facecolor='none')
        if yaw != 0.0: 
            print(yaw)
        ax.add_patch(rect)

    # Plot each label bounding box
    for bbox in label_boxes_2d:
        c_x, c_y, w, l = bbox
        lower_left_x = c_x - w / 2
        lower_left_y = c_y - l / 2
        w = w.item()
        l = l.item()
        rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item()), w, l, yaw, linewidth=1, edgecolor='g', facecolor='none')
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
    plt.savefig('bounding_boxes_plot.png')
    plt.close()


def iou_3d(box1, box2):
    """
    Calculate 3D IoU between two bounding boxes.
    """

    from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

    iou_3d = boxes_iou3d_gpu(box1, box2)
    return iou_3d


class F1Score3D(Metric):
    def __init__(self, iou_threshold=0.50, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: dict, labels: torch.Tensor):
        # collate all predictions into a single list
        pred_boxes = []
        pred_scores = []
        pred_cls = []
        for i in range(len(preds)):
            pred_boxes.append(preds[i]["box3d_lidar"])
            pred_scores.append(preds[i]["scores"])
            pred_cls.append(preds[i]['label_preds'])

        PRED_SCORE_THRESHOLD = 0.1
        labels = labels.view(labels.shape[0]*labels.shape[1], labels.shape[-1])
        pred_boxes = torch.cat(pred_boxes, dim=0)[:, [0, 1, 2, 3, 4, 5, -1]]
        pred_scores = torch.cat(pred_scores, dim=0)
        pred_cls = torch.cat(pred_cls, dim=0)
        label_boxes = labels[:, [0, 1, 2, 3, 4, 5, -2]]
        label_cls = labels[:, -1]

        # filter out zero rows in labels and preds
        zero_row_mask_pred = pred_boxes.abs().sum(dim=1) != 0
        pred_boxes = pred_boxes[zero_row_mask_pred]
        pred_cls = pred_cls[zero_row_mask_pred]
        zero_row_mask_label = label_boxes.abs().sum(dim=1) != 0
        label_boxes = label_boxes[zero_row_mask_label]
        label_cls = label_cls[zero_row_mask_label]

        # filter out predictions with score < confidence threshold
        pred_score_mask = pred_scores > PRED_SCORE_THRESHOLD
        pred_boxes = pred_boxes[pred_score_mask]
        pred_cls = pred_cls[pred_score_mask]

        # add class labels to the end of the box tensor
        preds = torch.cat((pred_boxes, pred_cls.unsqueeze(1)), dim=1)
        labels = torch.cat((label_boxes, label_cls.unsqueeze(1)), dim=1)

        # for each class, calculate metrics
        precisions = []
        for cls in labels[:, -1].unique():
            _preds = preds[preds[:, -1] == cls]
            _labels = labels[labels[:, -1] == cls]
            _ious = iou_3d(_preds[:, :-1], _labels[:, :-1])
            print('Class:', cls.item())
            tp, fp, fn = self.calculate_tp_fp_fn_with_iou_matrix_per_class(
                _preds, _labels, _ious, self.iou_threshold)
            precisions.append(tp / (tp + fp))
        print('MEAN Average Precision:', sum(precisions) / len(precisions))

    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.fn + self.tp)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
        return f1

    def calculate_tp_fp_fn_with_iou_matrix_per_class(self, preds, labels, iou_matrix, iou_threshold=0.5):
        tp = fp = fn = 0

        # Goal: Calculate true positives (TP), false positives (FP), and false negatives (FN)
        # TP: Predictions with IoU greater than iou_threshold for at least one label with matching class
        # FP: Ghost predictions with no matching labels (IoU < iou_threshold for all labels), and mispredictions (IoU >= iou_threshold but class does not match, or )
        # FN: Labels with no matching predictions

        # Remove "ghost predictions" with no matching labels
        # No pairs of these predictions and any labels have IoU greater than iou_threshold
        # These predictions are false positives
        unmatched_preds = torch.all(iou_matrix < iou_threshold, dim=1)
        fp += unmatched_preds.sum().item()

        # remove rows for these predictions from the iou matrix
        remaining_preds_mask = ~unmatched_preds
        iou_matrix = iou_matrix[remaining_preds_mask]
        # remove these predictions from the list of predictions
        filtered_preds = preds[remaining_preds_mask]

        # The matrix now only contains predictions with IoU greater than iou_threshold for at least one label. Some of these predictions may still be false positives - mispredictions.

        # First, look for true positives - predictions with IoU greater than iou_threshold for at least one label with matching class
        # For each prediction, find the label with the highest IoU
        # If the class of the prediction and the label match, it is a true positive. We should remove the column for this label from the iou matrix so it cannot be matched.
        # If the class of the prediction and the label do not match, it is a false positive. We should remove this prediction from the list of predictions, but keep the label for future possible matches.

        for pred_ind in range(len(filtered_preds)):
            # Get best label match for this prediction
            best_label_iou, best_label_iou_ind = torch.max(
                iou_matrix[pred_ind], dim=0)

            # Check IoU  > threshold
            if best_label_iou >= iou_threshold:
                tp += 1
                # Remove this label from the iou matrix so it cannot be matched again, only unmatched labels remain
                iou_matrix = torch.cat(
                    (iou_matrix[:, :best_label_iou_ind], iou_matrix[:, best_label_iou_ind+1:]), dim=1)
            else:
                # False positive - IoU < threshold and/or class does not match
                fp += 1

        # Remaining unmatched labels are false negatives
        fn += iou_matrix.shape[1]

        assert tp + \
            fn == len(labels), "TP + FN should equal the number of labels"
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        return tp, fp, fn


class MLP(nn.Module):
    def __init__(self, num_hidden):
        from torchvision.ops import MLP
        super().__init__()
        self.out_dim = 384
        self.mlp = MLP(in_channels=64, hidden_channels=[
                       self.out_dim for _ in range(num_hidden)])
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, output_padding=0),
        #     nn.ReLU()
        # )

    def forward(self, x):
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        # Flatten the spatial dimensions: BxCxHW
        x = x.view(B, C, H * W).permute(0, 2, 1)  # BxHWxC
        x = self.mlp(x)  # Apply MLP: Bx625x128
        x = x.permute(0, 2, 1).view(B, self.out_dim, H, W)  # BxC'xHXW
        # Upsample spatial dimensions to 128x128
        # x = self.upsample(x)
        return x
