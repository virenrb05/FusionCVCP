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
        self.f1_metric = F1Score3D()

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
        pred_boxes, _ = self.bbox_head(x)

        preds = self.bbox_head.predict(example, pred_boxes, self.test_cfg)
        
        self.f1_metric(preds, example['gt_boxes_and_cls'])

        if return_loss:
            return self.bbox_head.loss(example, pred_boxes, self.test_cfg)
        else:
            return preds

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
        bev_feature = x
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        self.f1_metric(boxes, example['gt_boxes_and_cls'])

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None


def get_3d_box_vertices(center, dims, yaw):
    """
    Calculate the 8 vertices of the 3D bounding box.
    """

    corners = torch.tensor([])
    for i in range(len(center)):
        c = center[i]
        d = dims[i]
        y = yaw[i]
        dx = d[0] / 2
        dy = d[1] / 2
        dz = d[2] / 2

        corners_i = torch.tensor([
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz]
        ])
        rotation = torch.from_numpy(R.from_euler(
            'z', y, degrees=False).as_matrix())
        rotated_corners = torch.matmul(
            corners_i, rotation.T.type_as(corners_i))
        corners = torch.cat(
            (corners, (rotated_corners + c).unsqueeze(0)), dim=0)
    return corners


def iou_3d(box1, box2):
    """
    Calculate 3D IoU between two bounding boxes.
    """

    from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

    iou_3d = boxes_iou3d_gpu(box1, box2)

    # box1 = box1.cpu()
    # box2 = box2.cpu()
    # from det3d.core.bbox.box_torch_ops import center_to_corner_box3d
    # v1 = center_to_corner_box3d(centers=box1[:, :3], dims=box1[:, 3:6], angles=box1[:, 6])
    # v2 = center_to_corner_box3d(centers=box2[:, :3], dims=box2[:, 3:6], angles=box2[:, 6])
    # v1 = get_3d_box_vertices(box1[:, :3], box1[:, 3:6], box1[:, 6])
    # v2 = get_3d_box_vertices(box2[:, :3], box2[:, 3:6], box2[:, 6])
    # v2 = get_3d_box_vertices(box2[:3], box2[3:6], box2[6])

    # v1 = np.expand_dims(v1, axis=0)
    # v2 = np.expand_dims(v2, axis=0)

    # v1 = torch.from_numpy(v1)
    # v2 = torch.from_numpy(v2.astype(np.float32))

    # from pytorch3d.ops import box3d_overlap
    # _, iou_3d = box3d_overlap(v1, v2)
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

        ious = iou_3d(pred_boxes, label_boxes)

        # add class labels to the end of the box tensor
        preds = torch.cat((pred_boxes, pred_cls.unsqueeze(1)), dim=1)
        labels = torch.cat((label_boxes, label_cls.unsqueeze(1)), dim=1)
        self.calculate_tp_fp_fn_with_iou_matrix(
            preds, labels, ious, self.iou_threshold)

    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.fn + self.tp)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
        return f1

    def calculate_tp_fp_fn_with_iou_matrix(self, preds, labels, iou_matrix, iou_threshold=0.5):
        num_preds = len(preds)

        # Goal: Calculate true positives (TP), false positives (FP), and false negatives (FN)
        # TP: Predictions with IoU greater than iou_threshold for at least one label with matching class
        # FP: Ghost predictions with no matching labels (IoU < iou_threshold for all labels), and mispredictions (IoU >= iou_threshold but class does not match, or )
        # FN: Labels with no matching predictions

        # Remove "ghost predictions" with no matching labels
        # No pairs of these predictions and any labels have IoU greater than iou_threshold
        # These predictions are false positives
        unmatched_preds = torch.all(iou_matrix < iou_threshold, dim=1)
        self.fp += unmatched_preds.sum().item()
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

            # Check IoU  > threshold and class match
            if best_label_iou >= iou_threshold and filtered_preds[pred_ind][-1] == labels[best_label_iou_ind][-1]:
                self.tp += 1
                # Remove this label from the iou matrix so it cannot be matched again, only unmatched labels remain
                iou_matrix = torch.cat(
                    (iou_matrix[:, :best_label_iou_ind], iou_matrix[:, best_label_iou_ind+1:]), dim=1)
            else:
                # False positive - IoU < threshold and/or class does not match
                self.fp += 1

        # Remaining unmatched labels are false negatives
        self.fn += iou_matrix.shape[1]

        assert self.tp + \
            self.fn == len(labels), "TP + FN should equal the number of labels"
        print(f"TP: {self.tp}, FP: {self.fp}, FN: {self.fn}")
