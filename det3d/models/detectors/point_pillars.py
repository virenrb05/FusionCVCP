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
        preds, _ = self.bbox_head(x)

        loss = self.bbox_head.loss(example, preds, self.test_cfg)
        boxes = self.bbox_head.predict(example, preds, self.test_cfg)

        # preds should be preds with vel removed and class labels appended
        preds = []
        for boxes_batch in boxes:
            pred = boxes_batch["box3d_lidar"]
            pred = pred[:, list(range(6)) + [-1]]
            pred = torch.cat(
                (pred, boxes_batch['label_preds'].unsqueeze(1)), dim=1)
            preds.append(pred)

        # do the same with the labels
        labels = []
        for _boxes in example['gt_boxes_and_cls']:
            label = _boxes[:, list(range(6)) + [-2, -1]]
            labels.append(label)

        self.f1_metric.update(preds, labels)
        f1 = self.f1_metric.compute()

        if return_loss:
            return loss, f1
        else:
            return boxes, f1

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

    box1 = box1.cpu()
    box2 = box2.cpu()
    # from det3d.core.bbox.box_torch_ops import center_to_corner_box3d
    # v1 = center_to_corner_box3d(centers=box1[:, :3], dims=box1[:, 3:6], angles=box1[:, 6])
    # v2 = center_to_corner_box3d(centers=box2[:, :3], dims=box2[:, 3:6], angles=box2[:, 6])
    v1 = get_3d_box_vertices(box1[:, :3], box1[:, 3:6], box1[:, 6])
    v2 = get_3d_box_vertices(box2[:, :3], box2[:, 3:6], box2[:, 6])
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

    def update(self, preds, target):
        # preds and target are expected to be lists of bounding boxes with class labels
        preds = torch.cat(preds)
        target = torch.cat(target)
        ious = iou_3d(preds[:, :-1], target[:, :-1])
        for col_ind in range(ious.shape[1]):
            pred_col = ious[:, col_ind]
            # get best match for thresholding
            best_label_ind = pred_col.argmax()

            if pred_col[best_label_ind] >= self.iou_threshold and preds[col_ind][-1] == target[best_label_ind][-1]:
                self.tp += 1
                target = torch.cat(
                    (ious[:best_label_ind], ious[best_label_ind+1:]))

            else:
                self.fp += 1

        self.fn += len(target)

    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.fn + self.tp)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
