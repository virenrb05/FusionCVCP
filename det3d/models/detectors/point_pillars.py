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


        # # preds should be preds with vel removed and class labels appended
        # preds = []
        # for boxes_batch in boxes:
        #     pred = boxes_batch["box3d_lidar"]
        #     pred = pred[:, list(range(6)) + [-1]]
        #     pred = torch.cat(
        #         (pred, boxes_batch['label_preds'].unsqueeze(1)), dim=1)
        #     preds.append(pred)

        # # do the same with the labels
        # labels = []
        # for _boxes in example['gt_boxes_and_cls']:
        #     label = _boxes[:, list(range(6)) + [-2, -1]]
        #     labels.append(label)

        # self.f1_metric.update(preds, labels)
        # f1 = self.f1_metric.compute()

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

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

    iou_3d = boxes_iou3d_gpu(box1[:, :-1], box2[:, :-1])

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

    def update(self, preds, targets):
        from tqdm import tqdm
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        ious = iou_3d(preds, targets)

        # # Convert class labels to PyTorch tensors for vectorized comparison
        # preds_classes = torch.tensor([pred[-1] for pred in preds])
        # targets_classes = torch.tensor([target[-1] for target in targets])

        # # Calculate matches based on IoU threshold and class labels
        # iou_matches = ious > self.iou_threshold
        # class_matches = (preds_classes.unsqueeze(1) ==
        #                  targets_classes.unsqueeze(0))

        # # True Positives (TP): IoU and class match
        # tp_matrix = iou_matches & class_matches
        # self.tp = tp_matrix.sum().item()

        # # False Positives (FP): IoU or class match, but not both
        # fp_matrix = (iou_matches | class_matches) & ~tp_matrix
        # self.fp = fp_matrix.sum().item()

        # # False Negatives (FN): Target does not have any matching prediction
        # fn_matrix = ~(torch.any(iou_matches, dim=0) &
        #               torch.any(class_matches, dim=0))
        # self.fn = fn_matrix.sum().item()

        # print("True Positives (TP):", self.tp)
        # print("False Positives (FP):", self.fp)
        # print("False Negatives (FN):", self.fn)

        # preds and target are expected to be lists of bounding boxes with class labels
        # for target_i, target in enumerate(targets):
        #     for pred_i, pred in tqdm(enumerate(preds)):
        #         if ious[pred_i, target_i] > self.iou_threshold and pred[-1] == target[-1]:
        #             self.tp += 1
                
        #         if ious[pred_i, target_i] > self.iou_threshold or pred[-1] == target[-1]:
        #             self.fp += 1

        # for target_i, target in tqdm(enumerate(targets)):
        #     if not any(ious[:, target_i] > self.iou_threshold) or not any([pred[-1] == target[-1] for pred in preds]):
        #         self.fn += 1
        
        
        
        
        print("True Positives (TP):", self.tp)
        print("False Positives (FP):", self.fp)
        print("False Negatives (FN):", self.fn)

    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.fn + self.tp)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

def calculate_tp_fp_fn_with_iou_matrix(iou_matrix, iou_threshold=0.5):
    from scipy.optimize import linear_sum_assignment
    num_preds, num_gts = iou_matrix.shape
    
    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    tp = 0
    fp = 0
    fn = 0
    
    matched_pred = set()
    matched_gt = set()
    
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            tp += 1
            matched_pred.add(i)
            matched_gt.add(j)
    
    fp = num_preds - len(matched_pred)
    fn = num_gts - len(matched_gt)
    
    return tp, fp, fn