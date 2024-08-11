import torch
from torchmetrics import Metric
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from torchmetrics.utilities import dim_zero_cat

class AveragePrecision(Metric):
    def __init__(self, iou_threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        self.add_state("pred_boxes", default=[], dist_reduce_fx="cat")
        self.add_state("pred_scores", default=[], dist_reduce_fx="cat")
        self.add_state("gt_boxes", default=[], dist_reduce_fx="cat")

    def update(self, pred_boxes, pred_scores, gt_boxes):
        self.pred_boxes.append(pred_boxes)
        self.pred_scores.append(pred_scores)
        self.gt_boxes.append(gt_boxes)

    def compute(self):
        # self.pred_boxes = dim_zero_cat(self.pred_boxes)
        # self.pred_scores = dim_zero_cat(self.pred_scores)
        # self.gt_boxes = dim_zero_cat(self.gt_boxes)
        
        batch_aps = []
        for pred_boxes, pred_scores, gt_boxes in zip(self.pred_boxes, self.pred_scores, self.gt_boxes):
            if (len(pred_boxes) == 0) or (len(gt_boxes) == 0):
                batch_aps.append(torch.tensor(0.0))
                continue
            
            sorted_indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]

            # Vectorized IoU computation
            ious = boxes_iou3d_gpu(pred_boxes[:, [0, 1, 2, 3, 4, 5, -1]], 
                                   gt_boxes[:, [0, 1, 2, 3, 4, 5, -1]])

            # Find the best IoU for each prediction
            best_ious, best_gt_indices = torch.max(ious, dim=1)

            tp = torch.zeros(len(pred_boxes), dtype=torch.float32, device=pred_boxes.device)
            fp = torch.zeros(len(pred_boxes), dtype=torch.float32, device=pred_boxes.device)

            matched_gts = set()

            for i, (best_iou, best_gt_idx) in enumerate(zip(best_ious, best_gt_indices)):
                if best_iou >= self.iou_threshold and best_gt_idx.item() not in matched_gts:
                    tp[i] = 1
                    matched_gts.add(best_gt_idx.item())
                else:
                    fp[i] = 1

            cumulative_tp = torch.cumsum(tp, dim=0)
            cumulative_fp = torch.cumsum(fp, dim=0)

            precision = cumulative_tp / (cumulative_tp + cumulative_fp)
            recall = cumulative_tp / len(gt_boxes)

            # Adding (0,1) point to the precision-recall curve
            precision = torch.cat([torch.tensor([1.0], device=precision.device), precision])
            recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])

            # AP is the area under the precision-recall curve
            ap = torch.trapz(precision, recall)
            batch_aps.append(ap)

        # Average AP over all batches
        mean_ap = torch.stack(batch_aps).mean()

        return mean_ap
