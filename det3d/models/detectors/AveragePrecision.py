import torch
from torchmetrics import Metric
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from torch import Tensor

class AveragePrecision(Metric):
    def __init__(self, iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.add_state('total_precision', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_batches', default=torch.tensor(0), dist_reduce_fx='sum')
        
    def update(self, pred_boxes: Tensor, pred_scores: Tensor, gt_boxes: Tensor):
        avg_step_AP = torch.tensor(0.0)
        
        if pred_scores.ndim > 1:   
            for i in range(pred_scores.shape[0]):
                batch_AP = self.compute_batch_average_precision(pred_boxes[i], pred_scores[i], gt_boxes[i])
                # accumulate metric for epoch
                self.total_precision += batch_AP
                self.num_batches += 1
                
                avg_step_AP += batch_AP
        else:
            # single batch calculation
            batch_AP = self.compute_batch_average_precision(pred_boxes, pred_scores, gt_boxes)
            # accumulate metric for epoch
            self.total_precision += batch_AP
            self.num_batches += 1
            # return batch_AP

        # return avg_step_AP / pred_boxes.shape[0]

    def compute(self):        
        if self.num_batches == 0:
            return torch.tensor(0.0)
        return self.total_precision / self.num_batches
        
        return mean_ap

    def compute_batch_average_precision(self, pred_boxes, pred_scores, gt_boxes):
        if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
            return torch.tensor(0.0)
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
        return ap