from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 

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
        
        boxes = self.bbox_head.predict(example, preds, self.test_cfg)
        visualize(boxes[0]['box3d_lidar'], example['gt_boxes_and_cls'][0][:, :-1])

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
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
        rect = patches.Rectangle((lower_left_x.cpu().item(), lower_left_y.cpu().item()), l.cpu(
        ).item(), w.cpu().item(), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Plot each label bounding box
    for bbox in label_boxes_2d:
        c_x, c_y, w, l = bbox
        lower_left_x = c_x - w / 2
        lower_left_y = c_y - l / 2
        rect = patches.Rectangle((lower_left_x.cpu().item(), lower_left_y.cpu().item()), l.cpu(
        ).item(), w.cpu().item(), linewidth=1, edgecolor='g', facecolor='none')
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