import torch
import pathlib
import pickle
import numpy as np
import plotly.graph_objects as go


# This file was used to visualize CVCP-Fusion's predictions. To use, pass in a stored output tensor from the model.


for mainCounter in range(8):
    # Define a custom function to fix WindowsPath to PosixPath conversion
    def map_windows_to_posix(path):
        if isinstance(path, pathlib.WindowsPath):
            return pathlib.PosixPath(*path.parts)
        return path

    # Custom unpickler class to apply the path fix
    class WindowsPathFixerUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'pathlib' and name == 'WindowsPath':
                return pathlib.PosixPath  # Replace WindowsPath with PosixPath
            return super().find_class(module, name)

    if (mainCounter % 4) == 0:
        dataFolder = "logs"
    else:
        dataFolder = "logs" + str(mainCounter % 4)
    
    # Load the predicted bounding boxes using the custom unpickler
    preds_file_path = f"/Users/pranav/Desktop/UTD Lab/NewStuff/logsall/{dataFolder}/preds_cpu.pkl"
    with open(preds_file_path, 'rb') as f:
        preds = WindowsPathFixerUnpickler(f).load()

    # Load the ground truth bounding boxes
    ground_truth_file_path = f"/Users/pranav/Desktop/UTD Lab/NewStuff/logsall/{dataFolder}/gts.pt"
    groundTruth = torch.load(ground_truth_file_path, map_location=torch.device('cpu'))

    # Select the batch (e.g., batch index 0)
    batch_index = mainCounter % 2
    ground_truth_batch = groundTruth[batch_index]
    preds_batch = preds[batch_index]

    # Extract the ground truth bounding boxes (including z for height positioning)
    x_gt = ground_truth_batch[:, 0]
    y_gt = ground_truth_batch[:, 1]
    z_gt = ground_truth_batch[:, 2]  # z-coordinate (height in 3D space)
    width_gt = ground_truth_batch[:, 3]
    depth_gt = ground_truth_batch[:, 4]  # depth (length in the z-dimension)
    height_gt = ground_truth_batch[:, 5]
    yaw_gt = ground_truth_batch[:, 6]

    # Extract the predicted bounding boxes from 'box3d_lidar' (including z)
    pred_boxes = preds_batch['box3d_lidar']
    x_pred = pred_boxes[:, 0]  # X coordinate
    y_pred = pred_boxes[:, 1]  # Y coordinate
    z_pred = pred_boxes[:, 2]  # Z coordinate
    width_pred = pred_boxes[:, 3]  # Width
    depth_pred = pred_boxes[:, 4]  # Depth
    height_pred = pred_boxes[:, 5]  # Height
    yaw_pred = pred_boxes[:, 6]  # Yaw (rotation angle)

    # print(width_pred)
    # print(height_pred)
    # print(depth_pred)

    # print(x_pred)
    # print(y_pred)
    # print(z_pred)

    # Function to create vertices of 3D bounding boxes (cuboids) manually
    def create_bbox_vertices_manual(x, y, z, width, height, depth, yaw):
        l, w, h = width / 2, depth / 2, height / 2  # Correctly half the height for top and bottom
        
        # Compute the corners without any rotation
        corners = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # Bottom face
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]   # Top face
        ])

        # Apply yaw rotation manually for each point (corner) around the z-axis
        rotated_corners = []
        for corner in corners:
            # Extract corner coordinates
            x_c, y_c, z_c = corner
            
            # Apply yaw rotation manually (rotation around z-axis)
            rotated_x = x_c * np.cos(yaw) - y_c * np.sin(yaw)
            rotated_y = x_c * np.sin(yaw) + y_c * np.cos(yaw)
            
            # Add the rotated and translated corner to the list
            rotated_corners.append([rotated_x + x, rotated_y + y, z_c + z])
        
        return np.array(rotated_corners)

    # Indices to form faces from the vertices (0-indexed)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 2, 6], [1, 6, 5],  # Right face
    ])

    # Plot the 3D bounding boxes with Plotly
    fig = go.Figure()

    # Add ground truth bounding boxes
    for i in range(len(x_gt)):
        corners_gt = create_bbox_vertices_manual(float(x_gt[i]), float(y_gt[i]), float(z_gt[i]), float(width_gt[i]), float(height_gt[i]), float(depth_gt[i]), float(yaw_gt[i]))
        fig.add_trace(go.Mesh3d(
            x=corners_gt[:, 0], y=corners_gt[:, 1], z=corners_gt[:, 2],  # Correct orientation
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],  # Define the faces of the cuboid
            color='blue', opacity=0.3, name='Ground Truth'
        ))

    # Add predicted bounding boxes
    for i in range(len(x_pred)):
        corners_pred = create_bbox_vertices_manual(float(x_pred[i]), float(y_pred[i]), float(z_pred[i]), float(width_pred[i]), float(height_pred[i]), float(depth_pred[i]), float(yaw_pred[i]))
        fig.add_trace(go.Mesh3d(
            x=corners_pred[:, 0], y=corners_pred[:, 1], z=corners_pred[:, 2],  # Correct orientation
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],  # Define the faces of the cuboid
            color='red', opacity=0.3, name='Prediction'
        ))

    # Force the 3D aspect ratio to 1:1:1 for proper scaling
    fig.update_layout(
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            aspectmode='cube',  # This forces a 1:1:1 ratio across the axes
            xaxis=dict(range=[-70, 70]),
            yaxis=dict(range=[-70, 70]),
            zaxis=dict(range=[-70, 70]),
        ),
        title=f'Ground Truth (Blue) vs Predicted (Red) 3D Bounding Boxes for Batch {batch_index}, Data Point {dataFolder[-1:]}',
        showlegend=True,
    )

    # Display the interactive plot
    fig.show()
