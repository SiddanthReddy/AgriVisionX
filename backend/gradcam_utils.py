import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_heatmap(model, input_tensor, image_np):
    """
    model: loaded HybridCNNTransformer
    input_tensor: shape (1,3,224,224)
    image_np: original RGB image resized to 224x224, values 0-1
    """

    target_layers = [
        model.cnn_backbone.backbone.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
       input_tensor=input_tensor,
    targets=None
    )[0]

    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True
    )

    return visualization