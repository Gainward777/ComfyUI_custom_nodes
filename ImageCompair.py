import numpy as np
import torch
from PIL import Image
import cv2

class ImageCompair:
   

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {               
                "source_image": ("IMAGE",),
                "collage_image": ("IMAGE",),
                "source_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "Custom"

    

    def execute(self, source_image: torch.Tensor, 
                      collage_image: torch.Tensor, 
                      source_mask: torch.Tensor):

        _, source_height, source_width, _ = source_image.shape

        # Создаем пустую маску с размерами collage_image
        collage_batch, collage_height, collage_width, _ = collage_image.shape
        collage_mask = torch.zeros((collage_batch, collage_height, collage_width),
                                   dtype=source_mask.dtype,
                                   device=source_mask.device)

        if source_width > source_height:
            # Вертикальное объединение: collage_image будет сверху source_image
            final_image_tensor = torch.cat((collage_image, source_image), dim=1)
            final_mask = torch.cat((collage_mask, source_mask), dim=1)
        else:
            # Горизонтальное объединение: collage_image будет слева от source_image
            final_image_tensor = torch.cat((collage_image, source_image), dim=2)
            final_mask = torch.cat((collage_mask, source_mask), dim=2)

        return (final_image_tensor, final_mask)


# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageCompair": ImageCompair
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCompair": "ImageCompair"
}