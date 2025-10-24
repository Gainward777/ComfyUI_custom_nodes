import numpy as np
import torch
from PIL import Image
import cv2
import torch
import math
import torch.nn.functional as F

class DrawRedMaskContour:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),                
                "mask": ("MASK",),                
                "thickness": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider",
                }),
                "grow_by": ("INT", {
                    "default": 20, 
                    "min": 0, 
                    "max": 150, 
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("result_image", "result_mask",)
    FUNCTION = "execute"
    CATEGORY = "Custom"

    def expand_mask(self, mask: torch.Tensor, pixels: int, shape: str = "circle") -> torch.Tensor:
        """
        Увеличить бинарную маску на заданное число пикселей во все стороны (torch).
        Вход: mask (1, H, W) float32 в [0,1]. Выход: (1, H, W) float32 в {0,1}.
        """
        assert isinstance(mask, torch.Tensor), "mask must be torch.Tensor"
        assert mask.ndim == 3 and mask.shape[0] == 1, "mask must be (1,H,W)"

        if pixels <= 0:
            return (mask > 0.5).to(dtype=torch.float32)

        device = mask.device
        dtype = torch.float32
        k = 2 * pixels + 1

        if shape == "square":
            kernel = torch.ones((k, k), device=device, dtype=dtype)
        elif shape == "cross":
            kernel = torch.zeros((k, k), device=device, dtype=dtype)
            c = pixels
            kernel[c, :] = 1.0
            kernel[:, c] = 1.0
        else:  # circle
            yy, xx = torch.meshgrid(
                torch.arange(k, device=device),
                torch.arange(k, device=device),
                indexing='ij'
            )
            c = pixels
            dist2 = (yy - c) ** 2 + (xx - c) ** 2
            kernel = (dist2 <= (pixels ** 2)).to(dtype)

        weight = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        padding = pixels
        src = (mask > 0.5).to(dtype).unsqueeze(1)  # (1,1,H,W)
        summed = F.conv2d(src, weight, padding=padding)  # (1,1,H,W)
        dilated = (summed > 0).to(dtype)[:, 0, :, :]
        return dilated
        

    def draw_contour(self, im: torch.Tensor, grown_mask: torch.Tensor, thickness: int = 2, color=(0, 0, 255)):
        """
        Нарисовать контур по бинарной маске на изображении (torch).
        Вход: im (1,H,W,3) float32 [0,1], grown_mask (1,H,W) float32 {0,1}.
        Выход: (1,H,W,3) float32 [0,1].
        """
        assert isinstance(im, torch.Tensor) and isinstance(grown_mask, torch.Tensor), "inputs must be torch.Tensor"
        assert im.ndim == 4 and im.shape[0] == 1 and im.shape[-1] == 3, "image must be (1,H,W,3)"
        assert grown_mask.ndim == 3 and grown_mask.shape[0] == 1, "mask must be (1,H,W)"

        device = im.device
        dtype = im.dtype

        # Бинаризуем маску
        mask_bin = (grown_mask > 0.5).to(torch.float32)  # (1,H,W)

        # Толщина: аппроксимируем контур кольцом из разности дилатации и эрозии
        r = max(1, int(math.ceil(thickness / 2)))

        def _disk(radius: int) -> torch.Tensor:
            k = 2 * radius + 1
            yy, xx = torch.meshgrid(
                torch.arange(k, device=device),
                torch.arange(k, device=device),
                indexing='ij'
            )
            c = radius
            dist2 = (yy - c) ** 2 + (xx - c) ** 2
            return (dist2 <= (radius ** 2)).to(torch.float32)

        def _dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
            k = _disk(radius).unsqueeze(0).unsqueeze(0)
            s = F.conv2d(x.unsqueeze(1), k, padding=radius)
            return (s > 0).to(torch.float32)[:, 0]

        def _erode(x: torch.Tensor, radius: int) -> torch.Tensor:
            k2 = _disk(radius)
            k = k2.unsqueeze(0).unsqueeze(0)
            s = F.conv2d(x.unsqueeze(1), k, padding=radius)
            need = float(k2.sum().item())
            return (s >= need - 1e-6).to(torch.float32)[:, 0]

        dil = _dilate(mask_bin, r)
        ero = _erode(mask_bin, max(1, r - 1))
        edge = (dil > 0.5) & (ero <= 0.5)  # (1,H,W) bool

        # Подготовим цвет: входной color предполагался BGR (как в OpenCV)
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            bgr = torch.tensor([color[0], color[1], color[2]], device=device, dtype=torch.float32) / 255.0
            rgb = bgr[[2, 1, 0]]  # BGR -> RGB
        else:
            rgb = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)  # fallback: красный
        rgb = rgb.to(dtype).view(1, 1, 1, 3)

        img = im.clone()
        edge3 = edge.unsqueeze(-1)  # (1,H,W,1)
        img = torch.where(edge3, rgb, img)
        return img.clamp(0.0, 1.0)



    def execute(self, 
                image: torch.Tensor,                
                mask: torch.Tensor,
                thickness: int, 
                grow_by: int):
                    
            # Ожидаем: image (1,H,W,3) float32 [0,1], mask (1,H,W) float32 [0,1]
            result_mask = self.expand_mask(mask, grow_by)           
            result_image = self.draw_contour(image, result_mask, thickness)
                    
            return (result_image, result_mask,)
        

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "DrawRedMaskContour": DrawRedMaskContour
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawRedMaskContour": "Draw Red Mask Contour"
}
