import numpy as np
import torch
from PIL import Image
import cv2
import torch

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

    def expand_mask(self, mask: np.ndarray, pixels: int, shape: str = "circle") -> np.ndarray:
        """
        Увеличить бинарную маску на заданное число пикселей во все стороны.
        mask  : 2D массив (bool, 0/1 или 0/255)
        pixels: радиус расширения в пикселях (>=0)
        shape : 'circle' | 'square' | 'cross' — форма ядра
        Возвращает маску того же типа (bool -> bool, иначе uint8 0/255).
        """
        # Приводим маску к 2D (H, W): убираем возможные batch/канальную размерности
        m = mask
        if m.ndim > 2:
            # частые варианты: (1, H, W) или (H, W, 1)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            elif m.shape[-1] == 1:
                m = m[..., 0]
            else:
                m = np.squeeze(m)

        if pixels <= 0:
            return m.copy()

        # Подбираем форму ядра
        shape_map = {
            "circle": cv2.MORPH_ELLIPSE,  # хороший «квази-круг»
            "square": cv2.MORPH_RECT,
            "cross":  cv2.MORPH_CROSS,
        }
        morph_shape = shape_map.get(shape, cv2.MORPH_ELLIPSE)

        k = 2 * pixels + 1  # размер ядра
        kernel = cv2.getStructuringElement(morph_shape, (k, k))  # ядро для дилатации

        # Нормализуем тип входа
        if m.dtype == bool:
            src = m.astype(np.uint8) * 255
        else:
            src = (m > 0).astype(np.uint8) * 255

        dilated = cv2.dilate(src, kernel, iterations=1)  # дилатация

        return (dilated > 0) if m.dtype == bool else dilated
        

    def draw_contour(self, im, grown_mask, thickness=2, color=(255, 0, 0)):
        # --- снять batch: (B,H,W,C) -> (H,W,C), но запомнить, что он был ---
        had_batch = (im.ndim == 4 and im.shape[0] == 1)
        base = im[0] if had_batch else im
    
        # Маску привести к бинарной 8-битной: белый объект на чёрном фоне
        if grown_mask.ndim == 3:
            grown_mask_gray = cv2.cvtColor(grown_mask, cv2.COLOR_BGR2GRAY)
        else:
            grown_mask_gray = grown_mask
        grown_mask_gray = (grown_mask_gray > 0).astype(np.uint8) * 255
    
        # Контуры по бинарной маске
        contours, _ = cv2.findContours(grown_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Рисуем (OpenCV ждёт HxW[xC], массив должен быть «континуальным»)
        image_with_contours = np.ascontiguousarray(base.copy())
        draw_color = color if (image_with_contours.ndim == 3 and image_with_contours.shape[-1] in (3, 4)) else 255
        cv2.drawContours(image_with_contours, contours, -1, draw_color, thickness)
    
        # --- вернуть batch обратно при необходимости: (H,W,C) -> (1,H,W,C) ---
        if had_batch:
            image_with_contours = image_with_contours[np.newaxis, ...]
    
        return image_with_contours



    def execute(self, 
                image: torch.Tensor,                
                mask: torch.Tensor,
                thickness: int, 
                grow_by: int):

            image = image.cpu().numpy()
            mask = mask.cpu().numpy()
            print()
            print(image.shape)
        
            result_mask = self.expand_mask(mask, grow_by)
            print(result_mask.shape)
            print()
            result_image = self.draw_contour(image, result_mask, thickness)
            return (torch.from_numpy(result_image), torch.from_numpy(result_mask),)
        

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "DrawRedMaskContour": DrawRedMaskContour
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawRedMaskContour": "Draw Red Mask Contour"
}
