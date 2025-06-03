import numpy as np
import torch
from PIL import Image
import cv2

class InsertObjectNode:
    """
    Нод для ComfyUI, который вставляет область из inpainted_image в orig_image
    по заданной маске (тип MASK).

    Шаги:
    1) Приводим все входы к одному размеру (resize_to x resize_to).
    2) Извлекаем объект из inpainted_image по маске.
    3) Вставляем объект в orig_image.
    4) Возвращаем итоговое изображение.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orig_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "dilation_pixels": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                }),
                "blur_ksize": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Custom"
    
    def smooth_and_dilate_mask(self, mask: torch.Tensor, 
                               dilation_pixels: int, 
                               blur_ksize: int = 7, 
                               threshold: float = 0.5) -> torch.Tensor:
            """
            Расширяет маску на заданное количество пикселей и сглаживает её края с помощью гауссова размытия.
            
            Параметры:
              mask: torch.Tensor размера [1, H, W] с бинарными значениями (0 и 1)
              dilation_pixels: число пикселей, на которое расширяется маска по всем сторонам
              blur_ksize: размер ядра для гауссова размытия (должен быть нечетным, например, 7)
            
            Возвращает:
              Новая маска в виде torch.Tensor размера [1, H, W] с плавающими значениями в диапазоне [0, 1].
            """
            # Переводим маску в numpy (удаляем размерность батча) и приводим значения к 0-255
            mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    
            # Создаем ядро для дилатации: размер ядра равен (2*dilation_pixels+1)
            kernel = np.ones((2 * dilation_pixels + 1, 2 * dilation_pixels + 1), np.uint8)
            
            # Применяем дилатацию для расширения маски
            dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
            
            # Применяем гауссово размытие для сглаживания краёв
            blurred_mask = cv2.GaussianBlur(dilated_mask, (blur_ksize, blur_ksize), 0)
            
            # Нормализуем маску к диапазону [0, 1]
            blurred_mask = blurred_mask.astype(np.float32) / 255.0
            
            # Применяем пороговую операцию для восстановления бинарной маски
            # Значения выше порога становятся 1, остальные — 0.
            _, thresholded_mask = cv2.threshold(blurred_mask, threshold, 1, cv2.THRESH_BINARY)
            
            # Преобразуем обратно в torch.Tensor и восстанавливаем размерность [1, H, W]
            result_mask = torch.from_numpy(thresholded_mask).unsqueeze(0)
            
            return result_mask
        

    def execute(self, 
                orig_image: torch.Tensor,
                inpainted_image: torch.Tensor,
                mask: torch.Tensor,
                dilation_pixels: int, 
                blur_ksize: int,
                threshold: float) -> torch.Tensor:
        
            """
            Параметры:
              image – исходное изображение (torch.Tensor)
              inpainted_image – изображение с вкрашенным объектом (torch.Tensor)
              mask – бинарная маска области объекта (torch.Tensor), где 1 указывает на область объекта,
                     а 0 – на фон. Предполагается, что все тензоры имеют одинаковые размеры.
            
            Возвращает:
              result – изображение с вставленным объектом.
            """            
            
            # Приведение маски к тому же типу данных, что и изображение (например, float)
            mask = mask.to(orig_image.dtype)#.unsqueeze(-1)

            mask=self.smooth_and_dilate_mask(mask, dilation_pixels, blur_ksize, threshold)  

            mask = mask.unsqueeze(-1)
            # Вырезаем объект из inpainted_image по маске
            extracted_object = inpainted_image * mask
            
            # Берем области исходного изображения, где нет объекта
            background = orig_image * (1 - mask)
            
            # Объединяем вырезанный объект с фоном
            result = background + extracted_object
            return (result,)
        

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "InsertObjectNode": InsertObjectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsertObjectNode": "Insert Object Node"
}