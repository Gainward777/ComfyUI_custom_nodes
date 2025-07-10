import torch
import torch.nn.functional as F

class InContextSizeCheck:
    """
    Нод для ComfyUI, который проверяет размер изображения и масок и увеличивает их,
    если большая сторона не кратна 64. Увеличение происходит добавлением 
    белых пикселей сверху или слева.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {               
                "source_image": ("IMAGE",),
                "inpainting_mask": ("MASK",),
                "source_mask": ("MASK",),                
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT")
    RETURN_NAMES = ("source_image", "inpainting_mask", "source_image_mask", "pix_line_size",)
    FUNCTION = "execute"
    CATEGORY = "Custom"

    def execute(self, source_image: torch.Tensor, 
                      inpainting_mask: torch.Tensor, 
                      source_mask: torch.Tensor):

        _, height, width, _ = source_image.shape
        
        # 1. Определяем большую сторону
        longer_side = max(width, height)
        
        # 2. Если сторона делится на 64, возвращаем исходные данные
        if longer_side % 64 == 0:
            pix_line_size = 0
            return (source_image, inpainting_mask, source_mask, pix_line_size)
            
        # 3. Вычисляем новый размер и количество пикселей для добавления
        new_longer_side = ((longer_side // 64) + 1) * 64
        pix_line_size = new_longer_side - longer_side
        
        # ComfyUI использует формат NHWC, а torch.pad удобнее работать с NCHW
        image_nchw = source_image.permute(0, 3, 1, 2)
        
        # 4. Если высота - большая сторона
        if height >= width:
            # Увеличиваем высоту, добавляя пиксели сверху
            padding = (0, 0, pix_line_size, 0) # (left, right, top, bottom)
        # 5. Если ширина - большая сторона
        else:
            # Увеличиваем ширину, добавляя пиксели слева
            padding = (pix_line_size, 0, 0, 0) # (left, right, top, bottom)

        # Применяем padding к изображению и маскам
        # Для изображения добавляем белые пиксели (значение 1.0)
        padded_image_nchw = F.pad(image_nchw, padding, "constant", 1.0)
        # Для масок добавляем нейтральные пиксели (значение 0.0 - не маскированная область)
        padded_inpainting_mask = F.pad(inpainting_mask, padding, "constant", 0.0)
        padded_source_mask = F.pad(source_mask, padding, "constant", 0.0)
        
        # Возвращаем изображение в исходный формат NHWC
        padded_image_nhwc = padded_image_nchw.permute(0, 2, 3, 1)

        return (padded_image_nhwc, padded_inpainting_mask, padded_source_mask, pix_line_size)

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "InContextSizeCheck": InContextSizeCheck
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InContextSizeCheck": "In-Context Size Check"
}