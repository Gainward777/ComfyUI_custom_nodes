import numpy as np
import torch
from PIL import Image
import cv2

class MaskCompair:
    """
    Нод для ComfyUI, который масштабирует и вставляет одну маску в другую,
    ориентируясь по центру нижней грани ограничивающих прямоугольников.

    Шаги:
    1) Вычисление ограничивающих прямоугольников для обеих масок
    2) Масштабирование inpainting_mask относительно source_mask
    3) Вставка отмасштабированной маски в исходную по центру нижней грани
    4) Возврат итоговой маски
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {               
                "source_mask": ("MASK",),
                "inpainting_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("final_mask", "scaled_inpainting_mask",)
    FUNCTION = "execute"
    CATEGORY = "Custom"

    def get_bounding_box(self, mask):
        """Вычисляет ограничивающий прямоугольник для маски"""
        # Конвертируем тензор в numpy и бинаризуем
        mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
        
        # Проверяем размерность маски
        if len(mask_np.shape) != 2:
            mask_np = mask_np.squeeze()
            if len(mask_np.shape) != 2:
                raise ValueError(f"Unexpected mask shape: {mask_np.shape}. Expected 2D array.")
        
        # Находим ненулевые элементы
        nonzero = np.nonzero(mask_np)
        if len(nonzero[0]) == 0:
            return None
        
        y_coords, x_coords = nonzero[0], nonzero[1]
        return {
            'x': int(x_coords.min()),
            'y': int(y_coords.min()),
            'width': int(x_coords.max() - x_coords.min() + 1),
            'height': int(y_coords.max() - y_coords.min() + 1),
            'bottom_y': int(y_coords.max()),
            'bottom_center_x': int((x_coords.min() + x_coords.max()) // 2)
        }

    def scale_mask(self, mask, target_bbox, source_bbox):
        """Масштабирует маску, сохраняя пропорции"""
        # Определяем коэффициент масштабирования по меньшей стороне
        source_min_side = min(source_bbox['width'], source_bbox['height'])
        target_min_side = min(target_bbox['width'], target_bbox['height'])
        
        # Проверяем, что размеры больше нуля
        if source_min_side <= 0 or target_min_side <= 0:
            raise ValueError(f"Invalid bounding box dimensions: source_min_side={source_min_side}, target_min_side={target_min_side}")
            
        scale_factor = float(source_min_side) / float(target_min_side)
        
        # Защита от нулевого или отрицательного масштаба
        if scale_factor <= 0:
            raise ValueError(f"Invalid scale factor: {scale_factor}")

        # Конвертируем тензор в numpy array для обработки
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Проверяем размерность маски
        if len(mask_np.shape) != 2:
            mask_np = mask_np.squeeze()
        
        # Получаем текущие размеры
        current_height, current_width = mask_np.shape
        
        # Вычисляем новые размеры (минимум 1 пиксель)
        new_width = max(1, int(current_width * scale_factor))
        new_height = max(1, int(current_height * scale_factor))
        
        # Масштабируем маску
        try:
            scaled_mask = cv2.resize(mask_np, (new_width, new_height), 
                                   interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"Debug info: current_size=({current_width}, {current_height}), "
                  f"new_size=({new_width}, {new_height}), scale_factor={scale_factor}")
            raise e
        
        return torch.from_numpy(scaled_mask).float()

    def execute(self, source_mask: torch.Tensor, inpainting_mask: torch.Tensor):
        print(f"Source mask shape: {source_mask.shape}, Inpainting mask shape: {inpainting_mask.shape}")
        
        # Убираем лишнюю размерность batch если она есть
        if len(source_mask.shape) == 3 and source_mask.shape[0] == 1:
            source_mask = source_mask.squeeze(0)
        if len(inpainting_mask.shape) == 3 and inpainting_mask.shape[0] == 1:
            inpainting_mask = inpainting_mask.squeeze(0)
            
        # Конвертируем в numpy и бинаризуем маски
        source_np = (source_mask.cpu().numpy() > 0.5).astype(np.uint8)
        inpainting_np = (inpainting_mask.cpu().numpy() > 0.5).astype(np.uint8)
            
        # Получаем ограничивающие прямоугольники
        source_bbox = self.get_bounding_box(source_mask)
        inpainting_bbox = self.get_bounding_box(inpainting_mask)
        
        if source_bbox is None or inpainting_bbox is None:
            raise ValueError("One or both masks are empty")

        # Определяем коэффициент масштабирования по меньшей стороне
        source_min_side = min(source_bbox['width'], source_bbox['height'])
        target_min_side = min(inpainting_bbox['width'], inpainting_bbox['height'])
        if target_min_side == 0: # Защита от деления на ноль
            raise ValueError("Inpainting mask bounding box has zero width or height.")
        scale_factor = float(source_min_side) / float(target_min_side)

        # Масштабируем inpainting_mask целиком
        new_width = int(inpainting_np.shape[1] * scale_factor)
        new_height = int(inpainting_np.shape[0] * scale_factor)
        if new_width <= 0 or new_height <= 0: # Защита от нулевых размеров для resize
            # Если вся маска inpainting_np пуста, inpainting_bbox будет None и мы не дойдем сюда.
            # Это может случиться если scale_factor очень мал и исходные размеры малы.
            # В таком случае, вставлять нечего.
            final_mask_tensor = torch.from_numpy(source_np).float()
            if len(final_mask_tensor.shape) == 2:
                final_mask_tensor = final_mask_tensor.unsqueeze(0)
            return (final_mask_tensor,)
            
        scaled_inpainting = cv2.resize(inpainting_np, (new_width, new_height), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Бинаризуем результат масштабирования
        scaled_inpainting = (scaled_inpainting > 0.5).astype(np.uint8)

        # Целевая точка на source_mask (центр нижней грани bbox)
        target_paste_point_x = source_bbox['bottom_center_x']
        target_paste_point_y = source_bbox['bottom_y']

        # Якорная точка на inpainting_mask (центр нижней грани bbox)
        # Координаты этой точки относительно верхнего левого угла inpainting_np
        anchor_x_in_inpainting_np = inpainting_bbox['bottom_center_x']
        anchor_y_in_inpainting_np = inpainting_bbox['bottom_y']

        # Координаты этой якорной точки после масштабирования всего холста inpainting_np
        # (относительно верхнего левого угла scaled_inpainting)
        scaled_anchor_x = int(anchor_x_in_inpainting_np * scale_factor)
        scaled_anchor_y = int(anchor_y_in_inpainting_np * scale_factor)

        # Рассчитываем paste_x, paste_y (координаты верхнего левого угла scaled_inpainting на final_mask)
        # так, чтобы scaled_anchor совпал с target_paste_point
        paste_x = target_paste_point_x - scaled_anchor_x
        paste_y = target_paste_point_y - scaled_anchor_y
        
        print(f"Source bbox bottom center (target_paste_point): ({target_paste_point_x}, {target_paste_point_y})")
        print(f"Inpainting bbox bottom center (original): ({anchor_x_in_inpainting_np}, {anchor_y_in_inpainting_np})")
        print(f"Inpainting bbox bottom center (scaled, relative to scaled_inpainting): ({scaled_anchor_x}, {scaled_anchor_y})")
        print(f"Calculated paste_x, paste_y for top-left of scaled_inpainting: ({paste_x}, {paste_y})")
        
        # Копируем исходную маску и создаем пустую для второй маски
        final_mask = source_np.copy()
        inpainting_only_mask_np = np.zeros_like(source_np)
        
        # Определяем область вставки на final_mask и на scaled_inpainting
        # Координаты на final_mask
        fm_start_y = max(0, paste_y)
        fm_end_y = min(final_mask.shape[0], paste_y + new_height)
        fm_start_x = max(0, paste_x)
        fm_end_x = min(final_mask.shape[1], paste_x + new_width)

        # Соответствующие координаты на scaled_inpainting
        si_start_y = fm_start_y - paste_y
        si_end_y = fm_end_y - paste_y
        si_start_x = fm_start_x - paste_x
        si_end_x = fm_end_x - paste_x
        
        # Проверяем, что есть что вставлять
        if fm_start_y < fm_end_y and fm_start_x < fm_end_x and \
           si_start_y < si_end_y and si_start_x < si_end_x:
            
            region_to_paste = scaled_inpainting[si_start_y:si_end_y, si_start_x:si_end_x]
            current_region_in_final = final_mask[fm_start_y:fm_end_y, fm_start_x:fm_end_x]
            
            # Убедимся, что формы совпадают (важно из-за округлений/обрезок)
            if region_to_paste.shape == current_region_in_final.shape:
                # Вставляем в итоговую маску (объединение)
                final_mask[fm_start_y:fm_end_y, fm_start_x:fm_end_x] = \
                    np.logical_or(current_region_in_final, region_to_paste).astype(np.uint8)
                # Вставляем в маску только для inpainting
                inpainting_only_mask_np[fm_start_y:fm_end_y, fm_start_x:fm_end_x] = region_to_paste
            else:
                print(f"Warning: Shape mismatch during paste. Region to paste: {region_to_paste.shape}, Target region: {current_region_in_final.shape}")

        # Конвертируем обратно в тензоры
        final_mask_tensor = torch.from_numpy(final_mask).float()
        inpainting_only_tensor = torch.from_numpy(inpainting_only_mask_np).float()

        # Убеждаемся, что возвращаем тензоры нужной размерности
        if len(final_mask_tensor.shape) == 2:
            final_mask_tensor = final_mask_tensor.unsqueeze(0)
        if len(inpainting_only_tensor.shape) == 2:
            inpainting_only_tensor = inpainting_only_tensor.unsqueeze(0)

        return (final_mask_tensor, inpainting_only_tensor,)

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "MaskCompair": MaskCompair
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCompair": "MaskCompair"
}