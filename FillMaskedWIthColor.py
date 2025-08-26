import torch


class FillMaskedWIthColor:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "red": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "green": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "blue": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Custom"

    def execute(self, image: torch.Tensor, mask: torch.Tensor, red: int, green: int, blue: int):
        # Ensure mask has shape [B, H, W]
        if mask.dim() == 2:
            mask_bhw = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask_bhw = mask
        else:
            raise ValueError(f"Unsupported mask dimensions: {mask.shape}")

        batch, height, width, channels = image.shape

        # Expand mask batch if needed
        if mask_bhw.shape[0] == 1 and batch > 1:
            mask_bhw = mask_bhw.expand(batch, -1, -1)

        # Convert to image dtype and device, and normalize/clamp
        mask_bhw = mask_bhw.to(device=image.device, dtype=image.dtype).clamp(0, 1)
        mask_bhw = mask_bhw.unsqueeze(-1)  # [B, H, W, 1]

        # Build color tensor in range [0,1]
        color = torch.tensor([red, green, blue], device=image.device, dtype=image.dtype) / 255.0
        color = color.view(1, 1, 1, 3).expand(batch, height, width, 3)

        result = image * (1.0 - mask_bhw) + color * mask_bhw
        return (result,)


# Registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FillMaskedWIthColor": FillMaskedWIthColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillMaskedWIthColor": "FillMaskedWIthColor"
}


