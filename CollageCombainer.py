import numpy as np
import torch
from PIL import Image

class CollageCombainer:
 

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {               
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),                
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Custom"

    def _process_image_to_pil(self, image_tensor: torch.Tensor, target_quad_w: int, target_quad_h: int) -> Image.Image:
        # Assuming image_tensor is (B, H, W, C) and values are 0-1
        # If no batch dim (e.g. from a loader that strips it), add it.
        # ComfyUI tensors are usually (B,H,W,C)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Take the first image from the batch
        img_tensor_first_batch = image_tensor[0] # Shape: (H, W, C)
        
        # Convert to numpy
        numpy_image = img_tensor_first_batch.cpu().numpy() # Shape: (H, W, C)
        
        # Convert to PIL Image (values 0-1 -> 0-255, uint8)
        # Ensure channel order is RGB for PIL
        pil_image = Image.fromarray((numpy_image * 255).astype(np.uint8), 'RGB')
        
        # Resize while preserving aspect ratio to fit into quadrant
        img_copy = pil_image.copy()
        img_copy.thumbnail((target_quad_w, target_quad_h), Image.Resampling.LANCZOS)
        
        # Create a background for the quadrant
        quad_bg = Image.new("RGB", (target_quad_w, target_quad_h), (0, 0, 0)) # Black background
        
        # Calculate position to paste the resized image in the center of the quadrant
        paste_x = (target_quad_w - img_copy.width) // 2
        paste_y = (target_quad_h - img_copy.height) // 2
        
        # Paste the resized image onto the quadrant background
        quad_bg.paste(img_copy, (paste_x, paste_y))
        
        return quad_bg

    def execute(self, 
                image1: torch.Tensor,
                image2: torch.Tensor,
                image3: torch.Tensor,
                image4: torch.Tensor,
                height: int,
                width: int) -> tuple[torch.Tensor]:

        # Calculate dimensions for the four quadrants to handle odd total width/height
        w1 = width // 2
        w2 = width - w1  # width for the right column of images
        
        h1 = height // 2
        h2 = height - h1 # height for the bottom row of images

        # Process each image for its specific quadrant size
        pil_img1 = self._process_image_to_pil(image1, w1, h1)       # Top-left
        pil_img2 = self._process_image_to_pil(image2, w2, h1)       # Top-right
        pil_img3 = self._process_image_to_pil(image3, w1, h2)       # Bottom-left
        pil_img4 = self._process_image_to_pil(image4, w2, h2)       # Bottom-right

        # Create the final collage
        final_collage_pil = Image.new("RGB", (width, height))

        # Paste the processed images onto the final collage
        final_collage_pil.paste(pil_img1, (0, 0))
        final_collage_pil.paste(pil_img2, (w1, 0))
        final_collage_pil.paste(pil_img3, (0, h1))
        final_collage_pil.paste(pil_img4, (w1, h1))

        # Convert the final PIL image back to a torch.Tensor
        # PIL Image (RGB) to NumPy array (H, W, C)
        final_numpy_image = np.array(final_collage_pil).astype(np.float32) / 255.0
        
        # NumPy array to torch.Tensor (H, W, C)
        final_image_tensor = torch.from_numpy(final_numpy_image)
        
        # Add batch dimension (1, H, W, C) to match ComfyUI's expected IMAGE format
        final_image_tensor = final_image_tensor.unsqueeze(0)
                
        return (final_image_tensor,)

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "CollageCombainer": CollageCombainer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CollageCombainer": "CollageCombainer"
}