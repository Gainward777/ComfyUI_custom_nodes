import numpy as np
import torch
from PIL import Image
import cv2

class MaskSmoother:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {               
                "mask": ("MASK",),
                
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "Custom"

   

    def execute(self, mask: torch.Tensor):
        # Convert tensor to NumPy array
        # Assuming mask is [N, H, W] or [H, W], we'll take the first channel if N > 1
        # and squeeze if it's [1, H, W]
        if mask.ndim == 3:
            if mask.shape[0] > 1:
                # If multiple masks, how to combine? For now, take the first one.
                # Or sum them up and threshold? Let's assume one mask for now or take the first.
                mask_np = mask[0].cpu().numpy() 
            else: # Shape is [1, H, W]
                mask_np = mask.squeeze(0).cpu().numpy()
        elif mask.ndim == 2: # Shape is [H, W]
            mask_np = mask.cpu().numpy()
        else:
            raise ValueError(f"Unsupported mask dimensions: {mask.shape}")

        # Ensure it's a binary image (0 or 255) of type uint8
        # Values in the tensor are expected to be 0.0 to 1.0
        mask_np = (mask_np * 255).astype(np.uint8)
        
        # 1. Find contours
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # If no contours, return the original mask or an empty one of the same shape
            return (mask,)

        # Combine all contours into a single list of points for convex hull
        all_points = np.concatenate(contours, axis=0)

        # 2. Get convex hull
        hull_points = cv2.convexHull(all_points)

        # 3. Create a new mask by drawing and filling the convex hull
        filled_hull_mask = np.zeros_like(mask_np)
        cv2.drawContours(filled_hull_mask, [hull_points], -1, (255), thickness=cv2.FILLED)

        # 4. Make the bottom edge flat
        # Find the lowest y-coordinate (max y) of the hull
        if hull_points.size > 0:
            max_y = hull_points[:, :, 1].max()

            # Create the final flat bottom mask
            final_flat_mask = np.zeros_like(mask_np)
            
            # Get the x-coordinates spanned by the hull
            min_x_hull = hull_points[:, :, 0].min()
            max_x_hull = hull_points[:, :, 0].max()

            for x in range(min_x_hull, max_x_hull + 1):
                # Find the uppermost point of the hull at this x-coordinate
                col_pixels_in_hull = hull_points[hull_points[:,:,0] == x]
                if col_pixels_in_hull.size > 0:
                    min_y_at_x = col_pixels_in_hull[:,1].min()
                    # Fill from min_y_at_x down to max_y
                    final_flat_mask[min_y_at_x:max_y + 1, x] = 255
            
            # If the hull was very thin or had gaps, the above might not be enough.
            # A more robust way to make the bottom flat from the filled_hull_mask:
            # Find all points in the filled hull
            hull_y_coords, hull_x_coords = np.where(filled_hull_mask > 0)
            if hull_y_coords.size > 0:
                max_y_overall = hull_y_coords.max() # This is the bottom of the object
                
                # Create a new mask for the flat bottom operation
                processed_mask_np = np.zeros_like(mask_np)
                
                # Iterate over each column that contains part of the hull
                unique_x_in_hull = np.unique(hull_x_coords)
                for x_coord in unique_x_in_hull:
                    # Get y-coordinates of the hull in this column
                    ys_in_col = hull_y_coords[hull_x_coords == x_coord]
                    if ys_in_col.size > 0:
                        min_y_in_col = ys_in_col.min()
                        # Fill from the top of the hull in this column down to max_y_overall
                        processed_mask_np[min_y_in_col : max_y_overall + 1, x_coord] = 255
                mask_np = processed_mask_np
            else: # No points in hull, use original or empty
                 mask_np = filled_hull_mask # or np.zeros_like(mask_np)

        else: # No hull points, something went wrong or mask was empty
            mask_np = np.zeros_like(mask_np)


        # Convert NumPy array back to tensor
        # Ensure the tensor is in the [H, W] or [1, H, W] format expected by ComfyUI masks
        final_mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
        
        # If original mask was [H,W], then remove the batch dim
        if mask.ndim == 2:
            final_mask_tensor = final_mask_tensor.squeeze(0)
        elif mask.ndim == 3 and mask.shape[0] > 1 : # If original was [N,H,W]
             # The output is singular mask, so we should return [1,H,W]
             # which is what unsqueeze(0) does if mask_np is [H,W]
             pass


        return (final_mask_tensor,)

# Регистрация нода в ComfyUI
NODE_CLASS_MAPPINGS = {
    "MaskSmoother": MaskSmoother
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSmoother": "MaskSmoother"
}