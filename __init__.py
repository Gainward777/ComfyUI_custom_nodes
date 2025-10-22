from .InsertObjectNode import InsertObjectNode
from .MaskCompair import MaskCompair
from .MaskSmoother import MaskSmoother
from .CollageCombainer import CollageCombainer
from .ImageCompair import ImageCompair
from .InContextSizeCheck import InContextSizeCheck
from .DrawRedMaskContour import DrawRedMaskContour

NODE_CLASS_MAPPINGS = {
    "InsertObjectNode": InsertObjectNode,
    "MaskCompair": MaskCompair,
    "MaskSmoother": MaskSmoother,
    "ImageCompair": ImageCompair,
    "CollageCombainer": CollageCombainer,
    "InContextSizeCheck": InContextSizeCheck,
    "DrawRedMaskContour": DrawRedMaskContour
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsertObjectNode": "Insert Object Node",
    "MaskCompair": "MaskCompair",
    "MaskSmoother": "MaskSmoother",
    "ImageCompair": "ImageCompair",
    "CollageCombainer": "CollageCombainer",
    "InContextSizeCheck": "InContextSizeCheck",
    "DrawRedMaskContour": "Draw Red Mask Contour"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]