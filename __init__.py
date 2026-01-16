from .InsertObjectNode import InsertObjectNode
from .MaskCompair import MaskCompair
from .MaskSmoother import MaskSmoother
from .CollageCombainer import CollageCombainer
from .ImageCompair import ImageCompair
from .InContextSizeCheck import InContextSizeCheck
from .DrawRedMaskContour import DrawRedMaskContour
from .FillMaskedWIthColor import FillMaskedWIthColor
from .LoraLoaderFromURL import LoraLoaderFromURL

NODE_CLASS_MAPPINGS = {
    "InsertObjectNode": InsertObjectNode,
    "MaskCompair": MaskCompair,
    "MaskSmoother": MaskSmoother,
    "ImageCompair": ImageCompair,
    "CollageCombainer": CollageCombainer,
    "InContextSizeCheck": InContextSizeCheck,
    "DrawRedMaskContour": DrawRedMaskContour,
    "FillMaskedWIthColor": FillMaskedWIthColor,
    "LoraLoaderFromURL": LoraLoaderFromURL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsertObjectNode": "Insert Object Node",
    "MaskCompair": "MaskCompair",
    "MaskSmoother": "MaskSmoother",
    "ImageCompair": "ImageCompair",
    "CollageCombainer": "CollageCombainer",
    "InContextSizeCheck": "InContextSizeCheck",
    "DrawRedMaskContour": "Draw Red Mask Contour",
    "FillMaskedWIthColor": "FillMaskedWIthColor",
    "LoraLoaderFromURL": "Lora Loader From URL"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
