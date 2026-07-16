from .transform import (
    InstanceNameFilter,
    CircleObjectRangeFilter,
    NormalizeMultiviewImage,
    NuScenesSparse4DAdaptor,
    MultiScaleDepthMapGenerator,
    # --- Code Change ---
    # Reason: 导出 TemporalFlattenTransform 以在 pipeline 配置中通过 dict(type=...) 引用
    TemporalFlattenTransform,
    # --- End Code Change ---
)
from .augment import (
    ResizeCropFlipImage,
    BBoxRotation,
    PhotoMetricDistortionMultiViewImage,
)
from .loading import LoadMultiViewImageFromFiles, LoadPointsFromFile, LoadOccWorldLabels
from .vectorize import VectorizeMap

__all__ = [
    "InstanceNameFilter",
    "ResizeCropFlipImage",
    "BBoxRotation",
    "CircleObjectRangeFilter",
    "MultiScaleDepthMapGenerator",
    "NormalizeMultiviewImage",
    "PhotoMetricDistortionMultiViewImage",
    "NuScenesSparse4DAdaptor",
    "LoadMultiViewImageFromFiles",
    "LoadPointsFromFile",
    "VectorizeMap",
    "LoadOccWorldLabels",
    # --- Code Change ---
    # Reason: 导出 TemporalFlattenTransform
    "TemporalFlattenTransform",
    # --- End Code Change ---
]
