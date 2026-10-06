"""Native-scale ROI semantic classification, separate from binary detection."""

from satchange.semantic.roi import (
    ROI,
    ROISettings,
    align_whole_image_probabilities,
    extract_rois,
)

__all__ = ["ROI", "ROISettings", "align_whole_image_probabilities", "extract_rois"]
