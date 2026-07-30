from .calibration_node import ScreenCalibrationNode
from .analysis_node import ScreenAnalysisNode

NODE_CLASS_MAPPINGS = {
    "ScreenCalibration": ScreenCalibrationNode,
    "ScreenAnalysis": ScreenAnalysisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScreenCalibration": "Screen Calibration",
    "ScreenAnalysis": "Screen Analysis",
}