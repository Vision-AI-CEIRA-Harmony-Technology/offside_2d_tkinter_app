"""
offside_logique package.
"""

from .pipeline import OffsideDetectionPipeline, ImageProcessor
from .models import ModelManager, PoseEstimator, PlayerMasker
from .pitch_analyzer import PitchAnalyzer, FieldSegmenter, LineDetector, VanishingPointEstimator
from .team_classifier import TeamClassifier, AttackDirectionDetector
from .offside_detector import OffsideDetector, OffsideJudge, OffsideLineComputer
from .visualization import VisualizationRenderer
from .utils import ColorUtils, GeometryUtils, BBoxUtils, KeypointUtils
from .config import *

__all__ = [
    "OffsideDetectionPipeline",
    "ImageProcessor",
    "ModelManager",
    "PoseEstimator",
    "PlayerMasker",
    "PitchAnalyzer",
    "FieldSegmenter",
    "LineDetector",
    "VanishingPointEstimator",
    "TeamClassifier",
    "AttackDirectionDetector",
    "OffsideDetector",
    "OffsideJudge",
    "OffsideLineComputer",
    "VisualizationRenderer",
    "ColorUtils",
    "GeometryUtils",
    "BBoxUtils",
    "KeypointUtils",
]
