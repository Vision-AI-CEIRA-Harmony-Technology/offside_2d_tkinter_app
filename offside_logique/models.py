"""
Model Management for Offside Detection System
"""

from typing import Union, Dict, Any, Optional
import cv2
import numpy as np
# from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import onnxruntime as ort

from .config import (
    VITPOSE_MIN_CROP_SIZE, VITPOSE_CROP_SCALE_FACTOR, VITPOSE_CROP_PADDING,
    KEYPOINT_CONF_THRESH, POSE_CONF_THRESH, IoU_THRESHOLD
)
from .utils import BBoxUtils, KeypointUtils, GeometryUtils
import sys
from pathlib import Path
def resource_path(relative_path):
        try:
            base_path = Path(sys._MEIPASS)
        except Exception:
            base_path = Path(__file__).resolve().parent.parent

        return base_path / relative_path 

def preprocess_rtdetr(image):

    from PIL import Image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(image)

    orig_w, orig_h = pil.size

    img_resized = pil.resize((384, 384))

    img_data = np.array(img_resized).astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img_data = (img_data - mean) / std

    img_data = np.transpose(img_data, (2, 0, 1))

    img_data = np.expand_dims(img_data, axis=0)

    return img_data, orig_w, orig_h

def preprocess_onnx(image, input_size=640):

    h, w = image.shape[:2]

    scale = min(input_size / w, input_size / h)

    nw, nh = int(scale * w), int(scale * h)

    resized = cv2.resize(image, (nw, nh))

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)

    canvas[:nh, :nw] = resized

    img = canvas.astype(np.float32) / 255.0

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)

    return img, scale

class ModelManager:
    """Manages loading and initialization of all models."""
    
    def __init__(self, model_type: str = "vitpose", device: str = "cpu"):
        """
        Initialize model manager.
        
        Args:
            model_type: 'yolo', 'resnet', or 'vitpose'
            device: 'cpu', 'cuda', or 'mps'
        """
        self.model_type = model_type.lower()
        self.device = device
        self.model = None 

    def load_models(self) -> Any:
        """
        Load pose estimation models based on type.
        
        Returns:
            Model object or dict of models
        """

        if "resnet" in self.model_type:
            return self._load_resnet_keypoint()
        elif self.model_type == "vitpose":
            return self._load_vitpose()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    

    # def _load_resnet_keypoint(self) -> torch.nn.Module:
    #     """Load KeypointRCNN ResNet50 model."""
    #     print("Loading keypointrcnn_resnet50_fpn model")
    #     model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    #     model.eval()
    #     self.model = model
    #     return model
    
     
     
    def _load_vitpose(self) -> Dict[str, Any]:
        print(f"Loading ViTPose ONNX models on {self.device}")

        det_model = ONNXModel(
            str(resource_path("weights/best_trained.onnx"))
        )

        pose_model = ONNXModel(
            str(resource_path("weights/vitpose.onnx"))
        )

        model_dict = {
            'det_model': det_model,
            'pose_model': pose_model,
            'device': self.device
        }

        self.model = model_dict

        return model_dict
    
    def load_pitch_segmentation_model(self):

        model = ONNXModel(
            str(resource_path("weights/segmentation_rtdtr.onnx"))
        )

        return model

class ONNXModel:
    def __init__(self, model_path, providers=None):

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name

    def run(self, input_tensor):

        return self.session.run(
            None,
            {self.input_name: input_tensor}
        )

def preprocess_pose(crop):

    original_h, original_w = crop.shape[:2]

    img = cv2.resize(crop, (192, 256))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img = (img - mean) / std

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)

    return img, original_w, original_h

def decode_heatmaps(heatmaps, crop_w, crop_h):

    keypoints = []

    scores = []

    heatmaps = heatmaps[0]

    for hm in heatmaps:

        y, x = np.unravel_index(
            np.argmax(hm),
            hm.shape
        )

        conf = hm[y, x]

        x = (x / hm.shape[1]) * crop_w
        y = (y / hm.shape[0]) * crop_h

        keypoints.append((x, y))

        scores.append(float(conf))

    return keypoints, scores

class PoseEstimator:
    """Handles pose estimation with different backends."""
    
    def __init__(self, model: Any, model_type: str = "vitpose"):
        """
        Initialize pose estimator.
        
        Args:
            model: Loaded model or model dict
            model_type: Type of model
        """
        self.model = model
        self.model_type = model_type.lower()
    
    def estimate_pose(
        self,
        frame: np.ndarray,
        conf_thresh: float = POSE_CONF_THRESH,
        kp_conf_thresh: float = KEYPOINT_CONF_THRESH,
    ) -> list:
        """
        Estimate pose for all players in frame.
        
        Returns:
            List of detection dicts with bbox, keypoints, all_kps
        """
        if "resnet" in self.model_type:
            return self._estimate_resnet_pose(frame, conf_thresh, kp_conf_thresh)
        elif self.model_type == "vitpose":
            return self._estimate_vitpose_pose(frame, conf_thresh, kp_conf_thresh)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    
    
    # def _estimate_resnet_pose(
    #     self,
    #     frame: np.ndarray,
    #     conf_thresh: float,
    #     kp_conf_thresh: float
    # ) -> list:
    #     """Estimate pose using KeypointRCNN ResNet50."""
    #     tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    #     results = self.model([tensor_frame])[0]
    #     detections = []
    #     if results['keypoints'] is None:
    #         return detections
    #     boxes = results['boxes']
    #     kps_data = results['keypoints']
    #     scores = results['scores']
    #     for i, box in enumerate(boxes):
    #         if float(scores[i]) < conf_thresh:
    #             continue
    #         x1, y1, x2, y2 = map(int, box.tolist())
    #         all_kps = kps_data[i].detach().cpu().numpy()
    #         offside_kps = KeypointUtils.extract_offside_keypoints(all_kps, kp_conf_thresh)
    #         detections.append({
    #             "bbox": (x1, y1, x2, y2),
    #             "keypoints": offside_kps,
    #             "all_kps": all_kps,
    #         })
    #     return detections
    
    def _estimate_vitpose_pose(
        self,
        frame: np.ndarray,
        conf_thresh: float,
        kp_conf_thresh: float
    ) -> list:
        """Estimate pose using ViTPose."""
        det_model = self.model['det_model']
        pose_model = self.model['pose_model']
        device = self.model.get('device', 'cpu')
        H, W = frame.shape[:2]
        detections = []
        input_tensor, scale = preprocess_onnx(frame)
        outputs = det_model.run(input_tensor)

        predictions = outputs[0][0]

        boxes = []
        scores = []

        for pred in predictions:

            x1, y1, x2, y2, score, cls_id = pred

            if score < conf_thresh:
                continue

            cls_id = int(cls_id)

            # keep only player class
            if cls_id != 1:
                continue

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            # clamp to frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
        if not boxes:
            return detections
        keep = BBoxUtils.filter_boxes_by_iou(boxes, scores, IoU_THRESHOLD)
        boxes = [boxes[i] for i in keep]
        for x1, y1, x2, y2 in boxes:
            crop, x1p, y1p, scale = self._prepare_crop(frame, x1, y1, x2, y2)
            if crop is None:
                continue
            pose_results = self._run_vitpose_inference(crop, pose_model)
            if not pose_results:
                continue
            best_person = max(
                pose_results,
                key=lambda p: np.mean(p["scores"])
            )
            det = self._build_detection_from_vitpose(
                best_person, x1, y1, x2, y2, x1p, y1p, crop, scale, kp_conf_thresh
            )
            if det:
                detections.append(det)
        return detections
    
    def _prepare_crop(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> tuple:
        """
        Prepare cropped region for ViTPose.
        
        Returns:
            (crop, x1p, y1p, scale) or (None, None, None, None)
        """
        H, W = frame.shape[:2]
        pad = VITPOSE_CROP_PADDING
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return None, None, None, None
        h_c, w_c = crop.shape[:2]
        scale = 1
        if w_c < VITPOSE_MIN_CROP_SIZE or h_c < VITPOSE_MIN_CROP_SIZE:
            scale = VITPOSE_CROP_SCALE_FACTOR
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop, x1p, y1p, scale
    
    def _run_vitpose_inference(
        self,
        crop: np.ndarray,
        pose_model,
        processor=None,
        device=None
    ) -> list:

        input_tensor, crop_w, crop_h = preprocess_pose(crop)

        outputs = pose_model.run(input_tensor)

        heatmaps = outputs[0]

        keypoints, scores = decode_heatmaps(
            heatmaps,
            crop_w,
            crop_h
        )

        return [{
            "keypoints": keypoints,
            "scores": scores
        }]
    
    def _build_detection_from_vitpose(
        self,
        person_result: dict,
        x1: int, y1: int, x2: int, y2: int,
        x1p: int, y1p: int,
        crop: np.ndarray,
        scale: float,
        kp_conf_thresh: float
    ) -> Optional[dict]:
        """Build detection dict from ViTPose result."""
        keypoints = person_result["keypoints"]
        scores_kp = person_result["scores"]
        all_kps = np.zeros((17, 3))
        valid_kp_count = 0
        for i, ((kx, ky), score) in enumerate(zip(keypoints, scores_kp)):
            if score >= kp_conf_thresh:
                kx_crop = kx / scale
                ky_crop = ky / scale
                gx = kx_crop + x1p
                gy = ky_crop + y1p
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                margin_x = bbox_w * 0.15
                margin_y = bbox_h * 0.15
                if (x1 - margin_x <= gx <= x2 + margin_x and 
                    y1 - margin_y <= gy <= y2 + margin_y):
                    all_kps[i] = [gx, gy, score]
                    valid_kp_count += 1
        if valid_kp_count < 2:
            return None
        offside_kps = KeypointUtils.extract_offside_keypoints(all_kps, kp_conf_thresh)
        return {
            "bbox": (x1, y1, x2, y2),
            "keypoints": offside_kps,
            "all_kps": all_kps,
        }


class PlayerMasker:
    """Mask and hide players for clean pitch line detection."""
    
    @staticmethod
    def create_player_mask(detections: list, frame_shape: tuple) -> np.ndarray:
        """
        Create mask for all player bounding boxes.
        
        Args:
            detections: List of detection dicts
            frame_shape: Shape of frame (H, W, C)
            
        Returns:
            Binary mask
        """
        H, W = frame_shape[:2]
        player_mask = np.zeros((H, W), dtype=np.uint8)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(player_mask, (x1, y1), (x2, y2), 255, -1)
        kernel = np.ones((7, 7), np.uint8)
        player_mask = cv2.dilate(player_mask, kernel, iterations=2)
        return player_mask
    
    @staticmethod
    def hide_players(frame: np.ndarray, player_mask: np.ndarray) -> np.ndarray:
        """
        Remove players from frame using inpainting.
        
        Args:
            frame: Original frame
            player_mask: Binary mask of players
            
        Returns:
            Cleaned frame
        """
        return cv2.inpaint(frame, player_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
