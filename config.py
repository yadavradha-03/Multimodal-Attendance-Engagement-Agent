"""
Configuration settings for Multimodal Attendance & Engagement Agent.
All parameters can be tuned or overridden dynamically via the UI.
"""
from dataclasses import dataclass
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

@dataclass
class VisionConfig:
    # Face detection
    score_threshold: float = 0.5
    nms_threshold: float = 0.3
    input_size: tuple = (320, 320)
    
    # Head Pose & Gaze thresholds (degrees)
    yaw_focused_threshold: float = 22.0     # |yaw| <= 22 is focused forward
    pitch_focused_min: float = -20.0        # looking down threshold
    pitch_focused_max: float = 20.0         # looking up threshold
    
    # Motion & Inactivity
    inactivity_seconds_threshold: float = 12.0  # seconds stationary before flagging inactivity
    motion_energy_scale: float = 1.0
    
    # Privacy
    blur_faces: bool = False               # Edge privacy preservation toggle
    blur_kernel_size: int = 31

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_length_sec: float = 0.5          # window for audio feature extraction
    vad_energy_threshold: float = 0.015    # RMS threshold for speech detection
    zcr_threshold: float = 0.12            # Zero crossing rate threshold
    spectral_centroid_speech_min: float = 300.0  # Hz
    spectral_centroid_speech_max: float = 3500.0 # Hz

@dataclass
class PresenceConfig:
    # Sustained presence criteria (Scenario 1)
    # Attendance is marked based on sustained presence rather than single-frame detection
    min_sustained_seconds: float = 8.0     # minimum continuous presence for short demos / configurable
    min_presence_ratio: float = 0.60       # must be present for >= 60% of observed time
    max_coasting_seconds: float = 4.0      # allow brief disappearance/occlusion without resetting
    tardy_threshold_seconds: float = 15.0  # first seen after this -> Late Arrival

@dataclass
class FusionConfig:
    # Scenario 2: Engagement Dynamics Fusion Weights
    weight_visual_attention: float = 0.50
    weight_motion_vitality: float = 0.25
    weight_audio_synergy: float = 0.25
    
    # Attention drop & peak detection thresholds
    attention_drop_threshold: float = 45.0  # % engagement below which is an attention drop
    attention_drop_min_duration: float = 5.0 # seconds
    interaction_peak_threshold: float = 75.0 # % engagement above which is peak
    
@dataclass
class AdvisorConfig:
    # Scenario 3: AI-Driven Teaching Suggestions thresholds
    monologue_alert_seconds: float = 20.0   # prolonged continuous lecture without interaction
    inactivity_ratio_alert: float = 0.35    # > 35% of class inactive -> trigger physical/interactive prompt
    low_engagement_alert_threshold: float = 45.0
