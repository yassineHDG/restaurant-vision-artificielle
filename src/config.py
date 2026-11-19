from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGIONS_CONFIG_PATH = PROJECT_ROOT / "config" / "regions.json"
CAMERAS_CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"))

    VIDEO_PATH: str = "videos/restaurant-demo.mp4"
    DINING_VIDEO_PATH: str = "videos/restaurant-demo.mp4"
    ENTRANCE_VIDEO_PATH: str = ""
    MODEL_PATH: str = "models/yolo26n-pose.pt"
    ROLE_MODEL_PATH: str = ""
    ROLE_MODEL_CONF: float = Field(0.45, ge=0, le=1)
    CALIBRATION_MODEL_PATH: str = "models/yolo26n-seg.pt"
    CONF: float = 0.40
    IOU: float = 0.50
    IMGSZ: int = 640
    DEVICE: str = "cuda:0"
    REQUIRE_CUDA: bool = True
    HALF_PRECISION: bool = True
    TARGET_FPS: float = Field(10.0, gt=0)
    INFERENCE_BATCH_SIZE: int = Field(2, ge=1, le=2)
    QUEUE_DWELL_SECONDS: float = Field(3.0, ge=0)
    STAFF_MARKER_HSV_LOWER: str = "80,80,60"
    STAFF_MARKER_HSV_UPPER: str = "110,255,255"
    STAFF_MARKER_MIN_RATIO: float = Field(0.025, ge=0, le=1)
    TABLE_ASSIGNMENT_MARGIN_RATIO: float = Field(0.012, ge=0, le=0.1)
    TABLE_OCCLUSION_HOLD_SECONDS: float = Field(1.5, ge=0, le=10)
    COUNT_INCREASE_DELAY: float = 2.0
    COUNT_DECREASE_DELAY: float = 10.0
    COUNT_EMPTY_DELAY: float = 10.0
    COUNT_HISTORY_SIZE: int = 10
    SAVE_OUTPUT_VIDEO: bool = False
    OUTPUT_VIDEO_PATH: str = "videos/output/restaurant_analysis_output.mp4"
    RESIZE_VIDEO: bool = False
    VIDEO_WIDTH: int = Field(1280, gt=0)
    SHOW_DASHBOARD: bool = True
    DASHBOARD_WIDTH: int = Field(200, gt=0)

    @field_validator("OUTPUT_VIDEO_PATH")
    @classmethod
    def _resolve_output_path(cls, value: str) -> str:
        """Relative output paths resolve against the project root, not the CWD."""
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    @staticmethod
    def parse_hsv(value: str) -> tuple[int, int, int]:
        parts = tuple(int(part.strip()) for part in value.split(","))
        if len(parts) != 3:
            raise ValueError("HSV settings require three comma-separated integers")
        return parts


settings = Settings()
