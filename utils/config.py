from pathlib import Path
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Main Folder
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    SRC_DIR = BASE_DIR / "src"

    # Model 
    MODEL_DIR = ARTIFACTS_DIR / 'model'
    RESNET50V2_DIR = MODEL_DIR / "ResNet50V2.keras"
    MOBILENETV2_DIR = MODEL_DIR / "MobileNetV2.keras"
    
    # Metrics 
    METRICS_DIR = ARTIFACTS_DIR / "metrics.json"

    # MLFlow
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME = "pneunomia_prediction"

    #API 
    API_TITLE = "Pneunomia Prediction API"
    API_DESCRIPTION = "API for classifying chest X-ray images to detect pneumonia using a trained deep learning model."
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000