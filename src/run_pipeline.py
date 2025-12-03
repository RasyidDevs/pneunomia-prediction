import os
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent  
sys.path.insert(0, str(root_path))
from utils.logging import setup_logger
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from utils.config import Config
from src.image_processing import ImageProcessing
from src.model import Modelling
from typing import List
logger = setup_logger("Running Pipeline")


def run_preprocessing(image_list):
    logger.info("Processing images...")
    try:
        processor = ImageProcessing(image_list)
        processed = processor.preprocess()
        logger.info("Processing images done!")
        return processed
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None


def run_pipeline(image_list: List, model="ResNet50V2"):
    """
    run the complete pipeline with two branch, one is for evaluate metrics , the other to predict image
    """
    try:
        if model == "ResNet50V2":
            loaded = Modelling().load_model([Config.RESNET50V2_DIR])
            model = loaded["ResNet50V2.keras"]
            logger.info("Model ResNet50V2 successfully loaded")

        else:
            loaded = Modelling().load_model([Config.MOBILENETV2_DIR])
            model = loaded["MobileNetV2.keras"]
            logger.info("Model MobileNetV2 successfully loaded")
        image_list = run_preprocessing(image_list=image_list)
        prediction = []
        for img in image_list:   
            y_pred_prob = model.predict(img)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()  
            prediction.append(y_pred)
            logger.info("Inferencing successfully done!")
        return prediction
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return None
      