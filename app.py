from fastapi import FastAPI, HTTPException,File, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from utils.config import Config
from utils.logging import setup_logger
import uvicorn
from pydantic import Field
logger = setup_logger('api')
from pydantic import BaseModel
from typing import List, Optional
from src.image_processing import ImageProcessing
from src.model import Modelling
from src.run_pipeline import run_pipeline
app = FastAPI(title=Config.API_TITLE, description=Config.API_DESCRIPTION, version=Config.API_VERSION)

class ImageRequest(BaseModel):
    image_list: List
    model: Optional[str] = Field("ResNet50V2", description="Model to use for prediction")

@app.post("/predict")
async def predict(
    image_list: List[UploadFile] = File(...),
    model: Optional[str] = "ResNet50V2"
):
    try:
        logger.info("Prediction request received")

        # convert UploadFile → raw binary file-like object
        images = [img.file for img in image_list]

        preds = run_pipeline(
            image_list=images,
            model=model
        )
        prediction = [int(p[0]) for p in preds]

        return {
            "status": "success",
            "model_used": model,
            "predictions": prediction
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))