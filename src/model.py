import tensorflow as tf
from utils.logging import setup_logger
from typing import List
from utils.config import Config
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

logger = setup_logger("Modeling")
class Modelling:
     def load_model(self, model_path_list:List)-> dict: 
         """
         This method allows you to load keras model from specific path 
         """
         model_dict = {}
         try:
            for model_path in model_path_list:
                filename = os.path.basename(model_path)
                model_dict[filename] = tf.keras.models.load_model(model_path)
            logger.info("Model sucessfully loaded")
            return model_dict
         except Exception as e:
              logger.error("Models cannot be load", e)
              return
     
     def get_metrics(self) -> dict:
        """
        This method allows you to get json metrics from specific path
        """
        try:
            with open(Config.METRICS_DIR, "r") as f:
                metrics_dict = json.load(f)
            logger.info("Metrics sucessfully loaded")
            return metrics_dict
        except Exception as e:
            logger.error("Metrics cannot be load", e)
        
     def evaluate(self, y_true, y_pred):
        """
        this method allows you to evaluate the models
        """
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),

            #  pneumonia
            "precision_positive_pneumonia": precision_score(y_true, y_pred, pos_label=1),
            "recall_positive_pneumonia": recall_score(y_true, y_pred, pos_label=1),
            "f1_positive_pneumonia": f1_score(y_true, y_pred, pos_label=1),

            #  normal
            "precision_positive_normal": precision_score(y_true, y_pred, pos_label=0),
            "recall_positive_normal": recall_score(y_true, y_pred, pos_label=0),
            "f1_positive_normal": f1_score(y_true, y_pred, pos_label=0),
        }

          