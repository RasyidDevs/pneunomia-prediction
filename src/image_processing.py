import tensorflow as tf
from PIL import Image
import numpy as np
from utils.logging import setup_logger

logger = setup_logger("Image Preprocessing")

class ImageProcessing:
    def __init__(self, images):
        """
        images: list of UploadedFile (Streamlit), or single UploadedFile
        """
        try:
            if not isinstance(images, list):
                self.images = [images]
            else:
                self.images = images

            logger.info("Successfully loaded uploaded images")
        except Exception as e:
            logger.error(f"Failed to load images: {e}")

    def preprocess(self, target_size=(160, 160)):
        """
        Returns: list of tensorflow tensors (each size: (1, H, W, C))
        """
        processed = []

        try:
            for file in self.images:
                img = Image.open(file).convert("RGB")
                img = img.resize(target_size)
                img_tensor = tf.expand_dims(img, axis=0)
                processed.append(img_tensor)

            logger.info("Successfully preprocessed images")
            return processed

        except Exception as e:
            logger.error(f"Failed to preprocess images: {e}")
            return None
