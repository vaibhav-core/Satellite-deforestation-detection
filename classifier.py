import os
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2 as cv
import tensorflow as tf

class LandCoverClassifier:
    CLASSES_NAME=[ "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake"
    ]

    def __init__(self,modelpath):
        self.model=tf.keras.models.load_model(modelpath)
        print("Model Loaded Successfully")

    def load_image(self,imgpath):
        img=cv.imread(imgpath)
        if img is None:
            raise FileNotFoundError(imgpath)

        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

        return img

    def tile_img()

classifier = LandCoverClassifier(
    "/home/vaibhav/programming_projects/python/Satellite-deforestation-detection-/eurosat_resnet50_v5-finetunning20.keras")

image = classifier.load_image(
    "images/test.jpg"
)

print(image.shape)


