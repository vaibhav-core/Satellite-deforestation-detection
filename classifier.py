import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
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
        img=cv.resize(img,(224,224))

        return img

    def tile_img(self,img):
        height,width=img.shape[:2]

        pad_h=(224-(height%224))%224
        pad_w=(224-(width%224))%224

        img=cv.copyMakeBorder(img,0,pad_h,0,pad_w,cv.BORDER_CONSTANT,value=(0,0,0))

        tiles=[]
        positions=[]

        for y in range(0,img.shape[0],224):
            for x in range(0,img.shape[1],224):
                tile = img[y:y+224,x:x+224]
                tiles.append(tile)
                positions.append((y,x))
        return tiles,positions

    def predict_tiles(self, tiles, positions):

        predictions = []

        for tile, position in zip(tiles, positions):

            tile = tf.cast(tile, tf.float32)
            tile = preprocess_input(tile)

            tile = tf.expand_dims(tile, axis=0)

            prediction = self.model.predict(tile, verbose=0)

            class_index = tf.argmax(prediction[0]).numpy()

            confidence = float(tf.reduce_max(prediction[0]).numpy()) * 100

            predictions.append({
                "position": position,
                "class": self.CLASSES_NAME[class_index],
                "confidence": confidence
            })

        return predictions


classifier = LandCoverClassifier(
    "/home/vaibhav/programming_projects/python/Satellite-deforestation-detection-/eurosat_resnet50_v5-finetunning20.keras")

image = classifier.load_image("images/test.jpg")
tiles, positions = classifier.tile_img(image)

predictions = classifier.predict_tiles(tiles, positions)

print(predictions[0])