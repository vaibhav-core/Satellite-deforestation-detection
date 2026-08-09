import numpy as np
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

    CLASS_COLORS={
      "AnnualCrop": (255, 165, 0),          # Orange
    "Forest": (34, 139, 34),              # Forest Green
    "HerbaceousVegetation": (124, 252, 0),# Lawn Green
    "Highway": (255, 255, 0),             # Yellow
    "Industrial": (255, 0, 255),          # Magenta
    "Pasture": (173, 255, 47),            # Yellow Green
    "PermanentCrop": (0, 128, 0),         # Green
    "Residential": (255, 0, 0),           # Red
    "River": (30, 144, 255),              # Dodger Blue
    "SeaLake": (0, 0, 255)                 #blue
    }

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

    def build_production_grid(self,predictions):

        max_y=max(p["position"][0] for p in predictions)
        max_x=max(p["position"][1] for p in predictions)

        rows = max_y // 224 + 1
        cols = max_x // 224 + 1

        grid= [[None for _ in range(cols)] for _ in range(rows)]

        for prediction in predictions:

            y, x = prediction["position"]

            row = y // 224
            col = x // 224

            grid[row][col] = prediction["class"]
        return grid

    def overlay_prediction(self,img,predictions):
        overlay = img.copy()
        for prediction in predictions:
            y,x=prediction["position"]

            color=self.CLASS_COLORS[prediction["class"]]

            rect=cv.rectangle(overlay,(x,y),(x+224,y+224),color=color,thickness=-1)

        result=cv.addWeighted(overlay,.35,img,.65,0)

        return result

    def calculate_stats(self,predictions):
        class_count={}
        for prediction in predictions:
            class_name=prediction["class"]

            if class_name not in class_count:
                class_count[class_name]=0
            class_count[class_name]+=1
        total_tiles=len(predictions)
        percetages={}
        for class_name,count in class_count.items():
            percetages[class_name]=(count/total_tiles)*100
        return percetages

classifier = LandCoverClassifier(
    "/home/vaibhav/programming_projects/python/Satellite-deforestation-detection-/eurosat_resnet50_v5-finetunning20.keras")

image = classifier.load_image("images/test.jpg")
tiles, positions = classifier.tile_img(image)

predictions = classifier.predict_tiles(tiles, positions)
grid=classifier.build_production_grid(predictions=predictions)
print(predictions[0])
print(grid)

result=classifier.overlay_prediction(img=image,predictions=predictions)

stats=classifier.calculate_stats(predictions)

for class_name,percentage in stats.items():
    print(f"{class_name} {percentage:.2f}")

cv.imshow("overlay",cv.cvtColor(result,cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows