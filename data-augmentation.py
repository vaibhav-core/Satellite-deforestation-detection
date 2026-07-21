import os
from tensorflow.keras.applications.resnet50 import preprocess_input
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

import matplotlib.pyplot as plt

data_path=os.path.join(os.getcwd(),"EuroSAT")

train_ds=tf.keras.utils.image_dataset_from_directory(data_path,validation_split=0.2,subset="training",seed=42,image_size=(224,224),batch_size=32)
validation_ds=tf.keras.utils.image_dataset_from_directory(data_path,validation_split=0.2,subset="validation",seed=42,image_size=(224,224),batch_size=32)

# print(train_ds.class_names)
# print(len(train_ds.class_names))

# plt.figure(figsize=(10,10))

# for images,lables in train_ds.take(1):
#     for i in range(9):
#         ax=plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(train_ds.class_names[lables[i]])
#         plt.axis("off")

# plt.show()

data_aug=tf.keras.Sequential([tf.keras.Input(shape=(224,224,3)),layers.RandomFlip("horizontal_and_vertical"),layers.RandomRotation(0.2),layers.RandomZoom(0.2)])

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)

validation_ds = validation_ds.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)

AUTOTUNE=tf.data.AUTOTUNE

train_ds=train_ds.prefetch(AUTOTUNE)

validation_ds=validation_ds.prefetch(AUTOTUNE)

base_model=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))

base_model.trainable=False

model=models.Sequential([data_aug,base_model,layers.GlobalAveragePooling2D(),layers.Dense(256,activation="relu"),layers.Dropout(0.3),layers.Dense(10,activation="softmax")])

print(tf.__version__)
print(tf.keras.__version__)

for images, labels in train_ds.take(1):
    print(images.dtype)
    print(tf.reduce_min(images).numpy())
    print(tf.reduce_max(images).numpy())

model.summary()

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

history=model.fit(train_ds,validation_data=validation_ds,epochs=15,callbacks=[early_stop])

loss, accuracy = model.evaluate(validation_ds)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

model.save("eurosat_resnet50_v3.keras")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss")

plt.show()