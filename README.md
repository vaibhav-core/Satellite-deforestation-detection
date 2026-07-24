# EuroSAT Land Cover Classification

## Model
- Base Model: ResNet50 (ImageNet Pretrained)
- Dataset: EuroSAT RGB
- Classes: 10
- Input Size: 224 × 224
- Batch Size: 32
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

---

# Experiment 1

### Objective
Train ResNet50 with frozen backbone.

### Configuration
- Preprocessing: Rescaling(1./255)
- ResNet Frozen: Yes
- Dense Head:
    Dense(256, ReLU)
    Dropout(0.3)
    Dense(10, Softmax)

### Results
Validation Accuracy: 56.91%
Validation Loss: 1.3495

### Observation
Very poor performance.
Reason:
ResNet50 pretrained on ImageNet expects `preprocess_input()`, not simple normalization.

---

# Experiment 2

### Changes
Replaced

image / 255

with

tf.keras.applications.resnet50.preprocess_input()

Everything else unchanged.

### Results

Best Validation Loss:
0.1241

Validation Accuracy (Restored Best Weights):
95.98%

Highest Validation Accuracy During Training:
96.48%

Training Accuracy:
98.26%

EarlyStopping:
Patience = 5

Best Epoch:
Epoch 4

### Observation

Changing only the preprocessing improved validation accuracy from

56.91%
↓

95.98%

Improvement:
+39.07 percentage points.

This demonstrates the importance of matching preprocessing to the pretrained model.

---

Experiment 3
------------
Date: 8 July 2026

Objective:
Evaluate data augmentation.

Changes:
- RandomFlip(horizontal_and_vertical)
- RandomRotation(0.2)
- RandomZoom(0.2)

Validation Accuracy:
96.28%

Validation Loss:
0.1249

Training Accuracy:
95.89%

Observation:
Slight increase in validation accuracy.
Training became harder, reducing the gap between training and validation accuracy from about 2.2% to about 0.4%, indicating improved generalization.
Training time increased from about 117 s/epoch to about 186 s/epoch due to on-the-fly augmentation.

---

Experiment 4
-------------
Date: 21 July 2026

Objective:
Evaluate the effect of fine-tuning by unfreezing the last 10 layers of ResNet50.

Changes:
- Loaded ImageNet pretrained ResNet50.
- Unfroze the last 10 layers of the backbone.
- Earlier layers remained frozen.
- Reduced learning rate to 1e-5 for stable fine-tuning.
- Data augmentation pipeline retained from Experiment 3.
- EarlyStopping (patience=5, restore_best_weights=True) used.

Validation Accuracy:
97.33%

Validation Loss:
0.0915

Training Accuracy:
97.83%

Observation:
Fine-tuning the last 10 layers improved validation accuracy from 96.28% to 97.33% while reducing validation loss from 0.1249 to 0.0915. Training accuracy increased to approximately 97.8%, indicating that the network successfully adapted higher-level ImageNet features to the EuroSAT dataset. Validation loss reached its minimum around Epoch 11 before increasing slightly in later epochs, suggesting the onset of overfitting. However, EarlyStopping restored the best-performing weights, resulting in improved generalization over the frozen backbone model.

---

Experiment 5
------------
Date: 21 July 2026

Objective:
Evaluate the effect of fine-tuning by unfreezing the last 20 layers of ResNet50 while retaining the data augmentation pipeline and low learning rate.

Changes:

Unfroze the last 20 layers of the ResNet50 backbone.
Remaining layers stayed frozen.
Maintained a learning rate of 1e-5 for stable fine-tuning.
Used EarlyStopping (patience=5, restore_best_weights=True) to preserve the best-performing model.

Validation Accuracy:
97.59%

Validation Loss:
0.0718

Training Accuracy:
97.59% (evaluation accuracy on the validation pipeline after restoring the best weights; peak training accuracy during training reached approximately 98.0%.)

Observation:
Fine-tuning the last 20 layers further improved validation accuracy from 97.33% to 97.59%, while reducing validation loss from 0.0915 to 0.0718, indicating improved model confidence and generalization. Training exhibited temporary fluctuations in validation performance during the early epochs, which is expected when a larger portion of the pretrained backbone is unfrozen. After convergence, the model achieved its best performance around Epoch 9. Although subsequent epochs slightly increased training accuracy, validation loss also increased, suggesting the onset of mild overfitting

---
Experiment 6
------------
Date: 24 JUl 2026

Objective:
Continue fine-tuning from the already trained EuroSAT classifier instead of starting from the frozen model.

Changes:

Loaded the best frozen ResNet50 model trained with data augmentation.
Retained the same data augmentation pipeline from Experiment 3.


Validation Accuracy:
97.59%

Validation Loss:
0.0718

Training Accuracy:
97.59%
