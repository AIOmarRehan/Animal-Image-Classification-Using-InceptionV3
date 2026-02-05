[If you would like a detailed explanation of this project, please refer to the Medium article below.](https://medium.com/@ai.omar.rehan/building-a-clean-reliable-and-accurate-animal-classifier-using-inceptionv3-175f30fbe6f3)

---
# Animal Image Classification Using InceptionV3

*A complete end-to-end pipeline for building a clean, reliable deep-learning classifier.*

This project implements a **full deep-learning workflow** for classifying animal images using **TensorFlow + InceptionV3**, with a major focus on **dataset validation and cleaning**. Before training the model, I built a comprehensive system to detect corrupted images, duplicates, brightness/contrast issues, mislabeled samples, and resolution outliers.

This repository contains the full pipeline—from dataset extraction to evaluation and model saving.

---

## Features

### Full Dataset Validation

The project includes automated checks for:

* Corrupted or unreadable images
* Hash-based duplicate detection
* Duplicate filenames
* Misplaced or incorrectly labeled images
* File naming inconsistencies
* Extremely dark/bright images
* Very low-contrast (blank) images
* Outlier resolutions

### Preprocessing & Augmentation

* Resize to 256×256
* Normalization
* Light augmentation (probabilistic)
* Efficient `tf.data` pipeline with caching, shuffling, prefetching

### Transfer Learning with InceptionV3

* Pretrained ImageNet weights
* Frozen base model
* Custom classification head (GAP → Dense → Dropout → Softmax)
* EarlyStopping + ModelCheckpoint + ReduceLROnPlateau callbacks

### Clean & Reproducible Training

* 80% training
* 10% validation
* 10% test
* High stability due to dataset cleaning

---

## 1. Dataset Extraction

The dataset is stored as a ZIP file (Google Drive). After mounting the drive, it is extracted and indexed into a Pandas DataFrame:

```python
drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/Animals.zip'
extract_to = '/content/my_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
```

Each image entry records:

* Class
* Filename
* Full path

---

## 2. Dataset Exploration

Before any training, I analyzed:

* Class distribution
* Image dimensions
* Grayscale vs RGB
* Unique sizes
* Folder structures

Example class-count visualization:

```python
plt.figure(figsize=(32, 16))
class_count.plot(kind='bar')
```

This revealed imbalance and inconsistent image sizes early.

---

## 3. Visual Sanity Checks

Random images were displayed with their brightness, contrast, and shape to manually confirm dataset quality.

This step prevents hidden issues—especially in community-created or scraped datasets.

---

## 4. Data Quality Detection

The system checks for:

### Duplicate Images (Using MD5 Hashing)

```python
def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

df['file_hash'] = df['full_path'].apply(get_hash)
duplicate_hashes = df[df.duplicated('file_hash', keep=False)]
```

### Corrupted Files

```python
try:
    with Image.open(file_path) as img:
        img.verify()
except:
    corrupted_files.append(file_path)
```

### Brightness/Contrast Outliers

Using PIL’s `ImageStat` to detect very dark/bright samples.

### Label Consistency Check

```python
folder = os.path.basename(os.path.dirname(row["full_path"]))
```

This catches mislabeled entries where folder name ≠ actual class.

---

## 5. Preprocessing Pipeline

Custom preprocessing:

* Resize → Normalize
* Optional augmentation
* Efficient `tf.data` batching

```python
def preprocess_image(path, target_size=(256, 256), augment=True):
    img = tf.image.decode_image(...)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
```

Split structure:

| Split      | Percent |
| ---------- | ------- |
| Train      | 80%     |
| Validation | 10%     |
| Test       | 10%     |

---

## 6. Model — Transfer Learning with InceptionV3

```python
from tensorflow.keras.applications import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False
```

Classification head:

* GlobalAveragePooling2D
* Dense(512, ReLU)
* Dropout(0.5)
* Softmax

Compile:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 7. Training

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[...]
)
```

Using:

* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau

The model converged quickly thanks to the cleaned dataset.

---

## 8. Evaluation & Saving

```python
model.evaluate(test_ds)
model.save("Simple_CNN_Classification.h5")
```

---

## Key Takeaways

The biggest lesson from this project:

> **A strong deep-learning model starts with a clean dataset.**

Cleaning the data took more time than training the model—but it directly improved accuracy, stability, and model reliability.

If you're building your own image classification project, always verify:

* Dataset quality
* Brightness/contrast issues
* Duplicate removal
* Class consistency
* Resolution outliers

Clean data makes everything else easier.
