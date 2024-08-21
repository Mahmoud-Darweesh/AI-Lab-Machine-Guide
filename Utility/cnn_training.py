"""
Training and Testing CNN Models Template Code
Name: Mahmoud Darwish  
ID: b00095043

---

Info:

Training and Evaluating CNN Models

This Python script is designed to train and test a Convolutional Neural Network (CNN) model for image classification tasks. It involves steps like loading image datasets, defining the CNN architecture, training the model, and evaluating its performance. The script supports saving the trained model and relevant metrics for future reference.

---

Installation

To use this script, you need to have the following dependencies installed:

```sh
pip install tensorflow==2.10.0 keras==2.10.0 scikit-learn pandas numpy
```

---

Usage

You can execute the script by simply running it in your Python environment. Make sure you have your image dataset organized and accessible.

```python
# Assuming the script is saved as cnn_training.py
python cnn_training.py
```

---

Model Trainning and Testing

The `train` function handles the training process. It includes loading a CNN model as the base model, defining the custom layers, compiling the model, and fitting it on the training data. It also saves the best model and training history.
The `test_model` function evaluates the trained model on a test dataset and prints the classification report. It also saves the report to a text file.

Data Preparation

The script uses `ImageDataGenerator` for preprocessing images and generating batches of tensor image data with real-time data augmentation.

Training Execution

Finally, the script initiates the training process by calling the `train` function.

```python
train(train_generator, validation_generator, test_generator, "Model Name")
```

For loading a previously trained model and testing it again, you can uncomment and use the following lines:

```python
# model = load_model("Model/" + "Model Name" + ".h5")
# test_model(model, test_generator, "Model Name")
```

---

Examples

1. Train a new model:
   ```python
   python cnn_training.py
   ```

2. Use a pre-trained model for testing:
   ```python
   # Uncomment the loading lines in the script
   python cnn_training.py
   ```

---

Notes

- Ensure your image files are properly labeled and organized in folders.
- Adjust the image size and batch size if needed.
- Customize the `extract_label` function if your labeling method differs. (For example labels saved in a csv file)
- Monitor the training process for overfitting or underfitting and adjust hyperparameters accordingly.

"""

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

image_size = 224


def test_model(model, test_generator, model_name):
    y_true = []
    y_pred = []

    for idx in range(len(test_generator)):
        batch_x, batch_y = test_generator[idx]
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(model.predict(batch_x), axis=1))

    #f1_scores = f1_score(y_true, y_pred, average=None)
    class_report = classification_report(y_true, y_pred)
    #print("F1 scores per class:", f1_scores)
    print("Classification Report:")
    print(class_report)

    # Save F1 scores to a text file
    f1_scores_path = "Info/" + model_name + '_F1_Scores.txt'
    with open(f1_scores_path, 'w') as f:
        #f.write("F1 scores per class:\n")
        #f.write(np.array2string(f1_scores, separator=', '))
        f.write("Classification Report:\n")
        f.write(class_report)

def train(train_generator, val_generator, test_generator, model_name):
    K.clear_session()
    optimizer = SGD(learning_rate=0.01)
    
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(15, activation='softmax')
    ])

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    model_checkpoint_path = "Model/" + model_name + ".h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(train_generator,
                        epochs=50,
                        validation_data=val_generator,
                        workers=16,
                        use_multiprocessing=True, max_queue_size=60,
                        callbacks=[model_checkpoint_callback])
    
    # Save model and history
    model_path = os.path.join("Model", model_name + ".h5")
    model.save(model_path)
    print("Model weights have been saved in the 'Model' folder!")
    
    history_path = "Info/" + model_name + '_History.pickle'
    with open(history_path, 'wb') as history_file:
        pickle.dump(history, history_file)
    
    test_model(model, test_generator, model_name)


# Get your dataset in a variable called image_files it is a list of paths
image_files = []

def get_dataset(main_dir): # Change if your dataset is saved in a diffrent way
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

image_files.sort()
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

# Extract class labels from file paths
def extract_label(file_path): # Write your own label functions if its not the folder name
    return os.path.basename(os.path.dirname(file_path))

# Create a DataFrame for training, validation, and test sets
train_df = pd.DataFrame({'file_path': train_files})
train_df['label'] = train_df['file_path'].apply(extract_label)

val_df = pd.DataFrame({'file_path': val_files})
val_df['label'] = val_df['file_path'].apply(extract_label)

test_df = pd.DataFrame({'file_path': test_files})
test_df['label'] = test_df['file_path'].apply(extract_label)

datagen = ImageDataGenerator(rescale=1./255)
common_args = {
    'x_col': 'file_path',
    'y_col': 'label',
    'target_size': (image_size, image_size),
    'batch_size': 32,
    'class_mode': 'categorical',
    'shuffle': False,
}

train_generator = datagen.flow_from_dataframe(train_df, **common_args)
validation_generator = datagen.flow_from_dataframe(val_df, **common_args)
test_generator = datagen.flow_from_dataframe(test_df, **common_args)

# Count the number of images in each class
train_class_counts = train_df['label'].value_counts()
val_class_counts = val_df['label'].value_counts()
test_class_counts = test_df['label'].value_counts()
total_class_counts = train_class_counts.add(val_class_counts, fill_value=0).add(test_class_counts, fill_value=0)

print("\nTotal Class Counts:")
print(total_class_counts)

train(train_generator, validation_generator, test_generator, "Model Name")
# model = load_model("Model/" + "Model Name" + ".h5")
# test_model(model, test_generator, "Model Name")