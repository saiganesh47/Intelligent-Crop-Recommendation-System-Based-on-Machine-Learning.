# Part 1 - Building the CNN
#importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution Layer
# Corrected to use kernel_size and strides for specifying kernel dimensions
classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second convolution layer
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding 3rd Convolution Layer
classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))  # Dropout for regularization to avoid overfitting
classifier.add(Dense(10, activation='softmax'))  # Output layer for 10 classes

# Compiling The CNN
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Part 2 - Fitting the CNN to the image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Correct import

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Preparing the data generators
training_set = train_datagen.flow_from_directory(
    'Data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'Data/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Training the model using the fit method (replace fit_generator)
model = classifier.fit(
    training_set,
    steps_per_epoch=100,  # Number of batches per epoch
    epochs=20,
    validation_data=test_set,
    validation_steps=6500  # Number of validation batches
)

# Saving the trained model
classifier.save('Trained_Model.h5')

# Print model history keys for validation
print(model.history.keys())

import matplotlib.pyplot as plt

# Summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
