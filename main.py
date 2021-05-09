import pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


# Folosim doar ClassId si Path din csv, scoatem celelalte coloane
train_df = pandas.read_csv('E:/~PI/Traffic_Sign_Detection/german/Train.csv')

filenames = ['E:/~PI/Traffic_Sign_Detection/german/' + fname for fname in train_df['Path']]
labels = train_df['ClassId'].to_numpy()
unique_signs = np.unique(labels)


# One-hot encoding
labels = tf.keras.utils.to_categorical(labels, 43)


X = filenames
y = labels

# Split
X_train_images, X_val_images, y_train_labels, y_val_labels = train_test_split(X, y, test_size=0.2, random_state=42)
print("Number of training images: ", len(X_train_images))
print("Number of training labels: ", len(y_train_labels))
print("Number of testing images: ", len(X_val_images))
print("Number of testing labels: ", len(y_val_labels))

IMG_SIZE = 32


def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


# Turning data into batches
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label


BATCH_SIZE = 64


def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    if test_data:
        print("\nCreating test data batches...\n")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
    elif valid_data:
        print("\nCreating validation dataset batches...\n")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    else:
        print("\nCreating training dataset batches...\n")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch


# Batchuri pt testare si validare
train_data = create_data_batches(X_train_images, y_train_labels)
val_data = create_data_batches(X_val_images, y_val_labels, valid_data=True)


print(train_data.element_spec)
print(val_data.element_spec)


INPUT_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = len(unique_signs)


def create_conv_nn(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    return model


def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE):
    model = create_conv_nn(input_shape=input_shape)
    print("Compiling the model")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model


CNN = create_model()
CNN.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

# TRAINING
NUM_EPOCHS = 10
model_saved = CNN.fit(x=train_data,
                      validation_data=val_data,
                      validation_freq=1,
                      epochs=NUM_EPOCHS,
                      callbacks=early_stopping_callback
                      )

CNN.save('mymodel', model_saved)
print("\nModel saved!\n")


# Cream batchurile pt test
test_df = pandas.read_csv('E:/~PI/Traffic_Sign_Detection/german/Test.csv')
test_img_paths = ['E:/~PI/Traffic_Sign_Detection/german/' + path for path in test_df['Path']]


X_test = create_data_batches(test_img_paths, test_data=True)
y_test = list(test_df['ClassId'])


predictions = CNN.predict(X_test, verbose=1)


# Convertim probabilitatile in labeluri
def get_prediction_label(prediction_probabilities):
    return unique_signs[np.argmax(prediction_probabilities)]


prediction_labels = []
for i in predictions:
    prediction_labels.append(get_prediction_label(i))
print(prediction_labels[:112])


acc = accuracy_score(y_test, prediction_labels)
print("\nAccuracy score: " + str(acc) + "\n")

