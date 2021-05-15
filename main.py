import pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


# Folosim doar ClassId si Path din csv, scoatem celelalte coloane
train_df = pandas.read_csv('E:/~PI/Traffic_Sign_Detection/german/Train.csv')
filenames = ['E:/~PI/Traffic_Sign_Detection/german/' + fname for fname in train_df['Path']]
labels = train_df['ClassId'].to_numpy()


X_train_images, X_test_images, y_train_labels, y_test_labels = train_test_split(filenames,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=45)
print("Number of training images: ", len(X_train_images))
print("Number of training labels: ", len(y_train_labels))
print("Number of testing images: ", len(X_test_images))
print("Number of testing labels: ", len(y_test_labels))


X_train = []
for i in range(len(X_train_images)):
    aux = cv.imread(X_train_images[i], cv.IMREAD_COLOR)
    aux = cv.resize(aux, (32, 32))
    print(i)
    X_train.append(aux)

X_test = []
for i in range(len(X_test_images)):
    aux = cv.imread(X_test_images[i], cv.IMREAD_COLOR)
    aux = cv.resize(aux, (32, 32))
    print(i)
    X_test.append(aux)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
print("X_train.shape : " + str(X_train.shape))
print("X_test.shape : " + str(X_test.shape))
plt.imshow(X_train[0])
plt.show()
print("y_train shape: " + str(y_train_labels.shape))
print("a value from y_train: " + str(y_train_labels[1]))


IMG_SIZE = 32
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print("X_train.shape: " + str(X_train.shape))
print("X_test.shape: " + str(X_test.shape))

print("\nX_train[0] inainte de normalizare: " + str(X_train[0]) + "\n")
X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)
print("\nX_train[0] dupa normalizare: " + str(X_train[0]) + "\n")

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(43))
model.add(Activation('softmax'))


print(model.summary())
# tf.keras.utils.plot_model(model, to_file='file.png', show_shapes=True)


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model_saved = model.fit(X_train, y_train_labels, epochs=5, batch_size=64)
loss, accuracy = model.evaluate(X_test, y_test_labels)
print("Loss: " + str(loss))
print("Accuracy: " + str(accuracy))

model.save('mymodel', model_saved)
print("\n Model Saved !")


