import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, LSTM, Lambda, ConvLSTM2D,  GRU, Reshape, Dropout  # ,CuDNNGRU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
import cv2

lines = []


with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip header
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def adjust_path(p):
    filename = p.split('/')[-1]
    current_path = '../data/IMG/' + filename
    return current_path


images = []
measurements = []


def get_image(line):
    camera_idx = np.random.choice([0, 1, 2])
    steering_adjust = [0., 0.25, -0.25]
    source_path = line[camera_idx]
    image = cv2.imread(adjust_path(source_path))
    measurement = float(line[3]) + steering_adjust[camera_idx]
    flip = np.random.random()
    if (flip > 0.5):
        image = cv2.flip(image, 1)
        measurement *= -1.0
    return image, measurement


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image, center_angle = get_image(batch_sample)
                images.append(center_image)
                angles.append(center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


X_train = np.array(images)
y_train = np.array(measurements)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

model.add(Lambda(lambda x: (x / 255.) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3,  activation='relu'))
model.add(Convolution2D(64, 3, 3,  activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)
# model.fit(X_train, y_train,
#           validation_split=0.2,
#           shuffle=True,
#           nb_epoch=10)

model.save('model.h5')
