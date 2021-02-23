import numpy as np
import cv2
import os

from Generator import Generator
from Model import *

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras import regularizers

from tensorflow.keras.applications import *

import json

from glob import glob

HEIGHT, WIDTH, CHANNELS = 256, 256, 3
BATCH_SIZE = 96
EPOCHS = 10
REPEAT = 2
DROPOUT = 0.2
N_CLASSES = 2

with open('../TS_Data/labels.json') as json_file:
    labels = json.load(json_file)

train_paths = glob("../TS_Data/train/*.jpg")
dev_paths = glob("../TS_Data/dev/*.jpg")
test_paths = glob("../TS_Data/test/*.jpg")

n_train = len(train_paths)
n_dev = len(dev_paths)
n_test = len(test_paths)

print('Train: ', n_train)
print('Dev: ', n_dev)
print('Test: ', n_test)

print(n_train + n_dev + n_test == len(labels))
print("Total: ", len(labels))

train_gen = Generator(train_paths, BATCH_SIZE, labels, N_CLASSES)
dev_gen = Generator(dev_paths, BATCH_SIZE, labels, N_CLASSES)

print(train_gen)
print(dev_gen)

opt = SGD(lr=1e-4, momentum=0.9)

model = create_model()
model.compile(optimizer=opt, loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_gen,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=dev_gen,
    )
