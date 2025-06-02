# dl_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical


def prepare_labels(y):
    """Convert class vectors to binary class matrices."""
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y - 1, num_classes=num_classes)  # Assuming labels start from 1
    return y_cat


def train_mlp(X_train, y_train, X_test, y_test):
    y_train_cat = prepare_labels(y_train)
    y_test_cat = prepare_labels(y_test)

    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=50, batch_size=32, verbose=0)
    return model


def train_lstm(X_train, y_train, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    y_train_cat = prepare_labels(y_train)
    y_test_cat = prepare_labels(y_test)

    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=50, batch_size=32, verbose=0)
    return model


def train_cnn(X_train, y_train, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    y_train_cat = prepare_labels(y_train)
    y_test_cat = prepare_labels(y_test)

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=50, batch_size=32, verbose=0)
    return model