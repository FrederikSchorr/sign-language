"""
Train our RNN on extracted features.

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from data import DataSet

import time
import os.path

def train(model, seq_length, features_length=2048, saved_model=None, class_limit=None, 
          batch_size=32, nb_epoch=100):
    
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join("../data/30-checkpoints", model + '-' + \
            ".{epoch:03d}.hdf5"),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join("../data/90-logs", model + '-' + 'training-' + str(timestamp) + '.log'))

    # Get generators.
    data = DataSet("../data/20-features")
    train_generator = data.frame_generator("train", batch_size)
    val_generator = data.frame_generator("test", batch_size)

    # Build a simple LSTM network. We pass the extracted features from
    # our CNN to this model     
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,
                    input_shape=(seq_length, features_length),
                    dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(data.nClasses, activation='softmax'))
    model.summary()

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit!
    model.fit_generator(
        generator=train_generator,
        #steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        #callbacks=[tb, early_stopper, csv_logger, checkpointer],
        #validation_data=val_generator,
        #validation_steps=40,
        #workers=4
    )

def main():
    """These are the main training settings. Set each before running
    this file."""
    
    model = 'lstm'
    seq_length = 20 # frames per video
    features_length = 2048 # output features from CNN
    class_limit = None  # int, can be 1-101 or None
    batch_size = 32
    nb_epoch = 5

    train(model, seq_length, features_length=features_length,
          saved_model=None, class_limit=class_limit, 
          batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
