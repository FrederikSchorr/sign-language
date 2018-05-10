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

def train(sModel, seq_length, features_length=2048, saved_model=None, 
          batch_size=32, nb_epoch=100):

    # Get generators.
    data = DataSet("../data/20-features")
    train_generator = data.frame_generator("train", batch_size)
    val_generator = data.frame_generator("test", batch_size)

    # Build a simple LSTM network. We pass the extracted features from
    # our CNN to this model     
    keModel = Sequential()
    keModel.add(LSTM(1024, return_sequences=False,
                    input_shape=(seq_length, features_length),
                    dropout=0.5))
    keModel.add(Dense(256, activation='relu'))
    keModel.add(Dropout(0.5))
    keModel.add(Dense(data.nClasses, activation='softmax'))
    keModel.summary()

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    sLog = time.strftime("%Y%m%d-%H%M") + "-" + sModel + "-" + str(data.nTrain) + "in" + str(data.nClasses)
    csv_logger = CSVLogger(os.path.join("../data/90-logs", sLog + '.log'))

    # Helper: Save the model.
    #checkpointer = ModelCheckpoint(
    #    filepath=os.path.join("../data/30-checkpoints", sLog + "-{epoch:03d}.hdf5"),
    #    verbose=1, save_best_only=True)
 
    # Fit!
    keModel.fit_generator(
        generator=train_generator,
        steps_per_epoch=data.nTrain // batch_size,
        epochs=nb_epoch,
        verbose=1,
        #callbacks=[tb, early_stopper, csv_logger, checkpointer],
        callbacks=[csv_logger],
        validation_data=val_generator,
        validation_steps=40,
        workers=4
    )

def main():
    """These are the main training settings. Set each before running
    this file."""
    
    model = 'lstm'
    seq_length = 20 # frames per video
    features_length = 2048 # output features from CNN
    batch_size = 16
    nb_epoch = 1000

    #os.chdir("/Users/Frederik/Dev/sign-language/02-ledasila")
    train(model, seq_length, features_length=features_length,
          saved_model=None, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
