import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ANN:
    
    class DataPreparation:
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def data_preparation(self, target):
            X = self.dataframe.drop(target, axis=1)
            y = self.dataframe[target]
            return X, y
        
        def train_test_split(self, X, y, train_ratio, validation_ratio, test_ratio):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, test_size=test_ratio)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=validation_ratio)
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            return X_train, X_val, X_test
        
        def stanford_scale(self):
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            X_train_scaled = scaler.transform(self.X_train)
            X_val_scaled = scaler.transform(self.X_val)
            X_test_scaled = scaler.transform(self.X_test)
            self.X_train_scaled = X_train_scaled
            self.X_val_scaled = X_val_scaled
            self.X_test_scaled = X_test_scaled
        
        def model_construction(self, model,n_batch_size, n_epochs):

            n_steps_per_epoch = int(self.X_train.shape[0] / n_batch_size)
            n_validation_steps = int(self.X_val.shape[0] / n_batch_size)
            n_test_steps = int(self.X_test.shape[0] / n_batch_size)

            print('Number of Epochs: ' + str(n_epochs))

            model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

            history = model.fit(self.X_train_scaled,
                    self.y_train,
                    steps_per_epoch=n_steps_per_epoch,
                    epochs=n_epochs,
                    batch_size=n_batch_size,
                    validation_data=(self.X_val_scaled, self.y_val),
                    validation_steps=n_validation_steps)
            
            self.model = model
            self.history = history
        
        def visualization(self):
            mae = self.history.history['mae']
            val_mae = self.history.history['val_mae']
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            epochs = range(1, len(mae) + 1)

            plt.plot(epochs, mae, 'bo', label='Training MAE')
            plt.plot(epochs, val_mae, 'b', label='Validation MAE')
            plt.title('Training and validation MAE')
            plt.legend()

            plt.figure()

            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

            plt.show()
        
        def save_model(self):
            self.model.save('model.h5')
            self.model.save_weights('model_weights.h5')
    
