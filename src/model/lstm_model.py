import numpy as np
import matplotlib.pyplot as plt

def run_lstm():
    
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    

    # these are just made up hyperparameters, change them as you wish
    hidden_size = 4

    lstm_model = tf.keras.models.Sequential([

        tf.keras.layers.LSTM(
            hidden_size, input_shape=(horizon, num_time_series)),


        tf.keras.layers.RepeatVector(forecast),

        tf.keras.layers.LSTM(hidden_size, return_sequences=True),

        #tf.keras.layers.Dense(hidden_size, activation=elu),

        tf.keras.layers.TimeDistributed(Dense(num_time_series, activation=elu))

    ])

    # lstm_model = tf.keras.models.Sequential([
    #     tf.keras.Input(shape=(horizon, num_time_series)), # shape not including the batch size
    #     # Shape [batch, time, features] => [batch, time, lstm_units]
    #     tf.keras.layers.LSTM(370, return_sequences=True),
    #     # Shape => [batch, time, features]
    #     tf.keras.layers.Dense(units=forecast)
    # ])

    lstm_model.summary()

    lstm_model.compile(optimizer='adam',
                       loss='mae',
                       metrics=['mae'])

    lstm_model.fit(X_train, Y_train, batch_size=hyperparams['batch_size'], epochs=hyperparams['epochs'],
                   steps_per_epoch=hyperparams['steps_per_epoch'], validation_data=(X_valid, Y_valid))
    test_pred = lstm_model.predict(X_test)
    plt.plot(Y_test[0, :, 6])
    plt.plot(test_pred[0, :, 6])
    plt.show()
    exit()
