from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#from model import mnist_lstm
import matplotlib.pyplot as plt
import numpy as np
import plotting
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import hls4ml

 
# fix random seed for reproducibility
np.random.seed(9)

# input dimension 
n_input = 28
# timesteps
n_step = 28
# output dimension
n_classes = 10

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# prepare the dataset for training and testing
# reshape input to be  [samples, time steps, features]
x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
 
y_train = to_categorical(y_train, n_classes)
y_test  = to_categorical(y_test,  n_classes)


model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]), name='lstm1'))
model.add(LSTM(16, name='lstm2'))
#model.add(Dense(16, activation='relu', name='fc1'))
model.add(Dense(10, activation='softmax', name='output'))

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train the model 
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          #callbacks=[lstm_es, lstm_mc],
          validation_split=0.2)

# evaluate the model 
scores = model.evaluate(x_test, y_test, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])


# ============================== HLS4ML =====================================
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
print("-----------------------------------")
plotting.print_dict(config)
print("-----------------------------------")

hls_model = hls4ml.converters.convert_from_keras_model(
         model,
         hls_config=config,
         output_dir='rnn_hls4ml/hls4ml_prj2',
         part='xcku115-flvb2104-2-i')
         #part='xcu250-figd2104-2L-e')

hls_model.compile()

y_keras = model.predict(np.ascontiguousarray(x_test))
y_hls = hls_model.predict(np.ascontiguousarray(x_test))

print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

hls_model.build(csim=False)

hls4ml.report.read_vivado_report('rnn_hls4ml/hls4ml_prj2')
