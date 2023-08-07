from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

#from model import mnist_lstm
import matplotlib.pyplot as plt
import numpy as np
import plotting
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import hls4ml

 
# fix random seed for reproducibility
np.random.seed(9)
batch_size=128
train_epochs = 30


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

# load the model
#model = mnist_lstm(x_train)


model = Sequential()
model.add(LSTM(32, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]), name='lstm1'))
#model.add(Dense(16, activation='relu', name='fc1'))
model.add(Dense(10, activation='softmax', name='output'))

model.summary()

for layer in model.layers:
    if layer.__class__.__name__ in ['Conv2D', 'Dense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize))

    if layer.__class__.__name__ in ['LSTM']:
        w1 = layer.get_weights()[0]
        w2 = layer.get_weights()[1]
        layersize1 = np.prod(w1.shape)
        layersize2 = np.prod(w2.shape)
        print("{}:for x_vector {}".format(layer.name,layersize1)) # 0 = weights, 1 = rnn_weights 2 = biases
        print("{}:for h_vector {}".format(layer.name,layersize2)) # 0 = weights, 1 = rnn_weights 2 = biases
        if (layersize1 > 4096): # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize1))
        if (layersize2 > 4096): # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize2))

p_rate = 0.5
pstep = int (len(x_train)/batch_size) * 3
pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(p_rate, begin_step=pstep, frequency=100)}
model = prune.prune_low_magnitude(model, **pruning_params)

#lstm_es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
#lstm_mc = ModelCheckpoint('lstm_mnist.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

lstm_pruning = pruning_callbacks.UpdatePruningStep()


outdir = f"rnn_hls4ml_p{p_rate}"

# train the model 
train = True
#train = False
if train:
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, 
              batch_size=batch_size,
              epochs=train_epochs, 
              verbose=1,
              validation_split=0.25, 
              #shuffle=True,
              callbacks = [lstm_pruning] )

    model = strip_pruning(model)
    model.save(f'{outdir}/KERAS_check_best_model.h5')
else:
    model = load_model(f'{outdir}/KERAS_check_best_model.h5')

# the model 
#scores = model.evaluate(x_test, y_test, verbose=0)
#print('LSTM test score:', scores[0])
#print('LSTM test accuracy:', scores[1])

y_keras = model.predict(np.ascontiguousarray(x_test))

print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))


w_x = model.layers[0].weights[0].numpy()
w_h = model.layers[1].weights[0].numpy()
#h, b = np.histogram(w, bins=100)
#h, b = np.histogram(w, bins=100)
print('% of zeros for W_x = {}'.format(np.sum(w_x==0)/np.size(w_x)))
print('% of zeros for W_h = {}'.format(np.sum(w_h==0)/np.size(w_h)))


#exit(1)


# ============================== HLS4ML =====================================
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
print("-----------------------------------")
plotting.print_dict(config)
print("-----------------------------------")

#hls_model = hls4ml.converters.convert_from_keras_model(
#         model,
#         hls_config=config,
#         output_dir='rnn_hls4ml/hls4ml_prj',
#         part='xcku115-flvb2104-2-i')
#         #part='xcu250-figd2104-2L-e')

hls_model = hls4ml.converters.convert_from_keras_model(
         model,
         hls_config=config,
         output_dir=f'{outdir}/hls4mli_intel_prj',
         backend = 'Quartus')


hls_model.compile()

y_keras = model.predict(np.ascontiguousarray(x_test))
y_hls = hls_model.predict(np.ascontiguousarray(x_test))

accuracy_keras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1)) 
accuracy_hls4ml= accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))  
print("Keras  Accuracy: {}".format(accuracy_keras))
print("hls4ml Accuracy: {}".format(accuracy_hls4ml))

accs = np.zeros(5)
accs[0] = accuracy_keras
accs[1] = accuracy_hls4ml
accs[2] = p_rate
accs[3] = float(np.sum(w_x==0)/np.size(w_x))
accs[4] = float(np.sum(w_h==0)/np.size(w_h))

np.savetxt("{}/acc_intel.txt".format(outdir), accs, fmt="%.6f")

print("output dir: ", outdir)

hls_model.build()

