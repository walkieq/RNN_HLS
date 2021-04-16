# LSTM-HLS


## lstm-mnist
A MNIST application implemented by HLS-based LSTM

- Model
```python
def mnist_lstm(x):
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    L1 = LSTM(32)(inputs)
    L2 = Dense(10)(L1)
    output = Activation('softmax')(L2)
    model = Model(inputs=inputs, outputs=output)
    return model
```

- How to run

```bat
cd lstm-mnist
vivado_hls -f build_prj.tcl
```

- Check the report

The reports are in the following directory: 
lstm_mnist/myproject_prj/solution1/syn/report

## Introduction of our LSTM unit
