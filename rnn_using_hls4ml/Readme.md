Some simple examples of LSTMs using hls4ml

- Python Environment: 
```python
conda env create -f environment.yml
conda activate hls4ml-rnn
```

- Vivado HLS Environment: 
You can simply run the following to add the Xilinx FPGA tool to the path:
```python
source /opt/Xilinx/Vivado/2019.2/settings64.sh
```
replace the vivado location with the one on your machine. 


- How to run
```python
python t1_lstm_mnist_hls4ml.py
```
The t1 example is for a single-layer lstm on mnist dataset while the t2 example is for a 2-layer lstm on the same dataset. 


- Model in t1 example
```python
def mnist_lstm(x):
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    L1 = LSTM(32)(inputs)
    L2 = Dense(10)(L1)
    output = Activation('softmax')(L2)
    model = Model(inputs=inputs, outputs=output)
    return model
```

- Model in t2 example
```python
def mnist_lstm(x):
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    L1 = LSTM(16, return_sequences=True)(inputs)
    L2 = LSTM(16)(inputs)
    L3 = Dense(10)(L2)
    output = Activation('softmax')(L3)
    model = Model(inputs=inputs, outputs=output)
    return model
```

Currently the default lstm implementation in hls4ml is used. I will update the template later. 
