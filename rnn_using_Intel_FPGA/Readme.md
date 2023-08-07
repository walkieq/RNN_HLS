Some simple examples of LSTMs using hls4ml and Intel FPGAs. 
I am using Intel HLS version 21.4. The default FPGA chip in hls4ml is A10 1150. 

- Python Environment: 
```python
conda env create -f environment_intel.yml
conda activate hls4ml-rnn-intel
```

- Intel HLS Environment: 
You can simply run the following to add the Xilinx FPGA tool to the path:
```python
source /opt/Intel/intelFPGA_pro/21.4/hls/init_hls.sh

```
replace the intel hls tool location with the one on your machine. 


- How to run
```python
python t1_lstm_mnist_hls4ml_intel.py
```
The t1 example is for a single-layer lstm on mnist dataset. 

- Model in t1 example
The Model
```python
def mnist_lstm(x):
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    L1 = LSTM(32)(inputs)
    L2 = Dense(10)(L1)
    output = Activation('softmax')(L2)
    model = Model(inputs=inputs, outputs=output)
    return model
```

- Model in t3 example
It is the same as the model in t1 example, but a unified pruning rate of 50% is applied. 

- Model in t0
It is a simple MLP model which targets Intel FPGA. 


- TODO
Currently the default lstm implementation in hls4ml is used. The latency and initiation interval are large. I will update the LSTM template later. 
