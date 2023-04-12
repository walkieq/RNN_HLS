Some simple examples of LSTMs using hls4ml

- Environment: 
```python
conda env create -f environment.yml
conda activate hls4ml-rnn
```

- How to run
```python
python t1_lstm_mnist_hls4ml.py
```
The t1 example is for a single-layer lstm on mnist dataset while the t2 example is for a 2-layer lstm on the same dataset. 

Currently the default lstm implementation in hls4ml is used. I will update it later. 