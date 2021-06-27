/* 
 * 
 */

#include "math.h"
#include "lstm.h"

#include "HLS_MNIST_2LAYER/lstm1_wx.h"
#include "HLS_MNIST_2LAYER/lstm1_wh.h"
#include "HLS_MNIST_2LAYER/lstm1_wb.h"

#include "HLS_MNIST_2LAYER/lstm2_wx.h"
#include "HLS_MNIST_2LAYER/lstm2_wh.h"
#include "HLS_MNIST_2LAYER/lstm2_wb.h"

#include "HLS_MNIST_2LAYER/dense1_w.h"
#include "HLS_MNIST_2LAYER/dense1_b.h"



#pragma hls_design top
void lstm(
		input_t lstm_in[N_TS*N1_LX],
		result_t lstm_out[MODEL_OUT]
){

	#pragma HLS ARRAY_RESHAPE variable=lstm_out complete dim=0

	result_t lstm1_out [N1_LH*N_TS];
	result_t lstm2_out [N2_LH];

	#pragma HLS DATAFLOW

	const int input_factor = N1_LX;
	const int layer1_lh = N1_LH;
    #pragma HLS ARRAY_PARTITION variable=lstm_in cyclic factor=input_factor
	#pragma HLS ARRAY_PARTITION variable=lstm1_out cyclic factor=layer1_lh

    nnet::lstm_seq<input_t, result_t, config1, config_a, config_x, config_h>(lstm_in, lstm1_wx, lstm1_wh, lstm1_wb, lstm1_out);

#ifndef __SYNTHESIS__
    if (DEBUG==1) {
    	std::cout <<"\n LSTM layer 1 output ";
    	for(int ff = 0; ff < N1_LH*N_TS; ff++) {
    		std::cout <<", "<< lstm1_out[ff];
    	}
    }
#endif

    nnet::lstm<input_t, result_t, config1_lstm2, config_a_lstm2, config_x_lstm2, config_h_lstm2>(lstm1_out, lstm2_wx, lstm2_wh, lstm2_wb, lstm2_out);

#ifndef __SYNTHESIS__
    if (DEBUG==1) {
    	std::cout <<"\n LSTM layer 2 output ";
    	for(int ff = 0; ff < N2_LH; ff++) {
    		std::cout <<", "<< lstm2_out[ff];
    	}
    }
#endif

    result_t layer3_out[DENSE1_OUT];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense_simple<input_t, result_t, config2>(lstm2_out, layer3_out, dense1_w, dense1_b);

#ifndef __SYNTHESIS__
    if (DEBUG==1) {
    	std::cout <<"\n Dense output ";
    	for(int ff = 0; ff < DENSE1_OUT; ff++) {
    		std::cout <<", "<< layer3_out[ff];
    	}
    }
#endif

    nnet::softmax<input_t, result_t, softmax_config>(layer3_out, lstm_out);


#ifndef __SYNTHESIS__
    if (DEBUG==1) {
    	std::cout <<"\n Softmax output ";
    	for(int ff = 0; ff < DENSE1_OUT; ff++) {
    		std::cout <<", "<< lstm_out[ff];
    	}
    }
#endif



}

