/* 
 * 
 */

#include "math.h"
#include "lstm.h"
#include "HLS_AE_SMALL/lstm1_wx.h"
#include "HLS_AE_SMALL/lstm1_wh.h"
#include "HLS_AE_SMALL/lstm1_wb.h"

#include "HLS_AE_SMALL/lstm2_wx.h"
#include "HLS_AE_SMALL/lstm2_wh.h"
#include "HLS_AE_SMALL/lstm2_wb.h"

#include "HLS_AE_SMALL/dense1_w.h"
#include "HLS_AE_SMALL/dense1_b.h"



#pragma hls_design top
void lstm(
		input_t lstm_in[N_TS*N1_LX],
		result_t lstm_out[MODEL_OUT]
){

	#pragma HLS ARRAY_RESHAPE variable=lstm_out complete dim=0

	result_t lstm1_out [N1_LH];
    result_t repeat_out [N1_LH*N_TS];


	#pragma HLS DATAFLOW
	//#pragma HLS INLINE

	const int input_factor = N1_LX;
	const int layer1_lh = N1_LH;

    //#pragma HLS ARRAY_PARTITION variable=lstm_in cyclic factor=input_factor
    #pragma HLS ARRAY_PARTITION variable=lstm1_out complete
	#pragma HLS ARRAY_PARTITION variable=repeat_out cyclic factor=layer1_lh

    // LSTM with seq=false
	nnet::lstm<input_t, result_t, config1, config2, config_x, config_h>(lstm_in, lstm1_wx, lstm1_wh, lstm1_wb, lstm1_out);

#ifndef __SYNTHESIS__
    if (DEBUG==1) {
    	std::cout <<"\n LSTM layer 1 output ";
    	for(int ff = 0; ff < N1_LH*N_TS; ff++) {
    		std::cout <<", "<< lstm1_out[ff];
    	}
    }
#endif

    REPEAT:
	for(int ii = 0; ii < N_TS; ii++){
		#pragma HLS unroll
		for(int jj = 0; jj < N1_LH; jj++){
			#pragma HLS unroll
			repeat_out[ii*N1_LH+jj] = lstm1_out[jj];
		}
	}

    // LSTM + TimeDistributed Dense
    nnet::lstm_seq_td<input_t, result_t, config1_lstm2, config2_lstm2, config_x_lstm2, config_h_lstm2, config3>(repeat_out, lstm2_wx, lstm2_wh, lstm2_wb, dense1_w, dense1_b, lstm_out);

#ifndef __SYNTHESIS__
    if (DEBUG==1) {
    	std::cout <<"\n LSTM layer 2 output ";
    	for(int ff = 0; ff < N_TS; ff++) {
    		std::cout <<", "<< lstm_out[ff];
    	}
    }
#endif


}

