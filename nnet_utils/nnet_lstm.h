
#ifndef NNET_LSTM_H_
#define NNET_LSTM_H_

//#include "nnet_common.h"
#include <cstdlib>
#include "nnet_activation.h"
#include "nnet_dense.h"
#include <math.h>
#include <assert.h>

namespace nnet {

struct lstm_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;
    typedef float mult_t;

    // parameters
    static const unsigned length_x = 4;
    static const unsigned length_h = 4;
    static const unsigned timestep = 4;

    static const unsigned LSTM_DEBUG = 0;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};


template<class data_T, typename CONFIG_T, typename CONFIG_A>
void lstm_tail(
	data_T gate_i[CONFIG_T::length_h],
	data_T gate_f[CONFIG_T::length_h],
	data_T gate_g[CONFIG_T::length_h],
	data_T gate_o[CONFIG_T::length_h],
//	data_T h_pre[CONFIG_T::length_h],
    typename CONFIG_T::accum_t c_pre[CONFIG_T::length_h],
// output
    typename CONFIG_T::accum_t c_cur[CONFIG_T::length_h],
	data_T h_cur[CONFIG_T::length_h]
){

    typename CONFIG_T::accum_t c_tmp1[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_tmp2[CONFIG_T::length_h];
    data_T c_cur_activ[CONFIG_T::length_h];

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor_tail

    #pragma HLS ARRAY_PARTITION variable=c_tmp1 complete
    #pragma HLS ARRAY_PARTITION variable=c_tmp2 complete
    #pragma HLS ARRAY_PARTITION variable=c_cur_activ complete

    int multiplier_limit  = ceil( (3*float(CONFIG_T::length_h)) / float(CONFIG_T::reuse_factor_tail));
    #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

	CELL:
	for(int icell = 0; icell < CONFIG_T::length_h; icell++){
        c_tmp1[icell] = gate_f[icell] * c_pre[icell];
        c_tmp2[icell] = gate_i[icell] * gate_g[icell];
        c_cur[icell]  = c_tmp1[icell] + c_tmp2[icell];
	}
	hard_tanh   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( c_cur, c_cur_activ);  // tanh
	//tanh   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( c_cur, c_cur_activ);  // tanh
	HIDDEN_UNITS:
	for(int itail = 0; itail < CONFIG_T::length_h; itail++){
        h_cur[itail] = gate_o[itail] * c_cur_activ[itail];
	}

};

// only the forward pass, no Timestep loop
template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_A>
void lstm_fw(
    data_T data[CONFIG_T::length_x],
    typename CONFIG_T::weight_t weights_x[CONFIG_T::length_x * CONFIG_T::length_h * 4],
    typename CONFIG_T::weight_t weights_h[CONFIG_T::length_h * CONFIG_T::length_h * 4],
	typename CONFIG_T::bias_t   biases[CONFIG_T::length_h * 4],
    typename CONFIG_T::accum_t c_pre[CONFIG_T::length_h],
    typename CONFIG_T::accum_t c_cur[CONFIG_T::length_h],
    data_T h_pre[CONFIG_T::length_h],
    data_T h_cur[CONFIG_T::length_h] 
//	res_T  res[CONFIG_T::length_h]
){

    typename CONFIG_T::mult_t mult_x[CONFIG_T::length_x * CONFIG_T::length_h * 4 ];
    typename CONFIG_T::mult_t mult_h[CONFIG_T::length_h * CONFIG_T::length_h * 4 ];
    typename CONFIG_T::accum_t acc[CONFIG_T::length_h * 4];
    data_T acc_activ[CONFIG_T::length_h * 4];

    #pragma HLS ARRAY_PARTITION variable=mult_x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=mult_h complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights_x,weights_h,biases

    // Parallel mode
	#pragma HLS PIPELINE
    int multipliers = 4* CONFIG_T::length_h;
	#pragma HLS allocation instances=mul limit=multipliers operation

	#pragma HLS ARRAY_PARTITION variable=weights_h complete dim=0
	#pragma HLS ARRAY_PARTITION variable=weights_x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0
	#pragma HLS ARRAY_PARTITION variable=h_pre complete dim=0
	#pragma HLS ARRAY_PARTITION variable=h_cur complete dim=0
	#pragma HLS ARRAY_PARTITION variable=c_pre complete dim=0
	#pragma HLS ARRAY_PARTITION variable=c_cur complete dim=0


    data_T c_cur_activ[CONFIG_T::length_h];

    typename CONFIG_T::accum_t gate_i[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_f[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_g[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_o[CONFIG_T::length_h];
    data_T gate_i_activ[CONFIG_T::length_h];
    data_T gate_f_activ[CONFIG_T::length_h];
    data_T gate_g_activ[CONFIG_T::length_h];
    data_T gate_o_activ[CONFIG_T::length_h];

    for(int ix = 0; ix < CONFIG_T::length_x; ix++){
    	for(int jx = 0; jx < CONFIG_T::length_h*4; jx++){
    		mult_x[ix*CONFIG_T::length_h*4+jx] = data[ix] * weights_x[ix*CONFIG_T::length_h*4+jx];
    	}
    }
    for(int ih = 0; ih < CONFIG_T::length_h; ih++){
    	for(int jh = 0; jh < CONFIG_T::length_h*4; jh++){
    		mult_h[ih*CONFIG_T::length_h*4+jh] = h_pre[ih] * weights_h[ih*CONFIG_T::length_h*4+jh];
    	}
    }
    for(int iacc = 0; iacc < CONFIG_T::length_h*4; iacc++){
    	acc[iacc] = biases[iacc];
    }
    for(int ix = 0; ix < CONFIG_T::length_x; ix++){
    	for(int jx = 0; jx < CONFIG_T::length_h*4; jx++){
    		acc[jx] += mult_x[ix*CONFIG_T::length_h*4+jx];
    	}
    }
    for(int ih = 0; ih < CONFIG_T::length_h; ih++){
    	for(int jh = 0; jh < CONFIG_T::length_h*4; jh++){
    		acc[jh] += mult_h[ih*CONFIG_T::length_h*4+jh];
    	}
    }

    for(int igate = 0; igate < CONFIG_T::length_h; igate++){
    	gate_i[igate] = acc[igate];
    	gate_f[igate] = acc[1*CONFIG_T::length_h+igate];
    	gate_g[igate] = acc[2*CONFIG_T::length_h+igate];
    	gate_o[igate] = acc[3*CONFIG_T::length_h+igate];
    }

    //template<class data_T, class res_T, typename CONFIG_T>
    sigmoid<typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_i, gate_i_activ);
    sigmoid<typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_f, gate_f_activ);
    tanh   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_g, gate_g_activ);
    sigmoid<typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_o, gate_o_activ);

    for(int icell = 0; icell < CONFIG_T::length_h; icell++){
        c_cur[icell] = gate_f_activ[icell]*c_pre[icell] + gate_i_activ[icell] * gate_g_activ[icell];
    }
    tanh   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( c_cur, c_cur_activ);
    for(int itail = 0; itail < CONFIG_T::length_h; itail++){
        h_cur[itail] = gate_o_activ[itail] * c_cur_activ[itail];
    }

    for(int ii = 0; ii < CONFIG_T::length_h; ii++){
        h_pre[ii] = h_cur[ii];
        c_pre[ii] = c_cur[ii];
    }


}// lstm_fw_00



// LSTM layer with setting the sequence return 
// output: hidden_size x timestep
template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_A, typename CONFIG_X, typename CONFIG_H>
void lstm_seq(
    data_T data[CONFIG_T::length_x*CONFIG_T::timestep],
    typename CONFIG_T::weight_t weights_x[CONFIG_T::length_x * CONFIG_T::length_h * 4],
    typename CONFIG_T::weight_t weights_h[CONFIG_T::length_h * CONFIG_T::length_h * 4],
    typename CONFIG_T::bias_t   biases[CONFIG_T::length_h * 4],
    res_T  res[CONFIG_T::length_h*CONFIG_T::timestep]
){

    //typename CONFIG_T::mult_t mult_x[CONFIG_T::length_x * CONFIG_T::length_h * 4 ];
    //typename CONFIG_T::mult_t mult_h[CONFIG_T::length_h * CONFIG_T::length_h * 4 ];
    typename CONFIG_T::accum_t acc_x[CONFIG_T::length_h * 4];
    typename CONFIG_T::accum_t acc[CONFIG_T::length_h * 4];
    data_T acc_activ[CONFIG_T::length_h * 4];

    // Parallel mode
    //#pragma HLS PIPELINE
    #pragma HLS INLINE


    data_T h_pre[CONFIG_T::length_h];
    data_T h_cur[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_pre[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_cur[CONFIG_T::length_h];
    data_T c_cur_activ[CONFIG_T::length_h];

    typename CONFIG_T::accum_t gate_i[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_f[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_g[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_o[CONFIG_T::length_h];
    data_T gate_i_activ[CONFIG_T::length_h];
    data_T gate_f_activ[CONFIG_T::length_h];
    data_T gate_g_activ[CONFIG_T::length_h];
    data_T gate_o_activ[CONFIG_T::length_h];


    data_T input_x[CONFIG_T::length_x];

    for(int ii = 0; ii < CONFIG_T::length_h; ii++){
        #pragma HLS unroll
        h_pre[ii] = 0;
        c_pre[ii] = 0;
    }

    TIMESTEP:for(int its = 0; its < CONFIG_T::timestep; its++) {
        #pragma HLS PIPELINE rewind


        INPUT_X:
        for(int ix = 0; ix < CONFIG_T::length_x; ix++){
            #pragma HLS UNROLL //factor=1
        //	#pragma HLS PIPELINE
            input_x[ix] = data[ix+its*CONFIG_T::length_x] ;
        }

        dense_simple<data_T, typename CONFIG_T::accum_t, CONFIG_X>(input_x, acc_x, weights_x, biases);
        dense_simple<data_T, typename CONFIG_T::accum_t, CONFIG_H>(h_pre, acc, weights_h, acc_x);

        GATES_SPLIT:
        for(int igate = 0; igate < CONFIG_T::length_h; igate++){
            #pragma HLS UNROLL
            gate_i[igate] = acc[igate];
            gate_f[igate] = acc[1*CONFIG_T::length_h+igate];
            gate_g[igate] = acc[2*CONFIG_T::length_h+igate];
            gate_o[igate] = acc[3*CONFIG_T::length_h+igate];
        }

        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_i, gate_i_activ);
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_f, gate_f_activ);
        hard_tanh <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_g, gate_g_activ); // tanh
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_o, gate_o_activ);

        lstm_tail<data_T, CONFIG_T, CONFIG_A> (gate_i_activ, gate_f_activ, gate_g_activ, gate_o_activ, c_pre, c_cur, h_cur);

        for(int ii = 0; ii < CONFIG_T::length_h; ii++){
            #pragma HLS UNROLL
            h_pre[ii] = h_cur[ii];
            c_pre[ii] = c_cur[ii];

            res[ii+its*CONFIG_T::length_h] = (res_T) h_cur[ii];
        }

    }

}// lstm_06_025


// LSTM layer without setting the sequence return 
// output: only the final hidden units 
template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_A, typename CONFIG_X, typename CONFIG_H>
void lstm(
	//int index,
    data_T data[CONFIG_T::length_x*CONFIG_T::timestep],
    typename CONFIG_T::weight_t weights_x[CONFIG_T::length_x * CONFIG_T::length_h * 4],
    typename CONFIG_T::weight_t weights_h[CONFIG_T::length_h * CONFIG_T::length_h * 4],
	typename CONFIG_T::bias_t   biases[CONFIG_T::length_h * 4],
	res_T  res[CONFIG_T::length_h]
){

    typename CONFIG_T::accum_t acc_x[CONFIG_T::length_h * 4];
    typename CONFIG_T::accum_t acc[CONFIG_T::length_h * 4];
    data_T acc_activ[CONFIG_T::length_h * 4];

    // Parallel mode
	//#pragma HLS PIPELINE
	#pragma HLS INLINE

    data_T h_pre[CONFIG_T::length_h];
    data_T h_cur[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_pre[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_cur[CONFIG_T::length_h];
    data_T c_cur_activ[CONFIG_T::length_h];

    typename CONFIG_T::accum_t gate_i[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_f[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_g[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_o[CONFIG_T::length_h];
    data_T gate_i_activ[CONFIG_T::length_h];
    data_T gate_f_activ[CONFIG_T::length_h];
    data_T gate_g_activ[CONFIG_T::length_h];
    data_T gate_o_activ[CONFIG_T::length_h];


    data_T input_x[CONFIG_T::length_x];

    for(int ii = 0; ii < CONFIG_T::length_h; ii++){
		#pragma HLS unroll
        h_pre[ii] = 0;
        c_pre[ii] = 0;
    }

    LSTM_TS:
	for(int its = 0; its < CONFIG_T::timestep; its++) {
		#pragma HLS PIPELINE rewind

    	INUTT_X:for(int ix = 0; ix < CONFIG_T::length_x; ix++){
            #pragma HLS UNROLL
		//	#pragma HLS PIPELINE
    		input_x[ix] = data[ix+its*CONFIG_T::length_x] ;
    	}

        dense_simple<data_T, typename CONFIG_T::accum_t, CONFIG_X>(input_x, acc_x, weights_x, biases);
        dense_simple<data_T, typename CONFIG_T::accum_t, CONFIG_H>(h_pre, acc, weights_h, acc_x);


        GATES_SPLIT:for(int igate = 0; igate < CONFIG_T::length_h; igate++){
            #pragma HLS UNROLL
    		gate_i[igate] = acc[igate];
    		gate_f[igate] = acc[1*CONFIG_T::length_h+igate];
    		gate_g[igate] = acc[2*CONFIG_T::length_h+igate];
    		gate_o[igate] = acc[3*CONFIG_T::length_h+igate];
    	}

    	//template<class data_T, class res_T, typename CONFIG_T>
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_i, gate_i_activ);
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_f, gate_f_activ);
        hard_tanh <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_g, gate_g_activ); // tanh
        //tanh <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_g, gate_g_activ); // tanh
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_o, gate_o_activ);

        lstm_tail<data_T, CONFIG_T, CONFIG_A> (gate_i_activ, gate_f_activ, gate_g_activ, gate_o_activ, c_pre, c_cur, h_cur);

    	OUTPUT: for(int ii = 0; ii < CONFIG_T::length_h; ii++){
            #pragma HLS UNROLL
            h_pre[ii] = h_cur[ii];
            c_pre[ii] = c_cur[ii];
    	}

    }

    OUTPUT_FINAL: for(int ii = 0; ii < CONFIG_T::length_h; ii++) {
		#pragma HLS unroll
        res[ii] = (res_T) h_cur[ii];
    }

}// lstm_

// LSTM + Timedistrbuted Dense
// improve timing and help vivado hls to synthesis easily when timestep ls large
template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_A, typename CONFIG_X, typename CONFIG_H, typename CONFIG_TD>
void lstm_seq_td(
    data_T data[CONFIG_T::length_x*CONFIG_T::timestep],
    typename CONFIG_T::weight_t weights_x[CONFIG_T::length_x * CONFIG_T::length_h * 4],
    typename CONFIG_T::weight_t weights_h[CONFIG_T::length_h * CONFIG_T::length_h * 4],
    typename CONFIG_T::bias_t   biases[CONFIG_T::length_h * 4],

	typename CONFIG_TD::weight_t weights_td[CONFIG_TD::n_in * CONFIG_TD::n_out],
	typename CONFIG_TD::bias_t   biases_td [CONFIG_TD::n_out],
    res_T  res[CONFIG_TD::n_out*CONFIG_T::timestep]
){

    //typename CONFIG_T::mult_t mult_x[CONFIG_T::length_x * CONFIG_T::length_h * 4 ];
    //typename CONFIG_T::mult_t mult_h[CONFIG_T::length_h * CONFIG_T::length_h * 4 ];
    typename CONFIG_T::accum_t acc_x[CONFIG_T::length_h * 4];
    typename CONFIG_T::accum_t acc[CONFIG_T::length_h * 4];
    data_T acc_activ[CONFIG_T::length_h * 4];

    // Parallel mode
    //#pragma HLS PIPELINE
    #pragma HLS INLINE


    data_T h_pre[CONFIG_T::length_h];
    data_T h_cur[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_pre[CONFIG_T::length_h];
    typename CONFIG_T::accum_t c_cur[CONFIG_T::length_h];
    data_T c_cur_activ[CONFIG_T::length_h];

    typename CONFIG_T::accum_t gate_i[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_f[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_g[CONFIG_T::length_h];
    typename CONFIG_T::accum_t gate_o[CONFIG_T::length_h];
    data_T gate_i_activ[CONFIG_T::length_h];
    data_T gate_f_activ[CONFIG_T::length_h];
    data_T gate_g_activ[CONFIG_T::length_h];
    data_T gate_o_activ[CONFIG_T::length_h];
    res_T tdense_out[CONFIG_TD::n_out];


    data_T input_x[CONFIG_T::length_x];

    for(int ii = 0; ii < CONFIG_T::length_h; ii++){
        #pragma HLS unroll
        h_pre[ii] = 0;
        c_pre[ii] = 0;
    }

    TIMESTEP_TD:for(int its = 0; its < CONFIG_T::timestep; its++) {
        #pragma HLS PIPELINE rewind


        INPUT_X:
        for(int ix = 0; ix < CONFIG_T::length_x; ix++){
            #pragma HLS UNROLL //factor=1
        //	#pragma HLS PIPELINE
            input_x[ix] = data[ix+its*CONFIG_T::length_x] ;
        }

        dense_simple<data_T, typename CONFIG_T::accum_t, CONFIG_X>(input_x, acc_x, weights_x, biases);
        dense_simple<data_T, typename CONFIG_T::accum_t, CONFIG_H>(h_pre, acc, weights_h, acc_x);

        GATES_SPLIT:
        for(int igate = 0; igate < CONFIG_T::length_h; igate++){
            #pragma HLS UNROLL
            gate_i[igate] = acc[igate];
            gate_f[igate] = acc[1*CONFIG_T::length_h+igate];
            gate_g[igate] = acc[2*CONFIG_T::length_h+igate];
            gate_o[igate] = acc[3*CONFIG_T::length_h+igate];
        }

        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_i, gate_i_activ);
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_f, gate_f_activ);
        hard_tanh <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_g, gate_g_activ); // tanh
        sigmoid   <typename CONFIG_T::accum_t, data_T, CONFIG_A> ( gate_o, gate_o_activ);

        lstm_tail<data_T, CONFIG_T, CONFIG_A> (gate_i_activ, gate_f_activ, gate_g_activ, gate_o_activ, c_pre, c_cur, h_cur);

        for(int ii = 0; ii < CONFIG_T::length_h; ii++){
            #pragma HLS UNROLL
            h_pre[ii] = h_cur[ii];
            c_pre[ii] = c_cur[ii];

            //res[ii+its*CONFIG_T::length_h] = (res_T) h_cur[ii];
        }
        nnet::dense_simple<data_T, res_T, CONFIG_TD>(h_cur, tdense_out, weights_td, biases_td);

        OUTPUT_FINAL: for(int ii = 0; ii < CONFIG_TD::n_out; ii++) {
    		#pragma HLS unroll
            res[ii+its*CONFIG_TD::n_out] = (res_T) tdense_out[ii];
        }
        //res[ii+its*CONFIG_T::length_h] = (res_T) h_cur[ii];

    }

}// lstm_seq_td





}//end namespace

#endif
