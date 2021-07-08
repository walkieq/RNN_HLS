#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "../nnet_utils/nnet_lstm.h"
#include "../nnet_utils/nnet_activation.h"
#include "../nnet_utils/nnet_dense.h"

#define DEBUG 1


//*** precision
// hls4ml default precision is used
#define ACT_TTL_BIT 16 // total bits
#define ACT_INT_BIT 6  // integer bits

#define ACC_TTL_BIT ACT_TTL_BIT // total bits for accumulator 
#define ACC_INT_BIT ACT_INT_BIT 


//*** model parameters
// Timestep (sequence length)
#define N_TS 28

// LSTM layer
#define N1_LX 28  // length of input x
#define N1_LH 16  // hidden units

#define N2_LX 16  // length of input x
#define N2_LH 16  // hidden units

#define DENSE1_IN  N2_LH
#define DENSE1_OUT 10

#define MODEL_OUT DENSE1_OUT

#define delta 8

//*** reuse factor
#define R1_H    1
#define R2_H    1

#define R1_X     (R1_H + delta)
#define R2_X     (R1_H + delta)

#define R_DENSE1 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<ACT_TTL_BIT, ACT_INT_BIT, AP_RND_CONV, AP_SAT> act_default_t;
typedef ap_fixed<ACT_TTL_BIT, ACT_INT_BIT> model_default_t;
typedef model_default_t input_t;
typedef model_default_t result_t;

typedef ap_fixed<ACC_TTL_BIT, ACC_INT_BIT> accum_lstm_t;
typedef accum_lstm_t model_biases_t;
typedef accum_lstm_t mult_p_t;


//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::lstm_config {
    static const unsigned length_x = N1_LX;
    static const unsigned length_h = N1_LH;
    static const unsigned timestep = N_TS;

    //static const unsigned reuse_factor_tail = R1_TAIL;
    static const unsigned LSTM_DEBUG = DEBUG;

    typedef accum_lstm_t    bias_t;
    typedef model_default_t weight_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     	mult_t;
};

struct config_a : nnet::activ_config {
    static const unsigned n_in = N1_LH;
    typedef model_default_t table_t;
    typedef act_default_t constant_t;

};


struct config_x : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef accum_lstm_t    bias_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = R1_X;
    static const unsigned n_in = N1_LX;
    static const unsigned n_out = N1_LH*4;

};
struct config_h : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef accum_lstm_t    bias_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = R1_H;
    static const unsigned n_in = N1_LH;
    static const unsigned n_out = N1_LH*4;

};

struct config1_lstm2 : nnet::lstm_config {
    static const unsigned length_x = N2_LX;
    static const unsigned length_h = N2_LH;
    static const unsigned timestep = N_TS;

    //static const unsigned reuse_factor = 1;
    //static const unsigned reuse_factor_tail = R2_TAIL;
    static const unsigned LSTM_DEBUG = DEBUG;
    //static const bool store_weights_in_bram = true;

    typedef accum_lstm_t    bias_t;
    typedef model_default_t weight_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     mult_t;
};

struct config_a_lstm2 : nnet::activ_config {
    static const unsigned n_in = N2_LH;
    typedef model_default_t table_t;
    typedef act_default_t constant_t;

};


struct config_x_lstm2 : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef accum_lstm_t    bias_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     mult_t;

    // Layer Sizes
    static const unsigned reuse_factor =R2_X;
    static const unsigned n_in = N2_LX;
    static const unsigned n_out = N2_LH*4;

};
struct config_h_lstm2 : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef accum_lstm_t    bias_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = R2_H;
    static const unsigned n_in = N2_LH;
    static const unsigned n_out = N2_LH*4;
};



struct config2 : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef accum_lstm_t    bias_t;
    typedef accum_lstm_t    accum_t;
    typedef mult_p_t     mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = R_DENSE1;
    static const unsigned n_in = DENSE1_IN;
    static const unsigned n_out = DENSE1_OUT;

};


struct softmax_config: nnet::activ_config {
    static const unsigned n_in = DENSE1_OUT;
    typedef ap_fixed<18,8> table_t;
    //typedef model_default_t table_t;
};

#endif
