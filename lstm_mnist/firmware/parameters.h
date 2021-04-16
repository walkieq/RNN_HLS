#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_lstm.h"
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"

#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/wr2.h"
#include "weights/w3.h"
#include "weights/b3.h"

struct config1_lstm2 : nnet::lstm_config {
    static const unsigned length_x = N_INPUT_2_1;
    static const unsigned length_h = N_LAYER_2;
    static const unsigned timestep = N_INPUT_1_1;
   
    static const unsigned reuse_factor_tail = 1;

    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef model_default_t accum_t;
    typedef model_default_t mult_t;
};

struct config2_lstm2 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    typedef ap_fixed<18,8> table_t;
    typedef model_default_t constant_t;
};

struct config_x_lstm2 : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef model_default_t bias_t;
    typedef model_default_t accum_t;
    typedef model_default_t mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = 9;
    static const unsigned n_in = N_INPUT_2_1;
    static const unsigned n_out = N_LAYER_2 * 4;

};

struct config_h_lstm2 : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef model_default_t bias_t;
    typedef model_default_t accum_t;
    typedef model_default_t mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = 1;
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_2 * 4;
};

struct config3 : nnet::dense_config {
    // Internal data type definitions
    typedef model_default_t weight_t;
    typedef model_default_t bias_t;
    typedef model_default_t accum_t;
    typedef model_default_t mult_t;

    // Layer Sizes
    static const unsigned reuse_factor = 1;
    static const unsigned n_in = N_SEQUENCE_OUT_2*N_LAYER_2;
    static const unsigned n_out = N_LAYER_3;

};

struct softmax_config5: nnet::activ_config {
    static const unsigned n_in = N_LAYER_3;
    typedef ap_fixed<18,8> table_t;
    typedef model_default_t constant_t;
};

#endif
