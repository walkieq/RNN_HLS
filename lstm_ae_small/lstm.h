/* lstm.h
 */

#ifndef _CONV_H_
#define _CONV_H_

#include <ap_fixed.h>

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"

void lstm(
		input_t lstm_in[N_TS*N1_LX],
		//model_default_t weights_x[N_LX*N_LH*4],
		//model_default_t weights_h[N_LH*N_LH*4],
		//model_default_t bias[N_LH*4],
		result_t conv_out[MODEL_OUT]
					);
#endif /* _LSTM_H_ */

