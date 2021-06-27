// testbench.cpp
 
#include <iomanip>
#include <fstream>
#include <iostream>
#include "math.h"
#include "lstm.h"
#include "HLS_MNIST_2LAYER/input.h"


int main() {
    std::cout << "# Starting Testbench \n";

    const int BATCH=1;
    input_t lstm_in[N_TS*N1_LX];
    result_t lstm_out[MODEL_OUT];

    std::cout << "# reuse factor of mvm_h  is " << R1_H << "\n";
    std::cout << "# reuse factor of mvm_x  is " << R1_X  << "\n";
    std::cout << "# reuse factor of dense is " << R_DENSE1  << "\n";

    for (int k = 0; k < BATCH; k++) {
    	std::cout <<"\n input ";
    	for (int i = 0; i < N_TS*N1_LX; i++) {
    		//conv_out[i] = 0;
    		lstm_in[i] = input[k*N_TS*N1_LX+i];
    		std::cout <<", "<< lstm_in[i];
    	}
    	for (int i = 0; i < MODEL_OUT; i++) {
    		lstm_out[i] =0;
    	}

    	lstm(lstm_in, lstm_out);
    	std::cout <<"\n output ";
    	for(int ff = 0; ff < MODEL_OUT; ff++) {
    		std::cout <<", "<< lstm_out[ff];
    	}
    	std::cout << "\n";
    }

    std::cout << "# End of Testbench \n";
}
