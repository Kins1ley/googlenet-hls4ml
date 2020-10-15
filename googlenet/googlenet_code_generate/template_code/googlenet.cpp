﻿#include <iostream>
#include <iomanip>
#include "header.h"
#include "googlenet.h"


int main(int argc, char const *argv[]) {
	
	static FIX_INT20 data_0[IMAGE_CH * IMAGE_H * IMAGE_W];
	static FIX_INT20 DDR_feature_0[DDR_FEATURE_LENGTH];
	static FIX_INT20 DDR_feature_1[DDR_FEATURE_LENGTH];
	static FIX_INT20 DDR_feature_2[DDR_FEATURE_LENGTH];
	//required weight; bias
	//save features that are too large to save in BRAM
	/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)
	static FIX_INT8 DDR_weight7x7[DDR_WEIGHT_7x7_CHANNEL][7][7];
	static FIX_INT8 DDR_weight5x5[DDR_WEIGHT_5x5_CHANNEL][5][5];
	static FIX_INT8 DDR_weight3x3[DDR_WEIGHT_3x3_CHANNEL][3][3];
	static FIX_INT8 DDR_weight1x1[DDR_WEIGHT_1x1_CHANNEL][1][1];
	static FIX_INT20 DDR_bias[DDR_BIAS_NUM];
	/////DRAM_insert/////

	googlenet(data_0, DDR_feature_0, DDR_feature_1, DDR_feature_2, DDR_weight7x7, DDR_weight5x5, DDR_weight3x3, DDR_weight1x1, DDR_bias,
	/////param_insert/////

		);
	std::cout << "hardware outputs: " << std::endl;
	for (int oc = 0; oc < loss3_classifier_out_channel; ++oc) {
		for (int oh = 0; oh < loss3_classifier_out_height; ++oh) {
			for (int ow = 0; ow < loss3_classifier_out_width; ++ow) {
				std::cout.width(4);
				std::cout  << (float)DDR_feature_2[oc* loss3_classifier_out_height* loss3_classifier_out_width+oh* loss3_classifier_out_width +ow] ;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;



	return 0;
}

