#include <iostream>
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

	nnet::load_data_from_txt<FIX_INT20, 3*224*224>(data_0, "image_in.txt");
	nnet::load_weights_from_txt<FIX_INT8, 9408/7/7  ,7 ,7>(DDR_weight7x7, "weight_7x7.txt");
	nnet::load_weights_from_txt<FIX_INT8, 595200 /5/5 , 5 , 5>(DDR_weight5x5, "weight_5x5.txt");
	nnet::load_weights_from_txt<FIX_INT8, 3101184 /3/3 , 3 ,3>(DDR_weight3x3, "weight_3x3.txt");
	nnet::load_weights_from_txt<FIX_INT8, 3284480 /1/1 , 1 , 1>(DDR_weight1x1, "weight_1x1.txt");
	nnet::load_data_from_txt<FIX_INT20, 8280>(DDR_bias, "bias.txt");

	/////DRAM_insert/////

	googlenet(data_0, DDR_feature_0, DDR_feature_1, DDR_feature_2, DDR_weight7x7, DDR_weight5x5, DDR_weight3x3, DDR_weight1x1, DDR_bias
	/////param_insert/////

		);
	std::cout << "hardware outputs: " << std::endl;
	int max_idx = 0;
	float max_logits = 0;
	for (int oc = 0; oc < loss3_classifier_out_channel; ++oc) {
		for (int oh = 0; oh < loss3_classifier_out_height; ++oh) {
			for (int ow = 0; ow < loss3_classifier_out_width; ++ow) {
				std::cout.width(4);
				std::cout  << (float)DDR_feature_1[oc* loss3_classifier_out_height* loss3_classifier_out_width+oh* loss3_classifier_out_width +ow] ;
				if ((float)DDR_feature_1[oc * loss3_classifier_out_height * loss3_classifier_out_width + oh * loss3_classifier_out_width + ow] > max_logits) {
					max_logits = (float)DDR_feature_1[oc * loss3_classifier_out_height * loss3_classifier_out_width + oh * loss3_classifier_out_width + ow];
					max_idx = oc;
				}
				
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout <<"label:"<< max_idx << std::endl;


	return 0;
}

