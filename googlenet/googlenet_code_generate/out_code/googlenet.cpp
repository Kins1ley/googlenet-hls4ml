#include <iostream>
#include <iomanip>
#include "header.h"
#include "googlenet.h"


int main(int argc, char const *argv[]) {
	
	static FIX_INT20 data_0[IMAGE_CH][IMAGE_H][IMAGE_W];
	//required weight; bias
	//save features that are too large to save in BRAM
	/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)
	static FIX_INT8 DDR_weight7x7[DDR_WEIGHT_7x7_OUT_CHANNEL][DDR_WEIGHT_7x7_IN_CHANNEL][7][7];
	static FIX_INT8 DDR_weight5x5[DDR_WEIGHT_5x5_OUT_CHANNEL][DDR_WEIGHT_5x5_IN_CHANNEL][5][5];
	static FIX_INT8 DDR_weight3x3[DDR_WEIGHT_3x3_OUT_CHANNEL][DDR_WEIGHT_3x3_IN_CHANNEL][3][3];
	static FIX_INT8 DDR_weight1x1[DDR_WEIGHT_1x1_OUT_CHANNEL][DDR_WEIGHT_1x1_IN_CHANNEL][1][1];
	static FIX_INT20 DDR_bias[DDR_BIAS_NUM];
	/////DRAM_insert/////
static FIX_INT20 conv1_7x7_s2_2[conv1_7x7_s2_out_channel][conv1_7x7_s2_out_height][conv1_7x7_s2_out_width];
static FIX_INT20 pool1_3x3_s2_1[pool1_3x3_s2_out_channel][pool1_3x3_s2_out_height][pool1_3x3_s2_out_width];
static FIX_INT20 pool1_norm1_1[pool1_norm1_out_channel][pool1_norm1_out_height][pool1_norm1_out_width];
static FIX_INT20 conv2_3x3_reduce_2[conv2_3x3_reduce_out_channel][conv2_3x3_reduce_out_height][conv2_3x3_reduce_out_width];
static FIX_INT20 conv2_3x3_2[conv2_3x3_out_channel][conv2_3x3_out_height][conv2_3x3_out_width];
static FIX_INT20 conv2_norm2_1[conv2_norm2_out_channel][conv2_norm2_out_height][conv2_norm2_out_width];
static FIX_INT20 pool2_3x3_s2_1[pool2_3x3_s2_out_channel][pool2_3x3_s2_out_height][pool2_3x3_s2_out_width];
static FIX_INT20 inception_3a_output_1[inception_3a_1x1_out_channel+inception_3a_3x3_out_channel+inception_3a_5x5_out_channel+inception_3a_pool_proj_out_channel][inception_3a_1x1_out_height][inception_3a_1x1_out_width];
static FIX_INT20 inception_3a_3x3_reduce_2[inception_3a_3x3_reduce_out_channel][inception_3a_3x3_reduce_out_height][inception_3a_3x3_reduce_out_width];
static FIX_INT20 inception_3a_5x5_reduce_2[inception_3a_5x5_reduce_out_channel][inception_3a_5x5_reduce_out_height][inception_3a_5x5_reduce_out_width];
static FIX_INT20 inception_3a_pool_1[inception_3a_pool_out_channel][inception_3a_pool_out_height][inception_3a_pool_out_width];
static FIX_INT20 inception_3b_output_1[inception_3b_1x1_out_channel+inception_3b_3x3_out_channel+inception_3b_5x5_out_channel+inception_3b_pool_proj_out_channel][inception_3b_1x1_out_height][inception_3b_1x1_out_width];
static FIX_INT20 inception_3b_3x3_reduce_2[inception_3b_3x3_reduce_out_channel][inception_3b_3x3_reduce_out_height][inception_3b_3x3_reduce_out_width];
static FIX_INT20 inception_3b_5x5_reduce_2[inception_3b_5x5_reduce_out_channel][inception_3b_5x5_reduce_out_height][inception_3b_5x5_reduce_out_width];
static FIX_INT20 inception_3b_pool_1[inception_3b_pool_out_channel][inception_3b_pool_out_height][inception_3b_pool_out_width];
static FIX_INT20 pool3_3x3_s2_1[pool3_3x3_s2_out_channel][pool3_3x3_s2_out_height][pool3_3x3_s2_out_width];
static FIX_INT20 inception_4a_output_1[inception_4a_1x1_out_channel+inception_4a_3x3_out_channel+inception_4a_5x5_out_channel+inception_4a_pool_proj_out_channel][inception_4a_1x1_out_height][inception_4a_1x1_out_width];
static FIX_INT20 inception_4a_3x3_reduce_2[inception_4a_3x3_reduce_out_channel][inception_4a_3x3_reduce_out_height][inception_4a_3x3_reduce_out_width];
static FIX_INT20 inception_4a_5x5_reduce_2[inception_4a_5x5_reduce_out_channel][inception_4a_5x5_reduce_out_height][inception_4a_5x5_reduce_out_width];
static FIX_INT20 inception_4a_pool_1[inception_4a_pool_out_channel][inception_4a_pool_out_height][inception_4a_pool_out_width];
static FIX_INT20 inception_4b_output_1[inception_4b_1x1_out_channel+inception_4b_3x3_out_channel+inception_4b_5x5_out_channel+inception_4b_pool_proj_out_channel][inception_4b_1x1_out_height][inception_4b_1x1_out_width];
static FIX_INT20 inception_4b_3x3_reduce_2[inception_4b_3x3_reduce_out_channel][inception_4b_3x3_reduce_out_height][inception_4b_3x3_reduce_out_width];
static FIX_INT20 inception_4b_5x5_reduce_2[inception_4b_5x5_reduce_out_channel][inception_4b_5x5_reduce_out_height][inception_4b_5x5_reduce_out_width];
static FIX_INT20 inception_4b_pool_1[inception_4b_pool_out_channel][inception_4b_pool_out_height][inception_4b_pool_out_width];
static FIX_INT20 inception_4c_output_1[inception_4c_1x1_out_channel+inception_4c_3x3_out_channel+inception_4c_5x5_out_channel+inception_4c_pool_proj_out_channel][inception_4c_1x1_out_height][inception_4c_1x1_out_width];
static FIX_INT20 inception_4c_3x3_reduce_2[inception_4c_3x3_reduce_out_channel][inception_4c_3x3_reduce_out_height][inception_4c_3x3_reduce_out_width];
static FIX_INT20 inception_4c_5x5_reduce_2[inception_4c_5x5_reduce_out_channel][inception_4c_5x5_reduce_out_height][inception_4c_5x5_reduce_out_width];
static FIX_INT20 inception_4c_pool_1[inception_4c_pool_out_channel][inception_4c_pool_out_height][inception_4c_pool_out_width];
static FIX_INT20 inception_4d_output_1[inception_4d_1x1_out_channel+inception_4d_3x3_out_channel+inception_4d_5x5_out_channel+inception_4d_pool_proj_out_channel][inception_4d_1x1_out_height][inception_4d_1x1_out_width];
static FIX_INT20 inception_4d_3x3_reduce_2[inception_4d_3x3_reduce_out_channel][inception_4d_3x3_reduce_out_height][inception_4d_3x3_reduce_out_width];
static FIX_INT20 inception_4d_5x5_reduce_2[inception_4d_5x5_reduce_out_channel][inception_4d_5x5_reduce_out_height][inception_4d_5x5_reduce_out_width];
static FIX_INT20 inception_4d_pool_1[inception_4d_pool_out_channel][inception_4d_pool_out_height][inception_4d_pool_out_width];
static FIX_INT20 inception_4e_output_1[inception_4e_1x1_out_channel+inception_4e_3x3_out_channel+inception_4e_5x5_out_channel+inception_4e_pool_proj_out_channel][inception_4e_1x1_out_height][inception_4e_1x1_out_width];
static FIX_INT20 inception_4e_3x3_reduce_2[inception_4e_3x3_reduce_out_channel][inception_4e_3x3_reduce_out_height][inception_4e_3x3_reduce_out_width];
static FIX_INT20 inception_4e_5x5_reduce_2[inception_4e_5x5_reduce_out_channel][inception_4e_5x5_reduce_out_height][inception_4e_5x5_reduce_out_width];
static FIX_INT20 inception_4e_pool_1[inception_4e_pool_out_channel][inception_4e_pool_out_height][inception_4e_pool_out_width];
static FIX_INT20 pool4_3x3_s2_1[pool4_3x3_s2_out_channel][pool4_3x3_s2_out_height][pool4_3x3_s2_out_width];
static FIX_INT20 inception_5a_output_1[inception_5a_1x1_out_channel+inception_5a_3x3_out_channel+inception_5a_5x5_out_channel+inception_5a_pool_proj_out_channel][inception_5a_1x1_out_height][inception_5a_1x1_out_width];
static FIX_INT20 inception_5a_3x3_reduce_2[inception_5a_3x3_reduce_out_channel][inception_5a_3x3_reduce_out_height][inception_5a_3x3_reduce_out_width];
static FIX_INT20 inception_5a_5x5_reduce_2[inception_5a_5x5_reduce_out_channel][inception_5a_5x5_reduce_out_height][inception_5a_5x5_reduce_out_width];
static FIX_INT20 inception_5a_pool_1[inception_5a_pool_out_channel][inception_5a_pool_out_height][inception_5a_pool_out_width];
static FIX_INT20 inception_5b_output_1[inception_5b_1x1_out_channel+inception_5b_3x3_out_channel+inception_5b_5x5_out_channel+inception_5b_pool_proj_out_channel][inception_5b_1x1_out_height][inception_5b_1x1_out_width];
static FIX_INT20 inception_5b_3x3_reduce_2[inception_5b_3x3_reduce_out_channel][inception_5b_3x3_reduce_out_height][inception_5b_3x3_reduce_out_width];
static FIX_INT20 inception_5b_5x5_reduce_2[inception_5b_5x5_reduce_out_channel][inception_5b_5x5_reduce_out_height][inception_5b_5x5_reduce_out_width];
static FIX_INT20 inception_5b_pool_1[inception_5b_pool_out_channel][inception_5b_pool_out_height][inception_5b_pool_out_width];
static FIX_INT20 pool5_7x7_s1_1[pool5_7x7_s1_out_channel][pool5_7x7_s1_out_height][pool5_7x7_s1_out_width];
static FIX_INT20 out[loss3_classifier_out_channel][loss3_classifier_out_height][loss3_classifier_out_width];

	googlenet(data_0, DDR_weight7x7, DDR_weight5x5, DDR_weight3x3, DDR_weight1x1, DDR_bias,
	/////param_insert/////
conv1_7x7_s2_2 ,
pool1_3x3_s2_1 ,
pool1_norm1_1 ,
conv2_3x3_reduce_2 ,
conv2_3x3_2 ,
conv2_norm2_1 ,
pool2_3x3_s2_1 ,
inception_3a_output_1 ,
inception_3a_3x3_reduce_2 ,
inception_3a_5x5_reduce_2 ,
inception_3a_pool_1 ,
inception_3b_output_1 ,
inception_3b_3x3_reduce_2 ,
inception_3b_5x5_reduce_2 ,
inception_3b_pool_1 ,
pool3_3x3_s2_1 ,
inception_4a_output_1 ,
inception_4a_3x3_reduce_2 ,
inception_4a_5x5_reduce_2 ,
inception_4a_pool_1 ,
inception_4b_output_1 ,
inception_4b_3x3_reduce_2 ,
inception_4b_5x5_reduce_2 ,
inception_4b_pool_1 ,
inception_4c_output_1 ,
inception_4c_3x3_reduce_2 ,
inception_4c_5x5_reduce_2 ,
inception_4c_pool_1 ,
inception_4d_output_1 ,
inception_4d_3x3_reduce_2 ,
inception_4d_5x5_reduce_2 ,
inception_4d_pool_1 ,
inception_4e_output_1 ,
inception_4e_3x3_reduce_2 ,
inception_4e_5x5_reduce_2 ,
inception_4e_pool_1 ,
pool4_3x3_s2_1 ,
inception_5a_output_1 ,
inception_5a_3x3_reduce_2 ,
inception_5a_5x5_reduce_2 ,
inception_5a_pool_1 ,
inception_5b_output_1 ,
inception_5b_3x3_reduce_2 ,
inception_5b_5x5_reduce_2 ,
inception_5b_pool_1 ,
pool5_7x7_s1_1 ,
out 

		);
	std::cout << "hardware outputs: " << std::endl;
	for (int oc = 0; oc < loss3_classifier_out_channel; ++oc) {
		for (int oh = 0; oh < loss3_classifier_out_height; ++oh) {
			for (int ow = 0; ow < loss3_classifier_out_width; ++ow) {
				std::cout.width(4);
				std::cout  << (float)out[oc][oh][ow] ;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;



	return 0;
}

