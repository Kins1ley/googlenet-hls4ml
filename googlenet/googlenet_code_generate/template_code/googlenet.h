#ifndef GOOGLENET_H_
#define GOOGLENET_H_


#include "template_config.h"
#include "header.h"
#include "allocate_config.h"
void googlenet(
	FIX_INT20 data_0[IMAGE_CH][IMAGE_H][IMAGE_W],
	//required weight, bias
	//save features that are too large to save in BRAM
	/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)
	FIX_INT8 DDR_weight_7x7[DDR_WEIGHT_7x7_OUT_CHANNEL][DDR_WEIGHT_7x7_IN_CHANNEL][7][7],
	FIX_INT8 DDR_weight_5x5[DDR_WEIGHT_5x5_OUT_CHANNEL][DDR_WEIGHT_5x5_IN_CHANNEL][5][5],
	FIX_INT8 DDR_weight_3x3[DDR_WEIGHT_3x3_OUT_CHANNEL][DDR_WEIGHT_3x3_IN_CHANNEL][3][3],
	FIX_INT8 DDR_weight_1x1[DDR_WEIGHT_1x1_OUT_CHANNEL][DDR_WEIGHT_1x1_IN_CHANNEL][1][1],
	FIX_INT20 DDR_bias[DDR_BIAS_NUM],
	/////DRAM_insert/////
	/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////(Binwu)


	/////////////////////////////// inception(5a) -> linear              ////////////////////////////(Qi)



) {
#pragma HLS INTERFACE m_axi depth=IMAGE_CH*IMAGE_H*IMAGE_W																port=data_0					offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=DDR_WEIGHT_7x7_OUT_CHANNEL*DDR_WEIGHT_7x7_IN_CHANNEL*7*7								port=DDR_weight7x7			offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=DDR_WEIGHT_5x5_OUT_CHANNEL*DDR_WEIGHT_5x5_IN_CHANNEL*5*5								port=DDR_weight5x5			offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=DDR_WEIGHT_3x3_OUT_CHANNEL*DDR_WEIGHT_3x3_IN_CHANNEL*3*3								port=DDR_weight3x3			offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=DDR_WEIGHT_1x1_OUT_CHANNEL*DDR_WEIGHT_1x1_IN_CHANNEL*1*1								port=DDR_weight1x1			offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=DDR_BIAS_NUM																			port=DDR_bias				offset=slave bundle=INPUT
/////interface_insert/////


#pragma HLS INTERFACE m_axi depth=OU									port=outputs offset=slave bundle=OUTPUT

	//global BRAM
	static FIX_INT8 global_weight_7x7[NUM_WEIGHT_GLOBAL_7x7][OUT_CHANNEL_WEIGHT_GLOBAL_7x7][IN_CHANNEL_WEIGHT_GLOBAL_7x7][7][7];
	static FIX_INT8 global_weight_5x5[NUM_WEIGHT_GLOBAL_5x5][OUT_CHANNEL_WEIGHT_GLOBAL_5x5][IN_CHANNEL_WEIGHT_GLOBAL_5x5][5][5];
	static FIX_INT8 global_weight_3x3[NUM_WEIGHT_GLOBAL_3x3][OUT_CHANNEL_WEIGHT_GLOBAL_3x3][IN_CHANNEL_WEIGHT_GLOBAL_3x3][3][3];
	static FIX_INT8 global_weight_1x1[NUM_WEIGHT_GLOBAL_1x1][OUT_CHANNEL_WEIGHT_GLOBAL_1x1][IN_CHANNEL_WEIGHT_GLOBAL_1x1][1][1];
	static FIX_INT20 global_feature[NUM_FEATURE_GLOBAL][CHANNEL_FEATURE_GLOBAL][WIDTH_FEATURE_GLOBAL][HEIGHT_FEATURE_GLOBAL];

	//local BRAM of conv layers
	static FIX_INT20 local_feature_in_CONV1x1_S1[NUM_PE_CONV1x1_S1][IN_CHAN_CONV1x1_S1][IN_HEIGHT_CONV1x1_S1][IN_WIDTH_CONV1x1_S1];
	static FIX_INT8 local_weight_CONV1x1_S1[NUM_PE_CONV1x1_S1][OUT_CHAN_CONV1x1_S1][IN_CHAN_CONV1x1_S1][1][1];
	static FIX_INT20 local_feature_out_CONV1x1_S1[NUM_PE_CONV1x1_S1][OUT_CHAN_CONV1x1_S1][OUT_HEIGHT_CONV1x1_S1][OUT_WIDTH_CONV1x1_S1];

	static FIX_INT20 local_feature_in_CONV3x3_S1[NUM_PE_CONV3x3_S1][IN_CHAN_CONV3x3_S1][IN_HEIGHT_CONV3x3_S1][IN_WIDTH_CONV3x3_S1];
	static FIX_INT8 local_weight_CONV3x3_S1[NUM_PE_CONV3x3_S1][OUT_CHAN_CONV3x3_S1][IN_CHAN_CONV3x3_S1][3][3];
	static FIX_INT20 local_feature_out_CONV3x3_S1[NUM_PE_CONV3x3_S1][OUT_CHAN_CONV3x3_S1][OUT_HEIGHT_CONV3x3_S1][OUT_WIDTH_CONV3x3_S1];

	static FIX_INT20 local_feature_in_CONV5x5_S1[NUM_PE_CONV5x5_S1][IN_CHAN_CONV5x5_S1][IN_HEIGHT_CONV5x5_S1][IN_WIDTH_CONV5x5_S1];
	static FIX_INT8 local_weight_CONV5x5_S1[NUM_PE_CONV5x5_S1][OUT_CHAN_CONV5x5_S1][IN_CHAN_CONV5x5_S1][5][5];
	static FIX_INT20 local_feature_out_CONV5x5_S1[NUM_PE_CONV5x5_S1][OUT_CHAN_CONV5x5_S1][OUT_HEIGHT_CONV5x5_S1][OUT_WIDTH_CONV5x5_S1];

	static FIX_INT20 local_feature_in_CONV7x7_S2[NUM_PE_CONV7x7_S2][IN_CHAN_CONV7x7_S2][IN_HEIGHT_CONV7x7_S2][IN_WIDTH_CONV7x7_S2];
	static FIX_INT8 local_weight_CONV7x7_S2[NUM_PE_CONV7x7_S2][OUT_CHAN_CONV7x7_S2][IN_CHAN_CONV7x7_S2][7][7];
	static FIX_INT20 local_feature_out_CONV7x7_S2[NUM_PE_CONV7x7_S2][OUT_CHAN_CONV7x7_S2][OUT_HEIGHT_CONV7x7_S2][OUT_WIDTH_CONV7x7_S2];


	//local BRAM of pooling layers
	static FIX_INT20 local_feature_in_MAXPOOL3x3_S1[NUM_PE_MAXPOOL3x3_S1][N_CHAN_MAXPOOL3x3_S1][IN_HEIGHT_MAXPOOL3x3_S1][IN_WIDTH_MAXPOOL3x3_S1];
	static FIX_INT20 local_feature_out_MAXPOOL3x3_S1[NUM_PE_MAXPOOL3x3_S1][N_CHAN_MAXPOOL3x3_S1][OUT_HEIGHT_MAXPOOL3x3_S1][OUT_WIDTH_MAXPOOL3x3_S1];

	static FIX_INT20 local_feature_in_MAXPOOL3x3_S2[NUM_PE_MAXPOOL3x3_S2][N_CHAN_MAXPOOL3x3_S2][IN_HEIGHT_MAXPOOL3x3_S2][IN_WIDTH_MAXPOOL3x3_S2];
	static FIX_INT20 local_feature_out_MAXPOOL3x3_S2[NUM_PE_MAXPOOL3x3_S2][N_CHAN_MAXPOOL3x3_S2][OUT_HEIGHT_MAXPOOL3x3_S2][OUT_WIDTH_MAXPOOL3x3_S2];

	static FIX_INT20 local_feature_in_AVGPOOL7x7_S1[NUM_PE_AVGPOOL7x7_S1][N_CHAN_AVGPOOL7x7_S1][IN_HEIGHT_AVGPOOL7x7_S1][IN_WIDTH_AVGPOOL7x7_S1];
	static FIX_INT20 local_feature_out_AVGPOOL7x7_S1[NUM_PE_AVGPOOL7x7_S1][N_CHAN_AVGPOOL7x7_S1][OUT_HEIGHT_AVGPOOL7x7_S1][OUT_WIDTH_AVGPOOL7x7_S1];

	//local BRAM of LRN
	static FIX_INT20 local_feature_in_LRN[NUM_PE_LRN][N_CHAN_LRN][IN_HEIGHT_LRN][IN_WIDTH_LRN];
	static FIX_INT20 local_feature_out_LRN[NUM_PE_LRN][N_CHAN_LRN][OUT_HEIGHT_LRN][OUT_WIDTH_LRN];

	//other param of LRN
	static FIX_INT20 bias = 1;
	static FIX_INT20 alpha = 1;
	static FIX_INT20 beta = 1;

	/////top_function_insert/////
	/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(top_function)


	/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////(top_function)


	/////////////////////////////// inception(5a) -> linear              ////////////////////////////(top_function)
}

#endif