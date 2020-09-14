#ifndef GOOGLENET_H_
#define GOOGLENET_H_


#include "template_config.h"
#include "header.h"
#include "allocate_config.h"
void googlenet(
	FIX_INT20 image_in[IMAGE_CH][IMAGE_H][IMAGE_W],
	//required weight, bias
	//save features that are too large to save in BRAM
	/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)
	FIX_INT8 conv1_7x7_s2_w_0[conv1_7x7_s2_kernel_num][conv1_7x7_s2_kernel_channel][conv1_7x7_s2_kernel_height][conv1_7x7_s2_kernel_width],
	FIX_INT20 conv1_7x7_s2_b_0[conv1_7x7_s2_bias_num],
	FIX_INT20 conv1_7x7_s2_1[conv1_7x7_s2_out_channel][conv1_7x7_s2_out_height][conv1_7x7_s2_out_width],

	FIX_INT20 pool3_3x3_s2[pool3_3x3_s2_out_channel][pool3_3x3_s2_out_height][pool3_3x3_s2_out_width],
	/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////(Binwu)



	FIX_INT20 pool4_3x3_s2[pool4_3x3_s2_out_channel][pool4_3x3_s2_out_height][pool4_3x3_s2_out_width],
	/////////////////////////////// inception(5a) -> linear              ////////////////////////////(Qi)


	FIX_INT20 out[OU]

) {
#pragma HLS INTERFACE m_axi depth=IMAGE_CH*IMAGE_H*IMAGE_W				port=inputs				offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=64*3*7*7								port=conv1_7x7_s2_w_0	offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=64									port=conv1_7x7_s2_b_0	offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=64*112*112							port=conv1_7x7_s2_1		offset=slave bundle=INPUT


#pragma HLS INTERFACE m_axi depth=OU									port=outputs offset=slave bundle=OUTPUT

	//global BRAM
	FIX_INT8 global_weight_7x7[NUM_WEIGHT_GLOBAL_7x7][OUT_CHANNEL_WEIGHT_GLOBAL_7x7][IN_CHANNEL_WEIGHT_GLOBAL_7x7][7][7];
	FIX_INT8 global_weight_5x5[NUM_WEIGHT_GLOBAL_5x5][OUT_CHANNEL_WEIGHT_GLOBAL_5x5][IN_CHANNEL_WEIGHT_GLOBAL_5x5][5][5];
	FIX_INT8 global_weight_3x3[NUM_WEIGHT_GLOBAL_3x3][OUT_CHANNEL_WEIGHT_GLOBAL_3x3][IN_CHANNEL_WEIGHT_GLOBAL_3x3][3][3];
	FIX_INT8 global_weight_1x1[NUM_WEIGHT_GLOBAL_1x1][OUT_CHANNEL_WEIGHT_GLOBAL_1x1][IN_CHANNEL_WEIGHT_GLOBAL_1x1][1][1];
	FIX_INT20 global_feature[NUM_FEATURE_GLOBAL][CHANNEL_FEATURE_GLOBAL][WIDTH_FEATURE_GLOBAL][HEIGHT_FEATURE_GLOBAL];

	//local BRAM of conv layers
	FIX_INT20 local_feature_in_conv1x1_s1[NUM_PE_CONV1x1_S1][IN_CHAN_CONV1x1_S1][IN_HEIGHT_CONV1x1_S1][IN_WIDTH_CONV1x1_S1];
	FIX_INT8 local_weight_conv1x1_s1[NUM_PE_CONV1x1_S1][OUT_CHAN_CONV1x1_S1][IN_CHAN_CONV1x1_S1][1][1];
	FIX_INT20 local_feature_out_conv1x1_s1[NUM_PE_CONV1x1_S1][OUT_CHAN_CONV1x1_S1][OUT_HEIGHT_CONV1x1_S1][OUT_WIDTH_CONV1x1_S1];

	FIX_INT20 local_feature_in_conv3x3_s1[NUM_PE_CONV3x3_S1][IN_CHAN_CONV3x3_S1][IN_HEIGHT_CONV3x3_S1][IN_WIDTH_CONV3x3_S1];
	FIX_INT8 local_weight_conv3x3_s1[NUM_PE_CONV3x3_S1][OUT_CHAN_CONV3x3_S1][IN_CHAN_CONV3x3_S1][3][3];
	FIX_INT20 local_feature_out_conv3x3_s1[NUM_PE_CONV3x3_S1][OUT_CHAN_CONV3x3_S1][OUT_HEIGHT_CONV3x3_S1][OUT_WIDTH_CONV3x3_S1];

	FIX_INT20 local_feature_in_conv5x5_s1[NUM_PE_CONV5x5_S1][IN_CHAN_CONV5x5_S1][IN_HEIGHT_CONV5x5_S1][IN_WIDTH_CONV5x5_S1];
	FIX_INT8 local_weight_conv5x5_s1[NUM_PE_CONV5x5_S1][OUT_CHAN_CONV5x5_S1][IN_CHAN_CONV5x5_S1][5][5];
	FIX_INT20 local_feature_out_conv5x5_s1[NUM_PE_CONV5x5_S1][OUT_CHAN_CONV5x5_S1][OUT_HEIGHT_CONV5x5_S1][OUT_WIDTH_CONV5x5_S1];

	FIX_INT20 local_feature_in_conv7x7_s2[NUM_PE_CONV7x7_S2][IN_CHAN_CONV7x7_S2][IN_HEIGHT_CONV7x7_S2][IN_WIDTH_CONV7x7_S2];
	FIX_INT8 local_weight_conv7x7_s2[NUM_PE_CONV7x7_S2][OUT_CHAN_CONV7x7_S2][IN_CHAN_CONV7x7_S2][7][7];
	FIX_INT20 local_feature_out_conv7x7_s2[NUM_PE_CONV7x7_S2][OUT_CHAN_CONV7x7_S2][OUT_HEIGHT_CONV7x7_S2][OUT_WIDTH_CONV7x7_S2];


	//local BRAM of pooling layers
	FIX_INT20 local_feature_in_maxpool3x3_s1[NUM_PE_MAXPOOL3x3_S1][N_CHAN_MAXPOOL3x3_S1][IN_HEIGHT_MAXPOOL3x3_S1][IN_WIDTH_MAXPOOL3x3_S1];
	FIX_INT20 local_feature_out_maxpool3x3_s1[NUM_PE_MAXPOOL3x3_S1][N_CHAN_MAXPOOL3x3_S1][OUT_HEIGHT_MAXPOOL3x3_S1][OUT_WIDTH_MAXPOOL3x3_S1];

	FIX_INT20 local_feature_in_maxpool3x3_s2[NUM_PE_MAXPOOL3x3_S2][N_CHAN_MAXPOOL3x3_S2][IN_HEIGHT_MAXPOOL3x3_S2][IN_WIDTH_MAXPOOL3x3_S2];
	FIX_INT20 local_feature_out_maxpool3x3_s2[NUM_PE_MAXPOOL3x3_S2][N_CHAN_MAXPOOL3x3_S2][OUT_HEIGHT_MAXPOOL3x3_S2][OUT_WIDTH_MAXPOOL3x3_S2];

	FIX_INT20 local_feature_in_avgpool7x7_s1[NUM_PE_AVGPOOL7x7_S1][N_CHAN_AVGPOOL7x7_S1][IN_HEIGHT_AVGPOOL7x7_S1][IN_WIDTH_AVGPOOL7x7_S1];
	FIX_INT20 local_feature_out_avgpool7x7_s1[NUM_PE_AVGPOOL7x7_S1][N_CHAN_AVGPOOL7x7_S1][OUT_HEIGHT_AVGPOOL7x7_S1][OUT_WIDTH_AVGPOOL7x7_S1];

	/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////
	//conv1_7x7_s2
	//outer loop
	//copy data and call PE to do calculation
	for (int outer_h_idx = 0; outer_h_idx < conv1_7x7_s2_outer_height; outer_h_idx++) {
		for (int outer_w_idx = 0; outer_w_idx < conv1_7x7_s2_outer_width; outer_w_idx++) {
			for (int outer_oc_idx; outer_oc_idx < conv1_7x7_s2_outer_out_channel; outer_oc_idx++) {
				for (int outer_ic_idx = 0; outer_ic_idx < conv1_7x7_s2_outer_in_channel; outer_ic_idx++) {
					//calculate the index to copy features and weights. padding considered here. 
					int DDR_feature_h_start_idx = outer_h_idx * conv1_7x7_s2_out_height_per_block*conv1_7x7_s2_stride;
					int DDR_feature_w_start_idx = outer_w_idx * conv1_7x7_s2_out_width_per_block*conv1_7x7_s2_stride;
					int DDR_feature_c_start_idx = outer_ic_idx * CHANNEL_FEATURE_GLOBAL;
					int global_feature_h_start_idx = 0;
					int global_feature_w_start_idx = 0;
					int global_feature_c_start_idx = 0;
					int global_feature_c_num = CHANNEL_FEATURE_GLOBAL;
					int global_feature_h_num = (conv1_7x7_s2_out_height_per_block-1)* conv1_7x7_s2_stride+ conv1_7x7_s2_kernel_height;
					int global_feature_w_num = (conv1_7x7_s2_out_width_per_block - 1)* conv1_7x7_s2_stride + conv1_7x7_s2_kernel_width;
					int DDR_weight_ic_start_idx = outer_ic_idx * CHANNEL_FEATURE_GLOBAL;
					int DDR_weight_oc_start_idx = outer_oc_idx * OUT_CHANNEL_WEIGHT_GLOBAL_7x7;
					int global_weight_ic_num = CHANNEL_FEATURE_GLOBAL;
					int global_weight_oc_num = OUT_CHANNEL_WEIGHT_GLOBAL_7x7;

					if (outer_h_idx == 0) {
						global_feature_h_start_idx += conv1_7x7_s2_pad_top;
						global_feature_h_num -= conv1_7x7_s2_pad_top;
					}
					if (outer_w_idx == 0) {
						global_feature_w_start_idx += conv1_7x7_s2_pad_left;
						global_feature_w_num -= conv1_7x7_s2_pad_left;
					}
					if (outer_h_idx == conv1_7x7_s2_outer_height-1) {
						global_feature_h_num -= conv1_7x7_s2_pad_bottom;
					}
					if (outer_w_idx == conv1_7x7_s2_outer_width-1) {
						global_feature_w_num -= conv1_7x7_s2_pad_right;
					}
					if (outer_ic_idx == conv1_7x7_s2_outer_in_channel - 1) {
						global_feature_c_num = conv1_7x7_s2_in_channel - outer_ic_idx * CHANNEL_FEATURE_GLOBAL;
						global_weight_ic_num = conv1_7x7_s2_in_channel - outer_ic_idx * CHANNEL_FEATURE_GLOBAL;
					}
					nnet::copy_features_DDR2BRAM_wjp<DDR_feature_image_in_config, global_feature_config>(image_in, global_feature[0],
						DDR_feature_c_start_idx, global_feature_c_start_idx, global_feature_c_num,
						DDR_feature_h_start_idx, global_feature_h_start_idx, global_feature_h_num,
						DDR_feature_w_start_idx, global_feature_w_start_idx, global_feature_w_num);
					nnet::copy_weights_DDR2BRAM<conv7x7_DDR_weight_config, conv7x7_global_weight_config>(conv1_7x7_s2_w_0, global_weight_7x7[0],
						DDR_weight_oc_start_idx, global_weight_oc_num,
						DDR_weight_ic_start_idx, global_weight_ic_num);
					//inner loop
					for (int h_idx = 0; h_idx < conv1_7x7_s2_inner_height; h_idx++) {
						for (int w_idx = 0; w_idx < conv1_7x7_s2_inner_width; w_idx++) {
							for (int o_idx = 0; o_idx < conv1_7x7_s2_inner_out_channel/ conv1_7x7_s2_inner_pe_parallel; o_idx++) {
								for (int pe_idx = 0; pe_idx < conv1_7x7_s2_inner_pe_parallel; pe_idx++) {
#pragma HLS unroll
									for (int i_idx = 0; i_idx < conv1_7x7_s2_inner_in_channel; i_idx++) {
#pragma HLS pipeline
										if (i_idx==0){
										nnet::set_bias<conv1_7x7_s2_set_bias_config>(local_feature_out_conv7x7_s2[pe_idx], conv1_7x7_s2_b_0[pe_idx + o_idx * conv1_7x7_s2_inner_pe_parallel]);
										}
										nnet::copy_features_g2l<global_feature_config, conv7x7_s2_local_feature_in_config>(global_feature[0],local_feature_in_conv7x7_s2[pe_idx],
											i_idx*IN_CHAN_CONV7x7_S2,0, IN_CHAN_CONV7x7_S2,
											h_idx*2, 0, IN_HEIGHT_CONV7x7_S2,
											w_idx*2, 0, IN_WIDTH_CONV7x7_S2);
										nnet::copy_weights_g2l<conv7x7_global_weight_config, conv7x7_s2_local_weight_config>(global_weight_7x7[0], local_weight_conv7x7_s2[pe_idx],
											o_idx*conv1_7x7_s2_inner_pe_parallel, OUT_CHAN_CONV7x7_S2,
											i_idx*IN_CHAN_CONV7x7_S2, IN_CHAN_CONV7x7_S2 );
										nnet::conv_output_reuse7x7<conv2d_config_7x7_s2>(local_feature_in_conv7x7_s2[pe_idx], local_weight_conv7x7_s2[pe_idx], local_feature_out_conv7x7_s2[pe_idx]);
									}
									nnet::copy_features_l2g<conv7x7_s2_local_feature_in_config, global_feature_config>(local_feature_out_conv7x7_s2[pe_idx], global_feature[1],
										o_idx*OUT_CHAN_CONV7x7_S2, OUT_CHAN_CONV7x7_S2,
										h_idx*OUT_HEIGHT_CONV7x7_S2, OUT_HEIGHT_CONV7x7_S2,
										w_idx*OUT_WIDTH_CONV7x7_S2, OUT_WIDTH_CONV7x7_S2);
								}
							}
						}
					}
					nnet::copy_features_BRAM2DDR<global_feature_config>(global_feature[1], conv1_7x7_s2_1);
				}
			}
		}
	}



	/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////


	/////////////////////////////// inception(5a) -> linear              ////////////////////////////
}