/////////////top_function/////////////
//conv1_7x7_s2
//outer loop
//copy data and call PE to do calculation
for (int outer_h_idx = 0; outer_h_idx < conv1_7x7_s2_outer_height; outer_h_idx++) {
	for (int outer_w_idx = 0; outer_w_idx < conv1_7x7_s2_outer_width; outer_w_idx++) {
		for (int outer_oc_idx = 0; outer_oc_idx < conv1_7x7_s2_outer_out_channel; outer_oc_idx++) {
			for (int outer_ic_idx = 0; outer_ic_idx < conv1_7x7_s2_outer_in_channel; outer_ic_idx++) {
				//std::cout << "outer loop" << outer_h_idx << outer_w_idx << outer_oc_idx << outer_ic_idx << std::endl;

				//calculate the index to copy features and weights.
				//index and shape of input feature in DRAM
				int DDR_block_in_feature_h_start_idx = outer_h_idx * conv1_7x7_s2_block_interval_height;
				int DDR_block_in_feature_w_start_idx = outer_w_idx * conv1_7x7_s2_block_interval_width;
				int DDR_block_in_feature_c_start_idx = outer_ic_idx * conv1_7x7_s2_block_in_channel;
				int global_block_in_feature_c_num = conv1_7x7_s2_block_in_channel;
				int global_block_in_feature_h_num = conv1_7x7_s2_block_in_height;
				int global_block_in_feature_w_num = conv1_7x7_s2_block_in_width;

				//index and shape of weight in DRAM
				int DDR_weight_ic_start_idx = outer_ic_idx * conv1_7x7_s2_block_in_channel;
				int DDR_weight_oc_start_idx = outer_oc_idx * conv1_7x7_s2_block_out_channel;
				int global_weight_ic_num = conv1_7x7_s2_block_in_channel;
				int global_weight_oc_num = conv1_7x7_s2_block_out_channel;

				{
					//handle the last iteration of the loop
					if (outer_h_idx == conv1_7x7_s2_outer_height - 1) {
						global_block_in_feature_h_num = conv1_7x7_s2_in_height - DDR_block_in_feature_h_start_idx;
					}
					if (outer_w_idx == conv1_7x7_s2_outer_width - 1) {
						global_block_in_feature_w_num = conv1_7x7_s2_in_width - DDR_block_in_feature_w_start_idx;
					}
					if (outer_oc_idx == conv1_7x7_s2_outer_out_channel - 1) {
						global_weight_oc_num = conv1_7x7_s2_out_channel - outer_oc_idx * conv1_7x7_s2_block_out_channel;
					}
					if (outer_ic_idx == conv1_7x7_s2_outer_in_channel - 1) {
						global_block_in_feature_c_num = conv1_7x7_s2_in_channel - outer_ic_idx * conv1_7x7_s2_block_in_channel;
						global_weight_ic_num = conv1_7x7_s2_in_channel - outer_ic_idx * conv1_7x7_s2_block_in_channel;
					}
				}
				//copy input feature and weight from DRAM to global BRAM
				for (int global_in_feature_idx = 0; global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_in_feature_idx++) {
					if (global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1) {
						nnet::clear_buffer<global_feature_config>(global_feature[conv1_7x7_s2_allocate_global_in_feature_start_idx + global_in_feature_idx]);
						nnet::copy_features_DDR2BRAM<DDR_feature_image_in_config, global_feature_config>(image_in, global_feature[conv1_7x7_s2_allocate_global_in_feature_start_idx + global_in_feature_idx],
							DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
							DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
							DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
					}
					else {
						nnet::clear_buffer<global_feature_config>(global_feature[conv1_7x7_s2_allocate_global_in_feature_start_idx + global_in_feature_idx]);
						nnet::copy_features_DDR2BRAM<DDR_feature_image_in_config, global_feature_config>(image_in, global_feature[conv1_7x7_s2_allocate_global_in_feature_start_idx + global_in_feature_idx],
							DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, global_block_in_feature_c_num - global_in_feature_idx * CHANNEL_FEATURE_GLOBAL,
							DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
							DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
					}
				}
				for (int global_weight_idx = 0; global_weight_idx < DIV_CEIL(global_weight_oc_num, OUT_CHANNEL_WEIGHT_GLOBAL_7x7); global_weight_idx++) {
					if (global_weight_idx < DIV_CEIL(global_weight_oc_num, OUT_CHANNEL_WEIGHT_GLOBAL_7x7) - 1)
						nnet::copy_weights_DDR2BRAM<DDR_weight_conv1_7x7_s2_config, WEIGHT_GLOBAL_7x7_config>(DDR_weight_7x7, global_weight_7x7[conv1_7x7_s2_allocate_global_weight_7x7_start_idx + global_weight_idx],
							DDR_weight_oc_start_idx + global_weight_idx * OUT_CHANNEL_WEIGHT_GLOBAL_7x7, OUT_CHANNEL_WEIGHT_GLOBAL_7x7,
							conv1_7x7_s2_kernel_channel_DDR_offset + DDR_weight_ic_start_idx, global_weight_ic_num);
					else
						nnet::copy_weights_DDR2BRAM<DDR_weight_conv1_7x7_s2_config, WEIGHT_GLOBAL_7x7_config>(DDR_weight_7x7, global_weight_7x7[conv1_7x7_s2_allocate_global_weight_7x7_start_idx + global_weight_idx],
							DDR_weight_oc_start_idx + global_weight_idx * OUT_CHANNEL_WEIGHT_GLOBAL_7x7, global_weight_oc_num - global_weight_idx * OUT_CHANNEL_WEIGHT_GLOBAL_7x7,
							conv1_7x7_s2_kernel_channel_DDR_offset + DDR_weight_ic_start_idx, global_weight_ic_num);
				}

				//std::cout << "(block)processing feature \n start_idx " << DDR_block_in_feature_c_start_idx<<","<< DDR_block_in_feature_h_start_idx << "," << DDR_block_in_feature_w_start_idx<<std::endl;
				//std::cout << "number " << global_block_in_feature_c_num << "," << global_block_in_feature_h_num << "," << global_block_in_feature_w_num <<std::endl;

				//dims of inner loop
				int inner_pad_top = (outer_h_idx == 0 ? conv1_7x7_s2_pad_top : 0);
				int inner_pad_bottom = (outer_h_idx == (conv1_7x7_s2_outer_height - 1) ? conv1_7x7_s2_pad_bottom : 0);
				int inner_pad_left = (outer_w_idx == 0 ? conv1_7x7_s2_pad_left : 0);
				int inner_pad_right = (outer_w_idx == (conv1_7x7_s2_outer_width - 1) ? conv1_7x7_s2_pad_bottom : 0);
				int inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + conv1_7x7_s2_pad_top + global_block_in_feature_h_num + inner_pad_bottom - KERNEL_HEIGHT_CONV7x7_S2) / (STRIDE_CONV7x7_S2)+1
					- DIV_CEIL(DDR_block_in_feature_h_start_idx + conv1_7x7_s2_pad_top, STRIDE_CONV7x7_S2),
					OUT_HEIGHT_CONV7x7_S2);
				int inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + conv1_7x7_s2_pad_left + global_block_in_feature_w_num + inner_pad_right - KERNEL_WIDTH_CONV7x7_S2) / (STRIDE_CONV7x7_S2)+1
					- DIV_CEIL(DDR_block_in_feature_w_start_idx + conv1_7x7_s2_pad_left, STRIDE_CONV7x7_S2),
					OUT_WIDTH_CONV7x7_S2);
				if (outer_h_idx == 0) {
					inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + global_block_in_feature_h_num + inner_pad_bottom + conv1_7x7_s2_pad_top - KERNEL_HEIGHT_CONV7x7_S2) / (STRIDE_CONV7x7_S2)+1
						- DIV_CEIL(DDR_block_in_feature_h_start_idx, STRIDE_CONV7x7_S2),
						OUT_HEIGHT_CONV7x7_S2);
				}
				if (outer_w_idx == 0) {
					inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + global_block_in_feature_w_num + inner_pad_right + conv1_7x7_s2_pad_left - KERNEL_WIDTH_CONV7x7_S2) / (STRIDE_CONV7x7_S2)+1
						- DIV_CEIL(DDR_block_in_feature_w_start_idx, STRIDE_CONV7x7_S2),
						OUT_WIDTH_CONV7x7_S2);
				}
				int inner_out_channel = DIV_CEIL(global_weight_oc_num, conv1_7x7_s2_inner_pe_parallel * OUT_CHAN_CONV7x7_S2);
				int inner_in_channel = DIV_CEIL(global_weight_ic_num, IN_CHAN_CONV7x7_S2);
				//do inner loop
				for (int h_idx = 0; h_idx < inner_height; h_idx++) {
					for (int w_idx = 0; w_idx < inner_width; w_idx++) {
						for (int o_idx = 0; o_idx < inner_out_channel; o_idx++) {
							int inner_pe_parallel = conv1_7x7_s2_inner_pe_parallel;
							if (o_idx == inner_out_channel - 1) inner_pe_parallel = global_weight_oc_num - o_idx * conv1_7x7_s2_inner_pe_parallel;
							for (int i_idx = 0; i_idx < inner_in_channel; i_idx++) {
#pragma HLS pipeline
								for (int pe_idx = 0; pe_idx < inner_pe_parallel; pe_idx++) {
#pragma HLS unroll
									//index and shape of weight in global BRAM
									int global_weight_ic_start_idx = i_idx * IN_CHAN_CONV7x7_S2;
									int global_weight_oc_start_idx = (o_idx * conv1_7x7_s2_inner_pe_parallel + pe_idx) * OUT_CHAN_CONV7x7_S2;
									int local_weight_ic_num = IN_CHAN_CONV7x7_S2;
									int local_weight_oc_num = OUT_CHAN_CONV7x7_S2;

									//index of input feature in global BRAM
									int global_in_feature_c_start_idx = i_idx * IN_CHAN_CONV7x7_S2;
									int global_in_feature_h_start_idx = h_idx * OUT_HEIGHT_CONV7x7_S2 * STRIDE_CONV7x7_S2 - inner_pad_top; //
									int global_in_feature_w_start_idx = w_idx * OUT_WIDTH_CONV7x7_S2 * STRIDE_CONV7x7_S2 - inner_pad_left;//

									//index and shape of input feature in local BRAM
									int local_in_feature_c_start_idx = 0;
									int local_in_feature_h_start_idx = 0;
									int local_in_feature_w_start_idx = 0;
									int local_in_feature_c_num = IN_CHAN_CONV7x7_S2;
									int local_in_feature_h_num = IN_HEIGHT_CONV7x7_S2;
									int local_in_feature_w_num = IN_WIDTH_CONV7x7_S2;

									//index of output feature in global BRAM
									int global_out_feature_c_start_idx = global_weight_oc_start_idx;
									int global_out_feature_h_start_idx = h_idx * OUT_HEIGHT_CONV7x7_S2;
									int global_out_feature_w_start_idx = w_idx * OUT_WIDTH_CONV7x7_S2;

									//index and shape of output feature in local BRAM
									int local_out_feature_c_start_idx = 0;
									int local_out_feature_h_start_idx = 0;
									int local_out_feature_w_start_idx = 0;
									int local_out_feature_c_num = local_weight_oc_num;
									int local_out_feature_h_num = OUT_HEIGHT_CONV7x7_S2;
									int local_out_feature_w_num = OUT_WIDTH_CONV7x7_S2;


									if (h_idx == 0) {
										//handle padding
										local_in_feature_h_num -= inner_pad_top;
										local_in_feature_h_start_idx = inner_pad_top;
										global_in_feature_h_start_idx = 0;
									}
									else if (h_idx == inner_height - 1) {
										//handle the last iteration of the loop and padding
										local_in_feature_h_num = global_block_in_feature_h_num + inner_pad_top - h_idx * OUT_HEIGHT_CONV7x7_S2 * STRIDE_CONV7x7_S2;
									}

									if (w_idx == 0) {
										//handle padding
										local_in_feature_w_num -= inner_pad_left;
										local_in_feature_w_start_idx = inner_pad_left;
										global_in_feature_w_start_idx = 0;
									}
									else if (w_idx == inner_width - 1) {
										//handle the last iteration of the loop and padding
										local_in_feature_w_num = global_block_in_feature_w_num + inner_pad_left - w_idx * OUT_WIDTH_CONV7x7_S2 * STRIDE_CONV7x7_S2;
									}
									if (o_idx == inner_out_channel - 1) {
										//handle the last iteration of the loop
										local_weight_oc_num = global_weight_oc_num - o_idx * OUT_CHAN_CONV7x7_S2 * conv1_7x7_s2_inner_pe_parallel - OUT_CHAN_CONV7x7_S2 * pe_idx;
									}
									if (i_idx == inner_in_channel - 1) {
										//handle the last iteration of the loop
										local_in_feature_c_num = global_block_in_feature_c_num - i_idx * IN_CHAN_CONV7x7_S2;
										local_weight_ic_num = global_block_in_feature_c_num - i_idx * IN_CHAN_CONV7x7_S2;
									}
									// handle the situation that convolution does not start from the first element
									if (outer_h_idx != 0) {
										global_in_feature_h_start_idx += (DDR_block_in_feature_h_start_idx + conv1_7x7_s2_pad_top) % STRIDE_CONV7x7_S2;
									}
									if (outer_w_idx != 0) {
										global_in_feature_w_start_idx += (DDR_block_in_feature_w_start_idx + conv1_7x7_s2_pad_left) % STRIDE_CONV7x7_S2;
									}

									if (i_idx == 0) {
										if (outer_ic_idx == 0) {
											//set bias
											//std::cout << "clearing buffer for bias" << std::endl;
											nnet::clear_buffer<CONV7x7_S2_local_feature_out_config>(local_feature_out_CONV7x7_S2[pe_idx]);
											nnet::set_bias<CONV7x7_S2_set_bias_config>(local_feature_out_CONV7x7_S2[pe_idx], DDR_bias + conv1_7x7_s2_bias_DDR_offset + (conv1_7x7_s2_allocate_bias_start_idx + pe_idx + o_idx * conv1_7x7_s2_inner_pe_parallel + outer_oc_idx * conv1_7x7_s2_block_out_channel));
											//std::cout << "";
										}
										else {
											//restore partial sum
											//std::cout << "clearing buffer for restoring partial sum" << std::endl;
											nnet::clear_buffer<CONV7x7_S2_local_feature_out_config>(local_feature_out_CONV7x7_S2[pe_idx]);
											nnet::copy_features_g2l<global_feature_config, CONV7x7_S2_local_feature_out_config>(global_feature[conv1_7x7_s2_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_out_CONV7x7_S2[pe_idx],
												global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_start_idx, local_out_feature_c_num,
												global_out_feature_h_start_idx, local_out_feature_h_start_idx, local_out_feature_h_num,
												global_out_feature_w_start_idx, local_out_feature_w_start_idx, local_out_feature_w_num);
										}
									}
									//copy input feature and weight from global BRAM to local BRAM
									//copy input feature
									//std::cout << "clearing buffer for input padding" << std::endl;
									nnet::clear_buffer<CONV7x7_S2_local_feature_in_config>(local_feature_in_CONV7x7_S2[pe_idx]);
									nnet::copy_features_g2l<global_feature_config, CONV7x7_S2_local_feature_in_config>(global_feature[conv1_7x7_s2_allocate_global_in_feature_start_idx + global_in_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_in_CONV7x7_S2[pe_idx],
										global_in_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_in_feature_c_start_idx, local_in_feature_c_num,
										global_in_feature_h_start_idx, local_in_feature_h_start_idx, local_in_feature_h_num,
										global_in_feature_w_start_idx, local_in_feature_w_start_idx, local_in_feature_w_num);
									nnet::copy_weights_g2l<WEIGHT_GLOBAL_7x7_config, CONV7x7_S2_local_weight_config>(global_weight_7x7[conv1_7x7_s2_allocate_global_weight_7x7_start_idx + global_weight_oc_start_idx / OUT_CHANNEL_WEIGHT_GLOBAL_7x7], local_weight_CONV7x7_S2[pe_idx],
										global_weight_oc_start_idx % OUT_CHANNEL_WEIGHT_GLOBAL_7x7, local_weight_oc_num,
										global_weight_ic_start_idx, local_weight_ic_num);
									//call PE and do calculation
									nnet::conv_output_reuse7x7<conv2d_config_CONV7x7_S2>(local_feature_in_CONV7x7_S2[pe_idx], local_weight_CONV7x7_S2[pe_idx][0], local_feature_out_CONV7x7_S2[pe_idx][0]);

									if (i_idx == inner_in_channel - 1) {
										//copy output feature from local BRAM to global BRAM
										if (outer_ic_idx == conv1_7x7_s2_outer_in_channel - 1) {
											nnet::relu_inplace<relu_conv2d_config_CONV7x7_S2>(local_feature_out_CONV7x7_S2[pe_idx]);
										}
										nnet::copy_features_l2g<CONV7x7_S2_local_feature_out_config, global_feature_config>(local_feature_out_CONV7x7_S2[pe_idx], global_feature[conv1_7x7_s2_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL],
											global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_num,
											global_out_feature_h_start_idx, local_out_feature_h_num,
											global_out_feature_w_start_idx, local_out_feature_w_num);
									}
								}
							}
						}
					}
				}//end inner loop
				//copy out feature from BRAM to DRAM
				if (outer_ic_idx == conv1_7x7_s2_outer_in_channel - 1)
				{
					//index and shape of output feature in DRAM
					int DDR_block_out_feature_c_start_idx = DDR_weight_oc_start_idx;
					int DDR_block_out_feature_h_start_idx = DIV_CEIL((DDR_block_in_feature_h_start_idx + conv1_7x7_s2_pad_top), STRIDE_CONV7x7_S2);
					int DDR_block_out_feature_w_start_idx = DIV_CEIL((DDR_block_in_feature_w_start_idx + conv1_7x7_s2_pad_left), STRIDE_CONV7x7_S2);
					if (outer_h_idx == 0) DDR_block_out_feature_h_start_idx = 0; //handle padding
					if (outer_w_idx == 0) DDR_block_out_feature_w_start_idx = 0; // handle padding
					int DDR_block_out_feature_c_num = global_weight_oc_num;
					int DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + conv1_7x7_s2_pad_top + global_block_in_feature_h_num - KERNEL_HEIGHT_CONV7x7_S2) / STRIDE_CONV7x7_S2 + 1 - DDR_block_out_feature_h_start_idx;
					int DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + conv1_7x7_s2_pad_left + global_block_in_feature_w_num - KERNEL_WIDTH_CONV7x7_S2) / STRIDE_CONV7x7_S2 + 1 - DDR_block_out_feature_w_start_idx;

					if (outer_h_idx == conv1_7x7_s2_outer_height - 1) {
						//handle the last iteration of the loop
						DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + conv1_7x7_s2_pad_top + conv1_7x7_s2_pad_bottom + global_block_in_feature_h_num - KERNEL_HEIGHT_CONV7x7_S2) / STRIDE_CONV7x7_S2 + 1 - DDR_block_out_feature_h_start_idx;
					}
					if (outer_w_idx == conv1_7x7_s2_outer_width - 1) {
						//handle the last iteration of the loop
						DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + conv1_7x7_s2_pad_left + conv1_7x7_s2_pad_right + global_block_in_feature_w_num - KERNEL_WIDTH_CONV7x7_S2) / STRIDE_CONV7x7_S2 + 1 - DDR_block_out_feature_w_start_idx;
					}
					//copy output feature from global BRAM to DRAM
					for (int global_out_feature_idx = 0; global_out_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_out_feature_idx++) {
						if (global_out_feature_idx < DIV_CEIL(DDR_block_out_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1)
							nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_conv1_7x7_s2_1_config>(global_feature[conv1_7x7_s2_allocate_global_out_feature_start_idx], conv1_7x7_s2_1,
								DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
								DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
								conv1_7x7_s2_out_feature_DDR_offset + DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
						else
							nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_conv1_7x7_s2_1_config>(global_feature[conv1_7x7_s2_allocate_global_out_feature_start_idx], conv1_7x7_s2_1,
								DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, DDR_block_out_feature_c_num - global_out_feature_idx * CHANNEL_FEATURE_GLOBAL,
								DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
								conv1_7x7_s2_out_feature_DDR_offset +DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
					}
				}//end copy out feature from BRAM to DRAM
			}// end outer_ic loop
		}
	}
}
/////////////top_function/////////////

/////////////template_config/////////////
///conv1_7x7_s2
struct DDR_feature_conv1_7x7_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = conv1_7x7_s2_out_channel;
	static const unsigned height = conv1_7x7_s2_out_height;
	static const unsigned width = conv1_7x7_s2_out_width;
};
/////////////template_config/////////////

/////////////allocate_config/////////////
//conv1_7x7_s2
///configuration
const int conv1_7x7_s2_allocate_global_in_feature_start_idx = 0;
const int conv1_7x7_s2_allocate_global_in_feature_num = 2;
const int conv1_7x7_s2_allocate_global_weight_7x7_start_idx = 0;
const int conv1_7x7_s2_allocate_global_weight_7x7_num = 2;
const int conv1_7x7_s2_allocate_global_out_feature_start_idx = 2;
const int conv1_7x7_s2_allocate_global_out_feature_num = 2;
const int conv1_7x7_s2_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int conv1_7x7_s2_block_overlap_height = KERNEL_HEIGHT_CONV7x7_S2 - 1;
const int conv1_7x7_s2_block_overlap_width = KERNEL_WIDTH_CONV7x7_S2 - 1;
///number of blocks(the dims of the outer loop)
const int conv1_7x7_s2_outer_in_channel = DIV_CEIL(conv1_7x7_s2_in_channel, conv1_7x7_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_7x7 ? conv1_7x7_s2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_7x7);
const int conv1_7x7_s2_outer_height = DIV_CEIL(conv1_7x7_s2_in_height, HEIGHT_FEATURE_GLOBAL- conv1_7x7_s2_block_overlap_height);
const int conv1_7x7_s2_outer_width = DIV_CEIL(conv1_7x7_s2_in_width , WIDTH_FEATURE_GLOBAL - conv1_7x7_s2_block_overlap_width);
///interval between blocks
const int conv1_7x7_s2_block_interval_height = DIV_CEIL(DIV_CEIL(conv1_7x7_s2_in_height, conv1_7x7_s2_outer_height), STRIDE_CONV7x7_S2)*STRIDE_CONV7x7_S2;//the spacing between blocks
const int conv1_7x7_s2_block_interval_width = DIV_CEIL(DIV_CEIL(conv1_7x7_s2_in_width, conv1_7x7_s2_outer_width), STRIDE_CONV7x7_S2)*STRIDE_CONV7x7_S2;
///dim of blocks
const int conv1_7x7_s2_block_in_height = conv1_7x7_s2_block_interval_height+ conv1_7x7_s2_block_overlap_height;
const int conv1_7x7_s2_block_in_width = conv1_7x7_s2_block_interval_height + conv1_7x7_s2_block_overlap_height;
const int conv1_7x7_s2_block_in_channel = MIN(conv1_7x7_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_7x7);
///set parallism
const int conv1_7x7_s2_inner_pe_parallel = NUM_PE_CONV7x7_S2;
///dim of kernels
const int conv1_7x7_s2_block_out_channel = MIN(conv1_7x7_s2_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , conv1_7x7_s2_allocate_global_weight_7x7_num*OUT_CHANNEL_WEIGHT_GLOBAL_7x7);
const int conv1_7x7_s2_outer_out_channel = DIV_CEIL(conv1_7x7_s2_kernel_num, conv1_7x7_s2_block_out_channel);//outer loop
/////////////allocate_config/////////////