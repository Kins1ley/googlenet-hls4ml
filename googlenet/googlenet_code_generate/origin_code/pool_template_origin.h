/////////////top_function/////////////
//pool1_3x3_s2
//outer loop
//copy data and call PE to do calculation
for (int outer_h_idx = 0; outer_h_idx < pool1_3x3_s2_outer_height; outer_h_idx++) {
    for (int outer_w_idx = 0; outer_w_idx < pool1_3x3_s2_outer_width; outer_w_idx++) {
        for (int outer_ic_idx = 0; outer_ic_idx < pool1_3x3_s2_outer_in_channel; outer_ic_idx++) {

            //calculate the index to copy features.
            //index and shape of input feature in DRAM
            int DDR_block_in_feature_h_start_idx = outer_h_idx * pool1_3x3_s2_block_interval_height;
            int DDR_block_in_feature_w_start_idx = outer_w_idx * pool1_3x3_s2_block_interval_width;
            int DDR_block_in_feature_c_start_idx = outer_ic_idx * pool1_3x3_s2_block_in_channel;
            int global_block_in_feature_c_num = pool1_3x3_s2_block_in_channel;
            int global_block_in_feature_h_num = pool1_3x3_s2_block_in_height;
            int global_block_in_feature_w_num = pool1_3x3_s2_block_in_width;

            {
                //handle the last iteration of the loop
                if (outer_h_idx == pool1_3x3_s2_outer_height - 1) {
                    global_block_in_feature_h_num = pool1_3x3_s2_in_height - DDR_block_in_feature_h_start_idx;
                }
                if (outer_w_idx == pool1_3x3_s2_outer_width - 1) {
                    global_block_in_feature_w_num = pool1_3x3_s2_in_width - DDR_block_in_feature_w_start_idx;
                }
                if (outer_ic_idx == pool1_3x3_s2_outer_in_channel - 1) {
                    global_block_in_feature_c_num = pool1_3x3_s2_in_channel - outer_ic_idx * pool1_3x3_s2_block_in_channel;
                }
            }
            //copy input feature from DRAM to global BRAM
            for (int global_in_feature_idx = 0; global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_in_feature_idx++) {
                if (global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1) {
                    nnet::clear_buffer<global_feature_config>(global_feature[pool1_3x3_s2_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_conv1_7x7_s2_1_config, global_feature_config>(conv1_7x7_s2_1, global_feature[pool1_3x3_s2_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                }
                else {
                    nnet::clear_buffer<global_feature_config>(global_feature[pool1_3x3_s2_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_conv1_7x7_s2_1_config, global_feature_config>(conv1_7x7_s2_1, global_feature[pool1_3x3_s2_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, global_block_in_feature_c_num - global_in_feature_idx * CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                }
            }

            //dims of inner loop
            int inner_pad_top = (outer_h_idx == 0 ? pool1_3x3_s2_pad_top : 0);
            int inner_pad_bottom = (outer_h_idx == (pool1_3x3_s2_outer_height - 1) ? pool1_3x3_s2_pad_bottom : 0);
            int inner_pad_left = (outer_w_idx == 0 ? pool1_3x3_s2_pad_left : 0);
            int inner_pad_right = (outer_w_idx == (pool1_3x3_s2_outer_width - 1) ? pool1_3x3_s2_pad_bottom : 0);
            int inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + pool1_3x3_s2_pad_top + global_block_in_feature_h_num + inner_pad_bottom - KERNEL_HEIGHT_MAXPOOL3x3_S2) / (STRIDE_MAXPOOL3x3_S2)+1
                - DIV_CEIL(DDR_block_in_feature_h_start_idx + pool1_3x3_s2_pad_top, STRIDE_MAXPOOL3x3_S2),
                OUT_HEIGHT_MAXPOOL3x3_S2);
            int inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + pool1_3x3_s2_pad_left + global_block_in_feature_w_num + inner_pad_right - KERNEL_WIDTH_MAXPOOL3x3_S2) / (STRIDE_MAXPOOL3x3_S2)+1
                - DIV_CEIL(DDR_block_in_feature_w_start_idx + pool1_3x3_s2_pad_left, STRIDE_MAXPOOL3x3_S2),
                OUT_WIDTH_MAXPOOL3x3_S2);
            if (outer_h_idx == 0) {
                inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + global_block_in_feature_h_num + inner_pad_bottom + pool1_3x3_s2_pad_top - KERNEL_HEIGHT_MAXPOOL3x3_S2) / (STRIDE_MAXPOOL3x3_S2)+1
                    - DIV_CEIL(DDR_block_in_feature_h_start_idx, STRIDE_MAXPOOL3x3_S2),
                    OUT_HEIGHT_MAXPOOL3x3_S2);
            }
            if (outer_w_idx == 0) {
                inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + global_block_in_feature_w_num + inner_pad_right + pool1_3x3_s2_pad_left - KERNEL_WIDTH_MAXPOOL3x3_S2) / (STRIDE_MAXPOOL3x3_S2)+1
                    - DIV_CEIL(DDR_block_in_feature_w_start_idx, STRIDE_MAXPOOL3x3_S2),
                    OUT_WIDTH_MAXPOOL3x3_S2);
            }
            int inner_channel = DIV_CEIL(global_block_in_feature_c_num, pool1_3x3_s2_inner_pe_parallel * N_CHAN_MAXPOOL3x3_S2);
            //do inner loop
            for (int h_idx = 0; h_idx < inner_height; h_idx++) {
                for (int w_idx = 0; w_idx < inner_width; w_idx++) {
                    for (int c_idx = 0; c_idx < inner_channel; c_idx++) {
                        int inner_pe_parallel = pool1_3x3_s2_inner_pe_parallel;
                        if (c_idx == inner_channel - 1) inner_pe_parallel = DIV_CEIL(global_block_in_feature_c_num - c_idx * pool1_3x3_s2_inner_pe_parallel * N_CHAN_MAXPOOL3x3_S2, N_CHAN_MAXPOOL3x3_S2);
#pragma HLS pipeline
                        for (int pe_idx = 0; pe_idx < inner_pe_parallel; pe_idx++) {
#pragma HLS unroll
                            //index of input feature in global BRAM
                            int global_in_feature_c_start_idx = (c_idx * pool1_3x3_s2_inner_pe_parallel + pe_idx) * N_CHAN_MAXPOOL3x3_S2;
                            int global_in_feature_h_start_idx = h_idx * OUT_HEIGHT_MAXPOOL3x3_S2 * STRIDE_MAXPOOL3x3_S2 - inner_pad_top; //
                            int global_in_feature_w_start_idx = w_idx * OUT_WIDTH_MAXPOOL3x3_S2 * STRIDE_MAXPOOL3x3_S2 - inner_pad_left;//

                            //index and shape of input feature in local BRAM
                            int local_in_feature_c_start_idx = 0;
                            int local_in_feature_h_start_idx = 0;
                            int local_in_feature_w_start_idx = 0;
                            int local_in_feature_c_num = N_CHAN_MAXPOOL3x3_S2;
                            int local_in_feature_h_num = IN_HEIGHT_MAXPOOL3x3_S2;
                            int local_in_feature_w_num = IN_WIDTH_MAXPOOL3x3_S2;

                            //index of output feature in global BRAM
                            int global_out_feature_c_start_idx = (c_idx * pool1_3x3_s2_inner_pe_parallel + pe_idx) * N_CHAN_MAXPOOL3x3_S2;
                            int global_out_feature_h_start_idx = h_idx * OUT_HEIGHT_MAXPOOL3x3_S2;
                            int global_out_feature_w_start_idx = w_idx * OUT_WIDTH_MAXPOOL3x3_S2;

                            //index and shape of output feature in local BRAM
                            int local_out_feature_c_start_idx = 0;
                            int local_out_feature_h_start_idx = 0;
                            int local_out_feature_w_start_idx = 0;
                            int local_out_feature_c_num = N_CHAN_MAXPOOL3x3_S2;
                            int local_out_feature_h_num = OUT_HEIGHT_MAXPOOL3x3_S2;
                            int local_out_feature_w_num = OUT_WIDTH_MAXPOOL3x3_S2;


                            if (h_idx == 0) {
                                //handle padding
                                local_in_feature_h_num -= inner_pad_top;
                                local_in_feature_h_start_idx = inner_pad_top;
                                global_in_feature_h_start_idx = 0;
                            }
                            else if (h_idx == inner_height - 1) {
                                //handle the last iteration of the loop and padding
                                local_in_feature_h_num = global_block_in_feature_h_num + inner_pad_top - h_idx * OUT_HEIGHT_MAXPOOL3x3_S2 * STRIDE_MAXPOOL3x3_S2;
                            }

                            if (w_idx == 0) {
                                //handle padding
                                local_in_feature_w_num -= inner_pad_left;
                                local_in_feature_w_start_idx = inner_pad_left;
                                global_in_feature_w_start_idx = 0;
                            }
                            else if (w_idx == inner_width - 1) {
                                //handle the last iteration of the loop and padding
                                local_in_feature_w_num = global_block_in_feature_w_num + inner_pad_left - w_idx * OUT_WIDTH_MAXPOOL3x3_S2 * STRIDE_MAXPOOL3x3_S2;
                            }
                            if ((c_idx == inner_channel - 1) && (pe_idx == inner_pe_parallel - 1)) {
                                //handle the last iteration of the loop
                                local_out_feature_c_num = global_block_in_feature_c_num - (c_idx * pool1_3x3_s2_inner_pe_parallel + pe_idx) * N_CHAN_MAXPOOL3x3_S2;
                                local_in_feature_c_num = global_block_in_feature_c_num - (c_idx * pool1_3x3_s2_inner_pe_parallel + pe_idx) * N_CHAN_MAXPOOL3x3_S2;
                            }
                            // handle the situation that convolution does not start from the first element
                            if (outer_h_idx != 0) {
                                global_in_feature_h_start_idx += (DDR_block_in_feature_h_start_idx + pool1_3x3_s2_pad_top) % STRIDE_MAXPOOL3x3_S2;
                            }
                            if (outer_w_idx != 0) {
                                global_in_feature_w_start_idx += (DDR_block_in_feature_w_start_idx + pool1_3x3_s2_pad_left) % STRIDE_MAXPOOL3x3_S2;
                            }

                            //copy input feature from global BRAM to local BRAM
                            //copy input feature
                            //std::cout << "clearing buffer for input padding" << std::endl;
                            nnet::clear_buffer<MAXPOOL3x3_S2_local_feature_in_config>(local_feature_in_MAXPOOL3x3_S2[pe_idx]);
                            nnet::copy_features_g2l<global_feature_config, MAXPOOL3x3_S2_local_feature_in_config>(global_feature[pool1_3x3_s2_allocate_global_in_feature_start_idx + global_in_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_in_MAXPOOL3x3_S2[pe_idx],
                                global_in_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_in_feature_c_start_idx, local_in_feature_c_num,
                                global_in_feature_h_start_idx, local_in_feature_h_start_idx, local_in_feature_h_num,
                                global_in_feature_w_start_idx, local_in_feature_w_start_idx, local_in_feature_w_num);
                            //call PE and do calculation
                            nnet::pool3x3<pool2d_config_MAX3x3_S2>(local_feature_in_MAXPOOL3x3_S2[pe_idx], local_feature_out_MAXPOOL3x3_S2[pe_idx]);

                            //copy output feature from local BRAM to global BRAM
                            nnet::copy_features_l2g<MAXPOOL3x3_S2_local_feature_out_config, global_feature_config>(local_feature_out_MAXPOOL3x3_S2[pe_idx], global_feature[pool1_3x3_s2_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL],
                                global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_num,
                                global_out_feature_h_start_idx, local_out_feature_h_num,
                                global_out_feature_w_start_idx, local_out_feature_w_num);

                        }
                    }
                }
            }//end inner loop
            //copy out feature from BRAM to DRAM
            {
                //index and shape of output feature in DRAM
                int DDR_block_out_feature_c_start_idx = DDR_block_in_feature_c_start_idx;
                int DDR_block_out_feature_h_start_idx = DIV_CEIL((DDR_block_in_feature_h_start_idx + pool1_3x3_s2_pad_top), STRIDE_MAXPOOL3x3_S2);
                int DDR_block_out_feature_w_start_idx = DIV_CEIL((DDR_block_in_feature_w_start_idx + pool1_3x3_s2_pad_left), STRIDE_MAXPOOL3x3_S2);
                if (outer_h_idx == 0) DDR_block_out_feature_h_start_idx = 0; //handle padding
                if (outer_w_idx == 0) DDR_block_out_feature_w_start_idx = 0; // handle padding
                int DDR_block_out_feature_c_num = global_block_in_feature_c_num;
                int DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + pool1_3x3_s2_pad_top + global_block_in_feature_h_num - KERNEL_HEIGHT_MAXPOOL3x3_S2) / STRIDE_MAXPOOL3x3_S2 + 1 - DDR_block_out_feature_h_start_idx;
                int DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + pool1_3x3_s2_pad_left + global_block_in_feature_w_num - KERNEL_WIDTH_MAXPOOL3x3_S2) / STRIDE_MAXPOOL3x3_S2 + 1 - DDR_block_out_feature_w_start_idx;

                if (outer_h_idx == pool1_3x3_s2_outer_height - 1) {
                    //handle the last iteration of the loop
                    DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + pool1_3x3_s2_pad_top + pool1_3x3_s2_pad_bottom + global_block_in_feature_h_num - KERNEL_HEIGHT_MAXPOOL3x3_S2) / STRIDE_MAXPOOL3x3_S2 + 1 - DDR_block_out_feature_h_start_idx;
                }
                if (outer_w_idx == pool1_3x3_s2_outer_width - 1) {
                    //handle the last iteration of the loop
                    DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + pool1_3x3_s2_pad_left + pool1_3x3_s2_pad_right + global_block_in_feature_w_num - KERNEL_WIDTH_MAXPOOL3x3_S2) / STRIDE_MAXPOOL3x3_S2 + 1 - DDR_block_out_feature_w_start_idx;
                }
                //copy output feature from global BRAM to DRAM
                for (int global_out_feature_idx = 0; global_out_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_out_feature_idx++) {
                    if (global_out_feature_idx < DIV_CEIL(DDR_block_out_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1)
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_pool1_3x3_s2_1_config>(global_feature[pool1_3x3_s2_allocate_global_out_feature_start_idx + global_out_feature_idx], pool1_3x3_s2_1,
                            pool1_3x3_s2_out_channel_DDR_offset + DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                    else
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_pool1_3x3_s2_1_config>(global_feature[pool1_3x3_s2_allocate_global_out_feature_start_idx + global_out_feature_idx], pool1_3x3_s2_1,
                            pool1_3x3_s2_out_channel_DDR_offset + DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, DDR_block_out_feature_c_num - global_out_feature_idx * CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                }
            }//end copy out feature from BRAM to DRAM

        }// end outer_ic loop
    }
}
/////////////top_function/////////////

/////////////template_config/////////////
///pool1_3x3_s2
struct DDR_feature_pool1_3x3_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = pool1_3x3_s2_out_channel;
	static const unsigned height = pool1_3x3_s2_out_height;
	static const unsigned width = pool1_3x3_s2_out_width;
};
/////////////template_config/////////////

/////////////allocate_config/////////////
//pool1_3x3_s2
///configuration
const int pool1_3x3_s2_allocate_global_in_feature_start_idx = 0;
const int pool1_3x3_s2_allocate_global_in_feature_num = 4;
const int pool1_3x3_s2_allocate_global_out_feature_start_idx = 4;
const int pool1_3x3_s2_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int pool1_3x3_s2_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S2 - 1;
const int pool1_3x3_s2_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S2 - 1;
///number of blocks(the dims of the outer loop)
const int pool1_3x3_s2_outer_in_channel = DIV_CEIL(pool1_3x3_s2_in_channel, pool1_3x3_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int pool1_3x3_s2_outer_height = DIV_CEIL(pool1_3x3_s2_in_height, HEIGHT_FEATURE_GLOBAL - pool1_3x3_s2_block_overlap_height);
const int pool1_3x3_s2_outer_width = DIV_CEIL(pool1_3x3_s2_in_width, WIDTH_FEATURE_GLOBAL - pool1_3x3_s2_block_overlap_width);
///interval between blocks
const int pool1_3x3_s2_block_interval_height = DIV_CEIL(DIV_CEIL(pool1_3x3_s2_in_height, pool1_3x3_s2_outer_height), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;//the spacing between blocks
const int pool1_3x3_s2_block_interval_width = DIV_CEIL(DIV_CEIL(pool1_3x3_s2_in_width, pool1_3x3_s2_outer_width), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;
///dim of blocks
const int pool1_3x3_s2_block_in_height = pool1_3x3_s2_block_interval_height + pool1_3x3_s2_block_overlap_height;
const int pool1_3x3_s2_block_in_width = pool1_3x3_s2_block_interval_height + pool1_3x3_s2_block_overlap_height;
const int pool1_3x3_s2_block_in_channel = pool1_3x3_s2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool1_3x3_s2_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S2;
///dim of kernels
const int pool1_3x3_s2_block_out_channel = pool1_3x3_s2_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
/////////////allocate_config/////////////