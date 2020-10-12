/////////////top_function/////////////
//pool1_norm1
//outer loop
//copy data and call PE to do calculation
for (int outer_h_idx = 0; outer_h_idx < pool1_norm1_outer_height; outer_h_idx++) {
    for (int outer_w_idx = 0; outer_w_idx < pool1_norm1_outer_width; outer_w_idx++) {
        for (int outer_ic_idx = 0; outer_ic_idx < pool1_norm1_outer_in_channel; outer_ic_idx++) {

            //calculate the index to copy features.
            //index and shape of input feature in DRAM
            int DDR_block_in_feature_h_start_idx = outer_h_idx * pool1_norm1_block_interval_height;
            int DDR_block_in_feature_w_start_idx = outer_w_idx * pool1_norm1_block_interval_width;
            int DDR_block_in_feature_c_start_idx = outer_ic_idx * pool1_norm1_block_interval_channel;
            int global_block_in_feature_c_num = pool1_norm1_block_in_channel;
            int global_block_in_feature_h_num = pool1_norm1_block_in_height;
            int global_block_in_feature_w_num = pool1_norm1_block_in_width;

            {
                //handle the last iteration of the loop
                if (outer_h_idx == pool1_norm1_outer_height - 1) {
                    global_block_in_feature_h_num = pool1_norm1_in_height - DDR_block_in_feature_h_start_idx;
                }
                if (outer_w_idx == pool1_norm1_outer_width - 1) {
                    global_block_in_feature_w_num = pool1_norm1_in_width - DDR_block_in_feature_w_start_idx;
                }
                if (outer_ic_idx == pool1_norm1_outer_in_channel - 1) {
                    global_block_in_feature_c_num = pool1_norm1_in_channel - outer_ic_idx * pool1_norm1_block_interval_channel;
                }
            }
            //copy input feature from DRAM to global BRAM
            for (int global_in_feature_idx = 0; global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_in_feature_idx++) {
                if (global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1) {
                    nnet::clear_buffer<global_feature_config>(global_feature[pool1_norm1_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_pool1_3x3_s2_1_config, global_feature_config>(pool1_3x3_s2_1, global_feature[pool1_norm1_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                }
                else {
                    nnet::clear_buffer<global_feature_config>(global_feature[pool1_norm1_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_pool1_3x3_s2_1_config, global_feature_config>(pool1_3x3_s2_1, global_feature[pool1_norm1_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, global_block_in_feature_c_num - global_in_feature_idx * CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                }
            }

            //dims of inner loop
            int inner_height = DIV_CEIL(global_block_in_feature_h_num, OUT_HEIGHT_LRN);
            int inner_width = DIV_CEIL(global_block_in_feature_w_num, OUT_WIDTH_LRN);
            int padding_channel_left = (outer_ic_idx == 0) ? pool1_norm1_depth_radius : 0;
            int padding_channel_right = (outer_ic_idx == (pool1_norm1_outer_in_channel - 1)) ? pool1_norm1_depth_radius : 0;
            int inner_channel = DIV_CEIL(global_block_in_feature_c_num - 2 * pool1_norm1_depth_radius + padding_channel_left + padding_channel_right, pool1_norm1_inner_pe_parallel * (N_CHAN_LRN - 2 * pool1_norm1_depth_radius));
            //do inner loop
            for (int h_idx = 0; h_idx < inner_height; h_idx++) {
                for (int w_idx = 0; w_idx < inner_width; w_idx++) {
                    for (int c_idx = 0; c_idx < inner_channel; c_idx++) {
                        int inner_pe_parallel = pool1_norm1_inner_pe_parallel;
                        if (c_idx == inner_channel - 1) inner_pe_parallel = DIV_CEIL(global_block_in_feature_c_num - c_idx * pool1_norm1_inner_pe_parallel * (N_CHAN_LRN - 2 * pool1_norm1_depth_radius), (N_CHAN_LRN - 2 * pool1_norm1_depth_radius));
#pragma HLS pipeline
                        for (int pe_idx = 0; pe_idx < inner_pe_parallel; pe_idx++) {
#pragma HLS unroll
                            //index of input feature in global BRAM
                            int global_in_feature_c_start_idx = (c_idx * pool1_norm1_inner_pe_parallel + pe_idx) * (N_CHAN_LRN - 2 * pool1_norm1_depth_radius) - padding_channel_left;
                            int global_in_feature_h_start_idx = h_idx * OUT_HEIGHT_LRN; //
                            int global_in_feature_w_start_idx = w_idx * OUT_WIDTH_LRN;//

                            //index and shape of input feature in local BRAM
                            int local_in_feature_c_start_idx = 0;
                            int local_in_feature_h_start_idx = 0;
                            int local_in_feature_w_start_idx = 0;
                            int local_in_feature_c_num = N_CHAN_LRN;
                            int local_in_feature_h_num = IN_HEIGHT_LRN;
                            int local_in_feature_w_num = IN_WIDTH_LRN;

                            //index of output feature in global BRAM
                            int global_out_feature_c_start_idx = (c_idx * pool1_norm1_inner_pe_parallel + pe_idx) * (N_CHAN_LRN - 2 * pool1_norm1_depth_radius);
                            int global_out_feature_h_start_idx = h_idx * OUT_HEIGHT_LRN;
                            int global_out_feature_w_start_idx = w_idx * OUT_WIDTH_LRN;

                            //index and shape of output feature in local BRAM
                            int local_out_feature_c_start_idx = 0;
                            int local_out_feature_h_start_idx = 0;
                            int local_out_feature_w_start_idx = 0;
                            int local_out_feature_c_num = N_CHAN_LRN - 2 * pool1_norm1_depth_radius;
                            int local_out_feature_h_num = OUT_HEIGHT_LRN;
                            int local_out_feature_w_num = OUT_WIDTH_LRN;


                            if (h_idx == inner_height - 1) {
                                //handle the last iteration of the loop and padding
                                local_in_feature_h_num = global_block_in_feature_h_num - h_idx * OUT_HEIGHT_LRN;
                            }

                            if (w_idx == inner_width - 1) {
                                //handle the last iteration of the loop and padding
                                local_in_feature_w_num = global_block_in_feature_w_num - w_idx * OUT_WIDTH_LRN;
                            }

                            if ((c_idx == inner_channel - 1) && (pe_idx == inner_pe_parallel - 1)) {
                                //handle the last iteration of the loop
                                local_in_feature_c_num = global_block_in_feature_c_num + padding_channel_left - (c_idx * pool1_norm1_inner_pe_parallel + pe_idx) * (N_CHAN_LRN - 2 * pool1_norm1_depth_radius);
                                local_out_feature_c_num = local_in_feature_c_num + padding_channel_right - 2 * pool1_norm1_depth_radius;
                            }
                            if ((c_idx * pool1_norm1_inner_pe_parallel + pe_idx) == 0) {
                                global_in_feature_c_start_idx = 0;
                                local_in_feature_c_start_idx = padding_channel_left;
                                local_in_feature_c_num -= padding_channel_left;
                            }
                            //copy input feature from global BRAM to local BRAM
                            //copy input feature
                            //std::cout << "clearing buffer for input padding" << std::endl;
                            nnet::clear_buffer<LRN_local_feature_in_config>(local_feature_in_LRN[pe_idx]);
                            nnet::copy_features_g2l<global_feature_config, LRN_local_feature_in_config>(global_feature[pool1_norm1_allocate_global_in_feature_start_idx + global_in_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_in_LRN[pe_idx],
                                global_in_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_in_feature_c_start_idx, local_in_feature_c_num,
                                global_in_feature_h_start_idx, local_in_feature_h_start_idx, local_in_feature_h_num,
                                global_in_feature_w_start_idx, local_in_feature_w_start_idx, local_in_feature_w_num);
                            //call PE and do calculation
                            nnet::LRN<LRN_config>(local_feature_in_LRN[pe_idx], local_feature_out_LRN[pe_idx], bias, alpha, beta, pool1_norm1_depth_radius);

                            //copy output feature from local BRAM to global BRAM
                            nnet::copy_features_l2g<LRN_local_feature_out_config, global_feature_config>(local_feature_out_LRN[pe_idx], global_feature[pool1_norm1_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL],
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
                int DDR_block_out_feature_c_start_idx = DDR_block_in_feature_c_start_idx + pool1_norm1_depth_radius - padding_channel_left;
                int DDR_block_out_feature_h_start_idx = DDR_block_in_feature_h_start_idx;
                int DDR_block_out_feature_w_start_idx = DDR_block_in_feature_w_start_idx;
                if (outer_ic_idx == 0) DDR_block_out_feature_c_start_idx = 0; //handle padding
                int DDR_block_out_feature_c_num = global_block_in_feature_c_num - 2 * pool1_norm1_depth_radius + padding_channel_left + padding_channel_right;
                int DDR_block_out_feature_h_num = global_block_in_feature_h_num;
                int DDR_block_out_feature_w_num = global_block_in_feature_w_num;

                if (outer_ic_idx == pool1_norm1_outer_in_channel - 1) {
                    //handle the last iteration of the loop
                    DDR_block_out_feature_c_num = pool1_norm1_in_channel - DDR_block_out_feature_c_start_idx;
                }

                //copy output feature from global BRAM to DRAM
                for (int global_out_feature_idx = 0; global_out_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_out_feature_idx++) {
                    if (global_out_feature_idx < DIV_CEIL(DDR_block_out_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1)
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_pool1_norm1_1_config>(global_feature[pool1_norm1_allocate_global_out_feature_start_idx + global_out_feature_idx], pool1_norm1_1,
                            DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                    else
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_pool1_norm1_1_config>(global_feature[pool1_norm1_allocate_global_out_feature_start_idx + global_out_feature_idx], pool1_norm1_1,
                            DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, DDR_block_out_feature_c_num - global_out_feature_idx * CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                }
            }//end copy out feature from BRAM to DRAM

        }// end outer_ic loop
    }
}
/////////////top_function/////////////

/////////////template_config/////////////
///pool1_norm1
struct DDR_feature_pool1_norm1_1_config : nnet::Feature_Memory {
    typedef FIX_INT20 feature_type;
    static const unsigned channel = pool1_norm1_out_channel;
    static const unsigned height = pool1_norm1_out_height;
    static const unsigned width = pool1_norm1_out_width;
};
/////////////template_config/////////////

/////////////allocate_config/////////////
//pool1_norm1_1
///configuration
const int pool1_norm1_allocate_global_in_feature_start_idx = 0;
const int pool1_norm1_allocate_global_in_feature_num = 1;//multi global BRAM is not supportted yet
const int pool1_norm1_allocate_global_out_feature_start_idx = 1;
const int pool1_norm1_allocate_global_out_feature_num = 1;
///overlapped features between blocks
const int pool1_norm1_block_overlap_channel = 2 * pool1_norm1_depth_radius;
///number of blocks(the dims of the outer loop)
const int pool1_norm1_outer_in_channel = DIV_CEIL(pool1_norm1_in_channel, pool1_norm1_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * pool1_norm1_depth_radius));
const int pool1_norm1_outer_height = DIV_CEIL(pool1_norm1_in_height, HEIGHT_FEATURE_GLOBAL);
const int pool1_norm1_outer_width = DIV_CEIL(pool1_norm1_in_width, WIDTH_FEATURE_GLOBAL);
///interval between blocks
const int pool1_norm1_block_interval_channel = pool1_norm1_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * pool1_norm1_depth_radius);//the spacing between blocks
const int pool1_norm1_block_interval_height = HEIGHT_FEATURE_GLOBAL;
const int pool1_norm1_block_interval_width = WIDTH_FEATURE_GLOBAL;
///dim of blocks
const int pool1_norm1_block_in_height = pool1_norm1_block_interval_height;
const int pool1_norm1_block_in_width = pool1_norm1_block_interval_height;
const int pool1_norm1_block_in_channel = pool1_norm1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool1_norm1_inner_pe_parallel = NUM_PE_LRN;//prallelism not supported
/////////////allocate_config/////////////