/////////////top_function/////////////
//{layer_name}
//outer loop
//copy data and call PE to do calculation
for (int outer_h_idx = 0; outer_h_idx < {layer_name}_outer_height; outer_h_idx++) {{
    for (int outer_w_idx = 0; outer_w_idx < {layer_name}_outer_width; outer_w_idx++) {{
        for (int outer_ic_idx = 0; outer_ic_idx < {layer_name}_outer_in_channel; outer_ic_idx++) {{

            //calculate the index to copy features.
            //index and shape of input feature in DRAM
            int DDR_block_in_feature_h_start_idx = outer_h_idx * {layer_name}_block_interval_height;
            int DDR_block_in_feature_w_start_idx = outer_w_idx * {layer_name}_block_interval_width;
            int DDR_block_in_feature_c_start_idx = outer_ic_idx * {layer_name}_block_in_channel;
            int global_block_in_feature_c_num = {layer_name}_block_in_channel;
            int global_block_in_feature_h_num = {layer_name}_block_in_height;
            int global_block_in_feature_w_num = {layer_name}_block_in_width;

            {{
                //handle the last iteration of the loop
                if (outer_h_idx == {layer_name}_outer_height - 1) {{
                    global_block_in_feature_h_num = {layer_name}_in_height - DDR_block_in_feature_h_start_idx;
                }}
                if (outer_w_idx == {layer_name}_outer_width - 1) {{
                    global_block_in_feature_w_num = {layer_name}_in_width - DDR_block_in_feature_w_start_idx;
                }}
                if (outer_ic_idx == {layer_name}_outer_in_channel - 1) {{
                    global_block_in_feature_c_num = {layer_name}_in_channel - outer_ic_idx * {layer_name}_block_in_channel;
                }}
            }}
            //copy input feature from DRAM to global BRAM
            for (int global_in_feature_idx = 0; global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_in_feature_idx++) {{
                if (global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1) {{
                    nnet::clear_buffer<global_feature_config>(global_feature[{layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_{layer_input_name}_config, global_feature_config>({DDR_in_feature}, global_feature[{layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                }}
                else {{
                    nnet::clear_buffer<global_feature_config>(global_feature[{layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_{layer_input_name}_config, global_feature_config>({DDR_in_feature}, global_feature[{layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, global_block_in_feature_c_num - global_in_feature_idx * CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                }}
            }}

            //dims of inner loop
            int inner_pad_top = (outer_h_idx == 0 ? {layer_name}_pad_top : 0);
            int inner_pad_bottom = (outer_h_idx == ({layer_name}_outer_height - 1) ? {layer_name}_pad_bottom : 0);
            int inner_pad_left = (outer_w_idx == 0 ? {layer_name}_pad_left : 0);
            int inner_pad_right = (outer_w_idx == ({layer_name}_outer_width - 1) ? {layer_name}_pad_bottom : 0);
            int inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + {layer_name}_pad_top + global_block_in_feature_h_num + inner_pad_bottom - KERNEL_HEIGHT_{layer_type}) / (STRIDE_{layer_type})+1
                - DIV_CEIL(DDR_block_in_feature_h_start_idx + {layer_name}_pad_top, STRIDE_{layer_type}),
                OUT_HEIGHT_{layer_type});
            int inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + {layer_name}_pad_left + global_block_in_feature_w_num + inner_pad_right - KERNEL_WIDTH_{layer_type}) / (STRIDE_{layer_type})+1
                - DIV_CEIL(DDR_block_in_feature_w_start_idx + {layer_name}_pad_left, STRIDE_{layer_type}),
                OUT_WIDTH_{layer_type});
            if (outer_h_idx == 0) {{
                inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + global_block_in_feature_h_num + inner_pad_bottom + {layer_name}_pad_top - KERNEL_HEIGHT_{layer_type}) / (STRIDE_{layer_type})+1
                    - DIV_CEIL(DDR_block_in_feature_h_start_idx, STRIDE_{layer_type}),
                    OUT_HEIGHT_{layer_type});
            }}
            if (outer_w_idx == 0) {{
                inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + global_block_in_feature_w_num + inner_pad_right + {layer_name}_pad_left - KERNEL_WIDTH_{layer_type}) / (STRIDE_{layer_type})+1
                    - DIV_CEIL(DDR_block_in_feature_w_start_idx, STRIDE_{layer_type}),
                    OUT_WIDTH_{layer_type});
            }}
            int inner_channel = DIV_CEIL(global_block_in_feature_c_num, {layer_name}_inner_pe_parallel * N_CHAN_{layer_type});
            //do inner loop
            for (int h_idx = 0; h_idx < inner_height; h_idx++) {{
                for (int w_idx = 0; w_idx < inner_width; w_idx++) {{
                    for (int c_idx = 0; c_idx < inner_channel; c_idx++) {{
                        int inner_pe_parallel = {layer_name}_inner_pe_parallel;
                        if (c_idx == inner_channel - 1) inner_pe_parallel = DIV_CEIL(global_block_in_feature_c_num - c_idx * {layer_name}_inner_pe_parallel * N_CHAN_{layer_type}, N_CHAN_{layer_type});
#pragma HLS pipeline
                        for (int pe_idx = 0; pe_idx < inner_pe_parallel; pe_idx++) {{
#pragma HLS unroll
                            //index of input feature in global BRAM
                            int global_in_feature_c_start_idx = (c_idx * {layer_name}_inner_pe_parallel + pe_idx) * N_CHAN_{layer_type};
                            int global_in_feature_h_start_idx = h_idx * OUT_HEIGHT_{layer_type} * STRIDE_{layer_type} - inner_pad_top; //
                            int global_in_feature_w_start_idx = w_idx * OUT_WIDTH_{layer_type} * STRIDE_{layer_type} - inner_pad_left;//

                            //index and shape of input feature in local BRAM
                            int local_in_feature_c_start_idx = 0;
                            int local_in_feature_h_start_idx = 0;
                            int local_in_feature_w_start_idx = 0;
                            int local_in_feature_c_num = N_CHAN_{layer_type};
                            int local_in_feature_h_num = IN_HEIGHT_{layer_type};
                            int local_in_feature_w_num = IN_WIDTH_{layer_type};

                            //index of output feature in global BRAM
                            int global_out_feature_c_start_idx = (c_idx * {layer_name}_inner_pe_parallel + pe_idx) * N_CHAN_{layer_type};
                            int global_out_feature_h_start_idx = h_idx * OUT_HEIGHT_{layer_type};
                            int global_out_feature_w_start_idx = w_idx * OUT_WIDTH_{layer_type};

                            //index and shape of output feature in local BRAM
                            int local_out_feature_c_start_idx = 0;
                            int local_out_feature_h_start_idx = 0;
                            int local_out_feature_w_start_idx = 0;
                            int local_out_feature_c_num = N_CHAN_{layer_type};
                            int local_out_feature_h_num = OUT_HEIGHT_{layer_type};
                            int local_out_feature_w_num = OUT_WIDTH_{layer_type};


                            if (h_idx == 0) {{
                                //handle padding
                                local_in_feature_h_num -= inner_pad_top;
                                local_in_feature_h_start_idx = inner_pad_top;
                                global_in_feature_h_start_idx = 0;
                            }}
                            else if (h_idx == inner_height - 1) {{
                                //handle the last iteration of the loop and padding
                                local_in_feature_h_num = global_block_in_feature_h_num + inner_pad_top - h_idx * OUT_HEIGHT_{layer_type} * STRIDE_{layer_type};
                            }}

                            if (w_idx == 0) {{
                                //handle padding
                                local_in_feature_w_num -= inner_pad_left;
                                local_in_feature_w_start_idx = inner_pad_left;
                                global_in_feature_w_start_idx = 0;
                            }}
                            else if (w_idx == inner_width - 1) {{
                                //handle the last iteration of the loop and padding
                                local_in_feature_w_num = global_block_in_feature_w_num + inner_pad_left - w_idx * OUT_WIDTH_{layer_type} * STRIDE_{layer_type};
                            }}
                            if ((c_idx == inner_channel - 1) && (pe_idx == inner_pe_parallel - 1)) {{
                                //handle the last iteration of the loop
                                local_out_feature_c_num = global_block_in_feature_c_num - (c_idx * {layer_name}_inner_pe_parallel + pe_idx) * N_CHAN_{layer_type};
                                local_in_feature_c_num = global_block_in_feature_c_num - (c_idx * {layer_name}_inner_pe_parallel + pe_idx) * N_CHAN_{layer_type};
                            }}
                            // handle the situation that convolution does not start from the first element
                            if (outer_h_idx != 0) {{
                                global_in_feature_h_start_idx += (DDR_block_in_feature_h_start_idx + {layer_name}_pad_top) % STRIDE_{layer_type};
                            }}
                            if (outer_w_idx != 0) {{
                                global_in_feature_w_start_idx += (DDR_block_in_feature_w_start_idx + {layer_name}_pad_left) % STRIDE_{layer_type};
                            }}

                            //copy input feature from global BRAM to local BRAM
                            //copy input feature
                            //std::cout << "clearing buffer for input padding" << std::endl;
                            nnet::clear_buffer<{layer_type}_local_feature_in_config>(local_feature_in_{layer_type}[pe_idx]);
                            nnet::copy_features_g2l<global_feature_config, {layer_type}_local_feature_in_config>(global_feature[{layer_name}_allocate_global_in_feature_start_idx + global_in_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_in_{layer_type}[pe_idx],
                                global_in_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_in_feature_c_start_idx, local_in_feature_c_num,
                                global_in_feature_h_start_idx, local_in_feature_h_start_idx, local_in_feature_h_num,
                                global_in_feature_w_start_idx, local_in_feature_w_start_idx, local_in_feature_w_num);
                            //call PE and do calculation
                            nnet::{pe_name}<pool2d_config_{layer_type}>(local_feature_in_{layer_type}[pe_idx], local_feature_out_{layer_type}[pe_idx]);

                            //copy output feature from local BRAM to global BRAM
                            nnet::copy_features_l2g<{layer_type}_local_feature_out_config, global_feature_config>(local_feature_out_{layer_type}[pe_idx], global_feature[{layer_name}_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL],
                                global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_num,
                                global_out_feature_h_start_idx, local_out_feature_h_num,
                                global_out_feature_w_start_idx, local_out_feature_w_num);

                        }}
                    }}
                }}
            }}//end inner loop
            //copy out feature from BRAM to DRAM
            {{
                //index and shape of output feature in DRAM
                int DDR_block_out_feature_c_start_idx = DDR_block_in_feature_c_start_idx;
                int DDR_block_out_feature_h_start_idx = DIV_CEIL((DDR_block_in_feature_h_start_idx + {layer_name}_pad_top), STRIDE_{layer_type});
                int DDR_block_out_feature_w_start_idx = DIV_CEIL((DDR_block_in_feature_w_start_idx + {layer_name}_pad_left), STRIDE_{layer_type});
                if (outer_h_idx == 0) DDR_block_out_feature_h_start_idx = 0; //handle padding
                if (outer_w_idx == 0) DDR_block_out_feature_w_start_idx = 0; // handle padding
                int DDR_block_out_feature_c_num = global_block_in_feature_c_num;
                int DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + {layer_name}_pad_top + global_block_in_feature_h_num - KERNEL_HEIGHT_{layer_type}) / STRIDE_{layer_type} + 1 - DDR_block_out_feature_h_start_idx;
                int DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + {layer_name}_pad_left + global_block_in_feature_w_num - KERNEL_WIDTH_{layer_type}) / STRIDE_{layer_type} + 1 - DDR_block_out_feature_w_start_idx;

                if (outer_h_idx == {layer_name}_outer_height - 1) {{
                    //handle the last iteration of the loop
                    DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + {layer_name}_pad_top + {layer_name}_pad_bottom + global_block_in_feature_h_num - KERNEL_HEIGHT_{layer_type}) / STRIDE_{layer_type} + 1 - DDR_block_out_feature_h_start_idx;
                }}
                if (outer_w_idx == {layer_name}_outer_width - 1) {{
                    //handle the last iteration of the loop
                    DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + {layer_name}_pad_left + {layer_name}_pad_right + global_block_in_feature_w_num - KERNEL_WIDTH_{layer_type}) / STRIDE_{layer_type} + 1 - DDR_block_out_feature_w_start_idx;
                }}
                //copy output feature from global BRAM to DRAM
                for (int global_out_feature_idx = 0; global_out_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_out_feature_idx++) {{
                    if (global_out_feature_idx < DIV_CEIL(DDR_block_out_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1)
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_{layer_output_name}_config>(global_feature[{layer_name}_allocate_global_out_feature_start_idx + global_out_feature_idx], {DDR_out_feature},
                            DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            {layer_name}_out_feature_DDR_offset + DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                    else
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_{layer_output_name}_config>(global_feature[{layer_name}_allocate_global_out_feature_start_idx + global_out_feature_idx], {DDR_out_feature},
                            DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, DDR_block_out_feature_c_num - global_out_feature_idx * CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            {layer_name}_out_feature_DDR_offset + DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                }}
            }}//end copy out feature from BRAM to DRAM

        }}// end outer_ic loop
    }}
}}
/////////////top_function/////////////

/////////////template_config/////////////
///{layer_name}
struct DDR_feature_{layer_output_name}_config : nnet::Feature_Memory {{
	typedef FIX_INT20 feature_type;
	static const unsigned channel = {layer_name}_out_channel;
	static const unsigned height = {layer_name}_out_height;
	static const unsigned width = {layer_name}_out_width;
}};
/////////////template_config/////////////

/////////////allocate_config/////////////
//{layer_name}
///configuration
const int {layer_name}_allocate_global_in_feature_start_idx = 0;
const int {layer_name}_allocate_global_in_feature_num = 4;
const int {layer_name}_allocate_global_out_feature_start_idx = 4;
const int {layer_name}_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int {layer_name}_block_overlap_height = KERNEL_HEIGHT_{layer_type} - 1;
const int {layer_name}_block_overlap_width = KERNEL_WIDTH_{layer_type} - 1;
///number of blocks(the dims of the outer loop)
const int {layer_name}_outer_in_channel = DIV_CEIL({layer_name}_in_channel, {layer_name}_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int {layer_name}_outer_height = DIV_CEIL({layer_name}_in_height, HEIGHT_FEATURE_GLOBAL - {layer_name}_block_overlap_height);
const int {layer_name}_outer_width = DIV_CEIL({layer_name}_in_width, WIDTH_FEATURE_GLOBAL - {layer_name}_block_overlap_width);
///interval between blocks
const int {layer_name}_block_interval_height = DIV_CEIL(DIV_CEIL({layer_name}_in_height, {layer_name}_outer_height), STRIDE_{layer_type})*STRIDE_{layer_type};//the spacing between blocks
const int {layer_name}_block_interval_width = DIV_CEIL(DIV_CEIL({layer_name}_in_width, {layer_name}_outer_width), STRIDE_{layer_type})*STRIDE_{layer_type};
///dim of blocks
const int {layer_name}_block_in_height = {layer_name}_block_interval_height + {layer_name}_block_overlap_height;
const int {layer_name}_block_in_width = {layer_name}_block_interval_height + {layer_name}_block_overlap_height;
const int {layer_name}_block_in_channel = {layer_name}_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int {layer_name}_inner_pe_parallel = NUM_PE_{layer_type};
///dim of kernels
const int {layer_name}_block_out_channel = {layer_name}_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
/////////////allocate_config/////////////