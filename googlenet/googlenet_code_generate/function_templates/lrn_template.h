/////////////top_function/////////////
//{lrn_layer_name}
//outer loop
//copy data and call PE to do calculation
for (int outer_h_idx = 0; outer_h_idx < {lrn_layer_name}_outer_height; outer_h_idx++) {left_bracket}
    for (int outer_w_idx = 0; outer_w_idx < {lrn_layer_name}_outer_width; outer_w_idx++) {left_bracket}
        for (int outer_ic_idx = 0; outer_ic_idx < {lrn_layer_name}_outer_in_channel; outer_ic_idx++) {left_bracket}

            //calculate the index to copy features.
            //index and shape of input feature in DRAM
            int DDR_block_in_feature_h_start_idx = outer_h_idx * {lrn_layer_name}_block_interval_height;
            int DDR_block_in_feature_w_start_idx = outer_w_idx * {lrn_layer_name}_block_interval_width;
            int DDR_block_in_feature_c_start_idx = outer_ic_idx * {lrn_layer_name}_block_interval_channel;
            int global_block_in_feature_c_num = {lrn_layer_name}_block_in_channel;
            int global_block_in_feature_h_num = {lrn_layer_name}_block_in_height;
            int global_block_in_feature_w_num = {lrn_layer_name}_block_in_width;

            {left_bracket}
                //handle the last iteration of the loop
                if (outer_h_idx == {lrn_layer_name}_outer_height - 1) {left_bracket}
                    global_block_in_feature_h_num = {lrn_layer_name}_in_height - DDR_block_in_feature_h_start_idx;
                {right_bracket}
                if (outer_w_idx == {lrn_layer_name}_outer_width - 1) {left_bracket}
                    global_block_in_feature_w_num = {lrn_layer_name}_in_width - DDR_block_in_feature_w_start_idx;
                {right_bracket}
                if (outer_ic_idx == {lrn_layer_name}_outer_in_channel - 1) {left_bracket}
                    global_block_in_feature_c_num = {lrn_layer_name}_in_channel - outer_ic_idx * {lrn_layer_name}_block_interval_channel;
                {right_bracket}
            {right_bracket}
            //copy input feature from DRAM to global BRAM
            for (int global_in_feature_idx = 0; global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_in_feature_idx++) {left_bracket}
                if (global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1) {left_bracket}
                    nnet::clear_buffer<global_feature_config>(global_feature[{lrn_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_{layer_input_name}_config, global_feature_config>({DDR_in_feature}_config, global_feature[{lrn_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                {right_bracket}
                else {left_bracket}
                    nnet::clear_buffer<global_feature_config>(global_feature[{lrn_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx]);
                    nnet::copy_features_DDR2BRAM<DDR_feature_{layer_input_name}_config, global_feature_config>({DDR_in_feature}_config, global_feature[{lrn_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx],
                        DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, global_block_in_feature_c_num - global_in_feature_idx * CHANNEL_FEATURE_GLOBAL,
                        DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
                        DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
                {right_bracket}
            {right_bracket}

            //dims of inner loop
            int inner_height = DIV_CEIL(global_block_in_feature_h_num, OUT_HEIGHT_{lrn_layer_type});
            int inner_width = DIV_CEIL(global_block_in_feature_w_num, OUT_WIDTH_{lrn_layer_type});
            int padding_channel_left = (outer_ic_idx == 0) ? {lrn_layer_name}_deepth_radius : 0;
            int padding_channel_right = (outer_ic_idx == ({lrn_layer_name}_outer_in_channel - 1)) ? {lrn_layer_name}_deepth_radius : 0;
            int inner_channel = DIV_CEIL(global_block_in_feature_c_num - 2 * {lrn_layer_name}_deepth_radius + padding_channel_left + padding_channel_right, {lrn_layer_name}_inner_pe_parallel * (N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius));
            //do inner loop
            for (int h_idx = 0; h_idx < inner_height; h_idx++) {left_bracket}
                for (int w_idx = 0; w_idx < inner_width; w_idx++) {left_bracket}
                    for (int c_idx = 0; c_idx < inner_channel; c_idx++) {left_bracket}
                        int inner_pe_parallel = {lrn_layer_name}_inner_pe_parallel;
                        if (c_idx == inner_channel - 1) inner_pe_parallel = DIV_CEIL(global_block_in_feature_c_num - c_idx * {lrn_layer_name}_inner_pe_parallel * (N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius), (N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius));
#pragma HLS pipeline
                        for (int pe_idx = 0; pe_idx < inner_pe_parallel; pe_idx++) {left_bracket}
#pragma HLS unroll
                            //index of input feature in global BRAM
                            int global_in_feature_c_start_idx = (c_idx * {lrn_layer_name}_inner_pe_parallel + pe_idx) * (N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius) - padding_channel_left;
                            int global_in_feature_h_start_idx = h_idx * OUT_HEIGHT_{lrn_layer_type}; //
                            int global_in_feature_w_start_idx = w_idx * OUT_WIDTH_{lrn_layer_type};//

                            //index and shape of input feature in local BRAM
                            int local_in_feature_c_start_idx = 0;
                            int local_in_feature_h_start_idx = 0;
                            int local_in_feature_w_start_idx = 0;
                            int local_in_feature_c_num = N_CHAN_{lrn_layer_type};
                            int local_in_feature_h_num = IN_HEIGHT_{lrn_layer_type};
                            int local_in_feature_w_num = IN_WIDTH_{lrn_layer_type};

                            //index of output feature in global BRAM
                            int global_out_feature_c_start_idx = (c_idx * {lrn_layer_name}_inner_pe_parallel + pe_idx) * (N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius);
                            int global_out_feature_h_start_idx = h_idx * OUT_HEIGHT_{lrn_layer_type};
                            int global_out_feature_w_start_idx = w_idx * OUT_WIDTH_{lrn_layer_type};

                            //index and shape of output feature in local BRAM
                            int local_out_feature_c_start_idx = 0;
                            int local_out_feature_h_start_idx = 0;
                            int local_out_feature_w_start_idx = 0;
                            int local_out_feature_c_num = N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius;
                            int local_out_feature_h_num = OUT_HEIGHT_{lrn_layer_type};
                            int local_out_feature_w_num = OUT_WIDTH_{lrn_layer_type};


                            if (h_idx == inner_height - 1) {left_bracket}
                                //handle the last iteration of the loop and padding
                                local_in_feature_h_num = global_block_in_feature_h_num - h_idx * OUT_HEIGHT_{lrn_layer_type};
                            {right_bracket}

                            if (w_idx == inner_width - 1) {left_bracket}
                                //handle the last iteration of the loop and padding
                                local_in_feature_w_num = global_block_in_feature_w_num - w_idx * OUT_WIDTH_{lrn_layer_type};
                            {right_bracket}

                            if ((c_idx == inner_channel - 1) && (pe_idx == inner_pe_parallel - 1)) {left_bracket}
                                //handle the last iteration of the loop
                                local_in_feature_c_num = global_block_in_feature_c_num + padding_channel_left - (c_idx * {lrn_layer_name}_inner_pe_parallel + pe_idx) * (N_CHAN_{lrn_layer_type} - 2 * {lrn_layer_name}_deepth_radius);
                                local_out_feature_c_num = local_in_feature_c_num + padding_channel_right - 2 * {lrn_layer_name}_deepth_radius;
                            {right_bracket}
                            if ((c_idx * {lrn_layer_name}_inner_pe_parallel + pe_idx) == 0) {left_bracket}
                                global_in_feature_c_start_idx = 0;
                                local_in_feature_c_start_idx = padding_channel_left;
                                local_in_feature_c_num -= padding_channel_left;
                            {right_bracket}
                            //copy input feature from global BRAM to local BRAM
                            //copy input feature
                            //std::cout << "clearing buffer for input padding" << std::endl;
                            nnet::clear_buffer<{lrn_layer_type}_local_feature_in_config>(local_feature_in_{lrn_layer_type}[pe_idx]);
                            nnet::copy_features_g2l<global_feature_config, {lrn_layer_type}_local_feature_in_config>(global_feature[{lrn_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_in_{lrn_layer_type}[pe_idx],
                                global_in_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_in_feature_c_start_idx, local_in_feature_c_num,
                                global_in_feature_h_start_idx, local_in_feature_h_start_idx, local_in_feature_h_num,
                                global_in_feature_w_start_idx, local_in_feature_w_start_idx, local_in_feature_w_num);
                            //call PE and do calculation
                            nnet::{lrn_layer_type}<{lrn_layer_type}_config>(local_feature_in_{lrn_layer_type}[pe_idx], local_feature_out_{lrn_layer_type}[pe_idx], bias, alpha, beta, {lrn_layer_name}_deepth_radius);

                            //copy output feature from local BRAM to global BRAM
                            nnet::copy_features_l2g<{lrn_layer_type}_local_feature_out_config, global_feature_config>(local_feature_out_{lrn_layer_type}[pe_idx], global_feature[{lrn_layer_name}_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL],
                                global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_num,
                                global_out_feature_h_start_idx, local_out_feature_h_num,
                                global_out_feature_w_start_idx, local_out_feature_w_num);

                        {right_bracket}
                    {right_bracket}
                {right_bracket}
            {right_bracket}//end inner loop
            //copy out feature from BRAM to DRAM
            {left_bracket}
                //index and shape of output feature in DRAM
                int DDR_block_out_feature_c_start_idx = DDR_block_in_feature_c_start_idx + {lrn_layer_name}_deepth_radius - padding_channel_left;
                int DDR_block_out_feature_h_start_idx = DDR_block_in_feature_h_start_idx;
                int DDR_block_out_feature_w_start_idx = DDR_block_in_feature_w_start_idx;
                if (outer_ic_idx == 0) DDR_block_out_feature_c_start_idx = 0; //handle padding
                int DDR_block_out_feature_c_num = global_block_in_feature_c_num - 2 * {lrn_layer_name}_deepth_radius + padding_channel_left + padding_channel_right;
                int DDR_block_out_feature_h_num = global_block_in_feature_h_num;
                int DDR_block_out_feature_w_num = global_block_in_feature_w_num;

                if (outer_ic_idx == {lrn_layer_name}_outer_in_channel - 1) {left_bracket}
                    //handle the last iteration of the loop
                    DDR_block_out_feature_c_num = {lrn_layer_name}_in_channel - DDR_block_out_feature_c_start_idx;
                {right_bracket}

                //copy output feature from global BRAM to DRAM
                for (int global_out_feature_idx = 0; global_out_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_out_feature_idx++) {left_bracket}
                    if (global_out_feature_idx < DIV_CEIL(DDR_block_out_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1)
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_{layer_output_name}_config>(global_feature[{lrn_layer_name}_allocate_global_out_feature_start_idx + global_out_feature_idx], {DDR_out_feature}_config,
                            DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                    else
                        nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_{layer_output_name}_config>(global_feature[{lrn_layer_name}_allocate_global_out_feature_start_idx + global_out_feature_idx], {DDR_out_feature}_config,
                            DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, DDR_block_out_feature_c_num - global_out_feature_idx * CHANNEL_FEATURE_GLOBAL,
                            DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
                            DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
                {right_bracket}
            {right_bracket}//end copy out feature from BRAM to DRAM

        {right_bracket}// end outer_ic loop
    {right_bracket}
{right_bracket}
/////////////top_function/////////////

/////////////template_config/////////////
///{lrn_layer_name}
struct DDR_feature_{layer_output_name}_config : nnet::Feature_Memory {left_bracket}
    typedef FIX_INT20 feature_type;
    static const unsigned channel = {lrn_layer_name}_out_channel;
    static const unsigned height = {lrn_layer_name}_out_height;
    static const unsigned width = {lrn_layer_name}_out_width;
{right_bracket};
/////////////template_config/////////////

/////////////allocate_config/////////////
//{DDR_out_feature}_config
///configuration
const int {lrn_layer_name}_allocate_global_in_feature_start_idx = 0;
const int {lrn_layer_name}_allocate_global_in_feature_num = 1;//multi global BRAM is not supportted yet
const int {lrn_layer_name}_allocate_global_out_feature_start_idx = 1;
const int {lrn_layer_name}_allocate_global_out_feature_num = 1;
///overlapped features between blocks
const int {lrn_layer_name}_block_overlap_channel = 2 * {lrn_layer_name}_depth_radius;
///number of blocks(the dims of the outer loop)
const int {lrn_layer_name}_outer_in_channel = DIV_CEIL({lrn_layer_name}_in_channel, {lrn_layer_name}_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * {lrn_layer_name}_depth_radius));
const int {lrn_layer_name}_outer_height = DIV_CEIL({lrn_layer_name}_in_height, HEIGHT_FEATURE_GLOBAL);
const int {lrn_layer_name}_outer_width = DIV_CEIL({lrn_layer_name}_in_width, WIDTH_FEATURE_GLOBAL);
///interval between blocks
const int {lrn_layer_name}_block_interval_channel = {lrn_layer_name}_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * {lrn_layer_name}_depth_radius);//the spacing between blocks
const int {lrn_layer_name}_block_interval_height = HEIGHT_FEATURE_GLOBAL;
const int {lrn_layer_name}_block_interval_width = WIDTH_FEATURE_GLOBAL;
///dim of blocks
const int {lrn_layer_name}_block_in_height = {lrn_layer_name}_block_interval_height;
const int {lrn_layer_name}_block_in_width = {lrn_layer_name}_block_interval_height;
const int {lrn_layer_name}_block_in_channel = {lrn_layer_name}_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int {lrn_layer_name}_inner_pe_parallel = NUM_PE_{lrn_layer_type};//prallelism not supported
/////////////allocate_config/////////////