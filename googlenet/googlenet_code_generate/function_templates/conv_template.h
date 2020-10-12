/////////////top_function/////////////
//{conv_layer_name}
//outer loop
//copy data and call PE to do calculation
for (int outer_h_idx = 0; outer_h_idx < {conv_layer_name}_outer_height; outer_h_idx++) {left_bracket}
	for (int outer_w_idx = 0; outer_w_idx < {conv_layer_name}_outer_width; outer_w_idx++) {left_bracket}
		for (int outer_oc_idx = 0; outer_oc_idx < {conv_layer_name}_outer_out_channel; outer_oc_idx++) {left_bracket}
			for (int outer_ic_idx = 0; outer_ic_idx < {conv_layer_name}_outer_in_channel; outer_ic_idx++) {left_bracket}
				//std::cout << "outer loop" << outer_h_idx << outer_w_idx << outer_oc_idx << outer_ic_idx << std::endl;

				//calculate the index to copy features and weights.
				//index and shape of input feature in DRAM
				int DDR_block_in_feature_h_start_idx = outer_h_idx * {conv_layer_name}_block_interval_height;
				int DDR_block_in_feature_w_start_idx = outer_w_idx * {conv_layer_name}_block_interval_width;
				int DDR_block_in_feature_c_start_idx = outer_ic_idx * {conv_layer_name}_block_in_channel;
				int global_block_in_feature_c_num = {conv_layer_name}_block_in_channel;
				int global_block_in_feature_h_num = {conv_layer_name}_block_in_height;
				int global_block_in_feature_w_num = {conv_layer_name}_block_in_width;

				//index and shape of weight in DRAM
				int DDR_weight_ic_start_idx = outer_ic_idx * {conv_layer_name}_block_in_channel;
				int DDR_weight_oc_start_idx = outer_oc_idx * {conv_layer_name}_block_out_channel;
				int global_weight_ic_num = {conv_layer_name}_block_in_channel;
				int global_weight_oc_num = {conv_layer_name}_block_out_channel;

				{left_bracket}
					//handle the last iteration of the loop
					if (outer_h_idx == {conv_layer_name}_outer_height - 1) {left_bracket}
						global_block_in_feature_h_num = {conv_layer_name}_in_height - DDR_block_in_feature_h_start_idx;
					{right_bracket}
					if (outer_w_idx == {conv_layer_name}_outer_width - 1) {left_bracket}
						global_block_in_feature_w_num = {conv_layer_name}_in_width - DDR_block_in_feature_w_start_idx;
					{right_bracket}
					if (outer_oc_idx == {conv_layer_name}_outer_out_channel - 1) {left_bracket}
						global_weight_oc_num = {conv_layer_name}_out_channel - outer_oc_idx * {conv_layer_name}_block_out_channel;
					{right_bracket}
					if (outer_ic_idx == {conv_layer_name}_outer_in_channel - 1) {left_bracket}
						global_block_in_feature_c_num = {conv_layer_name}_in_channel - outer_ic_idx * {conv_layer_name}_block_in_channel;
						global_weight_ic_num = {conv_layer_name}_in_channel - outer_ic_idx * {conv_layer_name}_block_in_channel;
					{right_bracket}
				{right_bracket}
				//copy input feature and weight from DRAM to global BRAM
				for (int global_in_feature_idx = 0; global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_in_feature_idx++) {left_bracket}
					if (global_in_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1) {left_bracket}
						nnet::clear_buffer<global_feature_config>(global_feature[{conv_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx]);
						nnet::copy_features_DDR2BRAM<DDR_feature_{conv_layer_input}_config, global_feature_config>({conv_layer_input}, global_feature[{conv_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx],
							DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
							DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
							DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
					{right_bracket}
					else {left_bracket}
						nnet::clear_buffer<global_feature_config>(global_feature[{conv_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx]);
						nnet::copy_features_DDR2BRAM<DDR_feature_{conv_layer_input}_config, global_feature_config>({conv_layer_input}, global_feature[{conv_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_idx],
							DDR_block_in_feature_c_start_idx + global_in_feature_idx * CHANNEL_FEATURE_GLOBAL, global_block_in_feature_c_num - global_in_feature_idx * CHANNEL_FEATURE_GLOBAL,
							DDR_block_in_feature_h_start_idx, global_block_in_feature_h_num,
							DDR_block_in_feature_w_start_idx, global_block_in_feature_w_num);
					{right_bracket}
				{right_bracket}
				for (int global_weight_idx = 0; global_weight_idx < DIV_CEIL(global_weight_oc_num, OUT_CHANNEL_{global_weight_type}); global_weight_idx++) {left_bracket}
					if (global_weight_idx < DIV_CEIL(global_weight_oc_num, OUT_CHANNEL_{global_weight_type}) - 1)
						nnet::copy_weights_DDR2BRAM<{DDR_weight_type}_config, {global_weight_type}_config>({DDR_weight_type}, {global_weight_name}[{conv_layer_name}_allocate_{global_weight_name}_start_idx + global_weight_idx],
							{conv_layer_name}_kernel_channel_DDR_offset + DDR_weight_oc_start_idx + global_weight_idx * OUT_CHANNEL_{global_weight_type}, OUT_CHANNEL_{global_weight_type},
							DDR_weight_ic_start_idx, global_weight_ic_num);
					else
						nnet::copy_weights_DDR2BRAM<{DDR_weight_type}_config, {global_weight_type}_config>({DDR_weight_type}, {global_weight_name}[{conv_layer_name}_allocate_{global_weight_name}_start_idx + global_weight_idx],
							{conv_layer_name}_kernel_channel_DDR_offset + DDR_weight_oc_start_idx + global_weight_idx * OUT_CHANNEL_{global_weight_type}, global_weight_oc_num - global_weight_idx * OUT_CHANNEL_{global_weight_type},
							DDR_weight_ic_start_idx, global_weight_ic_num);
				{right_bracket}

				//std::cout << "(block)processing feature \n start_idx " << DDR_block_in_feature_c_start_idx<<","<< DDR_block_in_feature_h_start_idx << "," << DDR_block_in_feature_w_start_idx<<std::endl;
				//std::cout << "number " << global_block_in_feature_c_num << "," << global_block_in_feature_h_num << "," << global_block_in_feature_w_num <<std::endl;

				//dims of inner loop
				int inner_pad_top = (outer_h_idx == 0 ? {conv_layer_name}_pad_top : 0);
				int inner_pad_bottom = (outer_h_idx == ({conv_layer_name}_outer_height - 1) ? {conv_layer_name}_pad_bottom : 0);
				int inner_pad_left = (outer_w_idx == 0 ? {conv_layer_name}_pad_left : 0);
				int inner_pad_right = (outer_w_idx == ({conv_layer_name}_outer_width - 1) ? {conv_layer_name}_pad_bottom : 0);
				int inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + {conv_layer_name}_pad_top + global_block_in_feature_h_num + inner_pad_bottom - KERNEL_HEIGHT_{conv_layer_type}) / (STRIDE_{conv_layer_type})+1
					- DIV_CEIL(DDR_block_in_feature_h_start_idx + {conv_layer_name}_pad_top, STRIDE_{conv_layer_type}),
					OUT_HEIGHT_{conv_layer_type});
				int inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + {conv_layer_name}_pad_left + global_block_in_feature_w_num + inner_pad_right - KERNEL_WIDTH_{conv_layer_type}) / (STRIDE_{conv_layer_type})+1
					- DIV_CEIL(DDR_block_in_feature_w_start_idx + {conv_layer_name}_pad_left, STRIDE_{conv_layer_type}),
					OUT_WIDTH_{conv_layer_type});
				if (outer_h_idx == 0) {left_bracket}
					inner_height = DIV_CEIL((DDR_block_in_feature_h_start_idx + global_block_in_feature_h_num + inner_pad_bottom + {conv_layer_name}_pad_top - KERNEL_HEIGHT_{conv_layer_type}) / (STRIDE_{conv_layer_type})+1
						- DIV_CEIL(DDR_block_in_feature_h_start_idx, STRIDE_{conv_layer_type}),
						OUT_HEIGHT_{conv_layer_type});
				{right_bracket}
				if (outer_w_idx == 0) {left_bracket}
					inner_width = DIV_CEIL((DDR_block_in_feature_w_start_idx + global_block_in_feature_w_num + inner_pad_right + {conv_layer_name}_pad_left - KERNEL_WIDTH_{conv_layer_type}) / (STRIDE_{conv_layer_type})+1
						- DIV_CEIL(DDR_block_in_feature_w_start_idx, STRIDE_{conv_layer_type}),
						OUT_WIDTH_{conv_layer_type});
				{right_bracket}
				int inner_out_channel = DIV_CEIL(global_weight_oc_num, {conv_layer_name}_inner_pe_parallel * OUT_CHAN_{conv_layer_type});
				int inner_in_channel = DIV_CEIL(global_weight_ic_num, IN_CHAN_{conv_layer_type});
				//do inner loop
				for (int h_idx = 0; h_idx < inner_height; h_idx++) {left_bracket}
					for (int w_idx = 0; w_idx < inner_width; w_idx++) {left_bracket}
						for (int o_idx = 0; o_idx < inner_out_channel; o_idx++) {left_bracket}
							int inner_pe_parallel = {conv_layer_name}_inner_pe_parallel;
							if (o_idx == inner_out_channel - 1) inner_pe_parallel = global_weight_oc_num - o_idx * {conv_layer_name}_inner_pe_parallel;
							for (int i_idx = 0; i_idx < inner_in_channel; i_idx++) {left_bracket}
#pragma HLS pipeline
								for (int pe_idx = 0; pe_idx < inner_pe_parallel; pe_idx++) {left_bracket}
#pragma HLS unroll
									//index and shape of weight in global BRAM
									int global_weight_ic_start_idx = i_idx * IN_CHAN_{conv_layer_type};
									int global_weight_oc_start_idx = (o_idx * {conv_layer_name}_inner_pe_parallel + pe_idx) * OUT_CHAN_{conv_layer_type};
									int local_weight_ic_num = IN_CHAN_{conv_layer_type};
									int local_weight_oc_num = OUT_CHAN_{conv_layer_type};

									//index of input feature in global BRAM
									int global_in_feature_c_start_idx = i_idx * IN_CHAN_{conv_layer_type};
									int global_in_feature_h_start_idx = h_idx * OUT_HEIGHT_{conv_layer_type} * STRIDE_{conv_layer_type} - inner_pad_top; //
									int global_in_feature_w_start_idx = w_idx * OUT_WIDTH_{conv_layer_type} * STRIDE_{conv_layer_type} - inner_pad_left;//

									//index and shape of input feature in local BRAM
									int local_in_feature_c_start_idx = 0;
									int local_in_feature_h_start_idx = 0;
									int local_in_feature_w_start_idx = 0;
									int local_in_feature_c_num = IN_CHAN_{conv_layer_type};
									int local_in_feature_h_num = IN_HEIGHT_{conv_layer_type};
									int local_in_feature_w_num = IN_WIDTH_{conv_layer_type};

									//index of output feature in global BRAM
									int global_out_feature_c_start_idx = global_weight_oc_start_idx;
									int global_out_feature_h_start_idx = h_idx * OUT_HEIGHT_{conv_layer_type};
									int global_out_feature_w_start_idx = w_idx * OUT_WIDTH_{conv_layer_type};

									//index and shape of output feature in local BRAM
									int local_out_feature_c_start_idx = 0;
									int local_out_feature_h_start_idx = 0;
									int local_out_feature_w_start_idx = 0;
									int local_out_feature_c_num = local_weight_oc_num;
									int local_out_feature_h_num = OUT_HEIGHT_{conv_layer_type};
									int local_out_feature_w_num = OUT_WIDTH_{conv_layer_type};


									if (h_idx == 0) {left_bracket}
										//handle padding
										local_in_feature_h_num -= inner_pad_top;
										local_in_feature_h_start_idx = inner_pad_top;
										global_in_feature_h_start_idx = 0;
									{right_bracket}
									else if (h_idx == inner_height - 1) {left_bracket}
										//handle the last iteration of the loop and padding
										local_in_feature_h_num = global_block_in_feature_h_num + inner_pad_top - h_idx * OUT_HEIGHT_{conv_layer_type} * STRIDE_{conv_layer_type};
									{right_bracket}

									if (w_idx == 0) {left_bracket}
										//handle padding
										local_in_feature_w_num -= inner_pad_left;
										local_in_feature_w_start_idx = inner_pad_left;
										global_in_feature_w_start_idx = 0;
									{right_bracket}
									else if (w_idx == inner_width - 1) {left_bracket}
										//handle the last iteration of the loop and padding
										local_in_feature_w_num = global_block_in_feature_w_num + inner_pad_left - w_idx * OUT_WIDTH_{conv_layer_type} * STRIDE_{conv_layer_type};
									{right_bracket}
									if (o_idx == inner_out_channel - 1) {left_bracket}
										//handle the last iteration of the loop
										local_weight_oc_num = global_weight_oc_num - o_idx * OUT_CHAN_{conv_layer_type} * {conv_layer_name}_inner_pe_parallel - OUT_CHAN_{conv_layer_type} * pe_idx;
									{right_bracket}
									if (i_idx == inner_in_channel - 1) {left_bracket}
										//handle the last iteration of the loop
										local_in_feature_c_num = global_block_in_feature_c_num - i_idx * IN_CHAN_{conv_layer_type};
										local_weight_ic_num = global_block_in_feature_c_num - i_idx * IN_CHAN_{conv_layer_type};
									{right_bracket}
									// handle the situation that convolution does not start from the first element
									if (outer_h_idx != 0) {left_bracket}
										global_in_feature_h_start_idx += (DDR_block_in_feature_h_start_idx + {conv_layer_name}_pad_top) % STRIDE_{conv_layer_type};
									{right_bracket}
									if (outer_w_idx != 0) {left_bracket}
										global_in_feature_w_start_idx += (DDR_block_in_feature_w_start_idx + {conv_layer_name}_pad_left) % STRIDE_{conv_layer_type};
									{right_bracket}

									if (i_idx == 0) {left_bracket}
										if (outer_ic_idx == 0) {left_bracket}
											//set bias
											//std::cout << "clearing buffer for bias" << std::endl;
											nnet::clear_buffer<{conv_layer_type}_local_feature_out_config>(local_feature_out_{conv_layer_type}[pe_idx]);
											nnet::set_bias<{conv_layer_type}_set_bias_config>(local_feature_out_{conv_layer_type}[pe_idx], DDR_bias + {conv_layer_name}_bias_DDR_offset + ({conv_layer_name}_allocate_bias_start_idx + pe_idx + o_idx * {conv_layer_name}_inner_pe_parallel + outer_oc_idx * {conv_layer_name}_block_out_channel));
											std::cout << "";
										{right_bracket}
										else {left_bracket}
											//restore partial sum
											//std::cout << "clearing buffer for restoring partial sum" << std::endl;
											nnet::clear_buffer<{conv_layer_type}_local_feature_out_config>(local_feature_out_{conv_layer_type}[pe_idx]);
											nnet::copy_features_g2l<global_feature_config, {conv_layer_type}_local_feature_out_config>(global_feature[{conv_layer_name}_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_out_{conv_layer_type}[pe_idx],
												global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_start_idx, local_out_feature_c_num,
												global_out_feature_h_start_idx, local_out_feature_h_start_idx, local_out_feature_h_num,
												global_out_feature_w_start_idx, local_out_feature_w_start_idx, local_out_feature_w_num);
										{right_bracket}
									{right_bracket}
									//copy input feature and weight from global BRAM to local BRAM
									//copy input feature
									//std::cout << "clearing buffer for input padding" << std::endl;
									nnet::clear_buffer<{conv_layer_type}_local_feature_in_config>(local_feature_in_{conv_layer_type}[pe_idx]);
									nnet::copy_features_g2l<global_feature_config, {conv_layer_type}_local_feature_in_config>(global_feature[{conv_layer_name}_allocate_global_in_feature_start_idx + global_in_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL], local_feature_in_{conv_layer_type}[pe_idx],
										global_in_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_in_feature_c_start_idx, local_in_feature_c_num,
										global_in_feature_h_start_idx, local_in_feature_h_start_idx, local_in_feature_h_num,
										global_in_feature_w_start_idx, local_in_feature_w_start_idx, local_in_feature_w_num);
									nnet::copy_weights_g2l<{global_weight_type}_config, {conv_layer_type}_local_weight_config>({global_weight_name}[{conv_layer_name}_allocate_{global_weight_name}_start_idx + global_weight_oc_start_idx / OUT_CHANNEL_{global_weight_type}], local_weight_{conv_layer_type}[pe_idx],
										global_weight_oc_start_idx % OUT_CHANNEL_{global_weight_type}, local_weight_oc_num,
										global_weight_ic_start_idx, local_weight_ic_num);
									//call PE and do calculation
									nnet::{conv_pe_name}<conv2d_config_{conv_layer_type}>(local_feature_in_{conv_layer_type}[pe_idx], local_weight_{conv_layer_type}[pe_idx][0], local_feature_out_{conv_layer_type}[pe_idx][0]);

									if (i_idx == inner_in_channel - 1) {left_bracket}
										//copy output feature from local BRAM to global BRAM
										if (outer_ic_idx == {conv_layer_name}_outer_in_channel - 1) {left_bracket}
											nnet::relu_inplace<relu_conv2d_config_{conv_layer_type}>(local_feature_out_{conv_layer_type}[pe_idx]);
										{right_bracket}
										nnet::copy_features_l2g<{conv_layer_type}_local_feature_out_config, global_feature_config>(local_feature_out_{conv_layer_type}[pe_idx], global_feature[{conv_layer_name}_allocate_global_out_feature_start_idx + global_out_feature_c_start_idx / CHANNEL_FEATURE_GLOBAL],
											global_out_feature_c_start_idx % CHANNEL_FEATURE_GLOBAL, local_out_feature_c_num,
											global_out_feature_h_start_idx, local_out_feature_h_num,
											global_out_feature_w_start_idx, local_out_feature_w_num);
									{right_bracket}
								{right_bracket}
							{right_bracket}
						{right_bracket}
					{right_bracket}
				{right_bracket}//end inner loop
				//copy out feature from BRAM to DRAM
				if (outer_ic_idx == {conv_layer_name}_outer_in_channel - 1)
				{left_bracket}
					//index and shape of output feature in DRAM
					int DDR_block_out_feature_c_start_idx = DDR_weight_oc_start_idx;
					int DDR_block_out_feature_h_start_idx = DIV_CEIL((DDR_block_in_feature_h_start_idx + {conv_layer_name}_pad_top), STRIDE_{conv_layer_type});
					int DDR_block_out_feature_w_start_idx = DIV_CEIL((DDR_block_in_feature_w_start_idx + {conv_layer_name}_pad_left), STRIDE_{conv_layer_type});
					if (outer_h_idx == 0) DDR_block_out_feature_h_start_idx = 0; //handle padding
					if (outer_w_idx == 0) DDR_block_out_feature_w_start_idx = 0; // handle padding
					int DDR_block_out_feature_c_num = global_weight_oc_num;
					int DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + {conv_layer_name}_pad_top + global_block_in_feature_h_num - KERNEL_HEIGHT_{conv_layer_type}) / STRIDE_{conv_layer_type} + 1 - DDR_block_out_feature_h_start_idx;
					int DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + {conv_layer_name}_pad_left + global_block_in_feature_w_num - KERNEL_WIDTH_{conv_layer_type}) / STRIDE_{conv_layer_type} + 1 - DDR_block_out_feature_w_start_idx;

					if (outer_h_idx == {conv_layer_name}_outer_height - 1) {left_bracket}
						//handle the last iteration of the loop
						DDR_block_out_feature_h_num = (DDR_block_in_feature_h_start_idx + {conv_layer_name}_pad_top + {conv_layer_name}_pad_bottom + global_block_in_feature_h_num - KERNEL_HEIGHT_{conv_layer_type}) / STRIDE_{conv_layer_type} + 1 - DDR_block_out_feature_h_start_idx;
					{right_bracket}
					if (outer_w_idx == {conv_layer_name}_outer_width - 1) {left_bracket}
						//handle the last iteration of the loop
						DDR_block_out_feature_w_num = (DDR_block_in_feature_w_start_idx + {conv_layer_name}_pad_left + {conv_layer_name}_pad_right + global_block_in_feature_w_num - KERNEL_WIDTH_{conv_layer_type}) / STRIDE_{conv_layer_type} + 1 - DDR_block_out_feature_w_start_idx;
					{right_bracket}
					//copy output feature from global BRAM to DRAM
					for (int global_out_feature_idx = 0; global_out_feature_idx < DIV_CEIL(global_block_in_feature_c_num, CHANNEL_FEATURE_GLOBAL); global_out_feature_idx++) {left_bracket}
						if (global_out_feature_idx < DIV_CEIL(DDR_block_out_feature_c_num, CHANNEL_FEATURE_GLOBAL) - 1)
							nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_{conv_layer_output}_config>(global_feature[{conv_layer_name}_allocate_global_out_feature_start_idx], {conv_layer_output},
								{conv_layer_name}_out_channel_DDR_offset + DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, CHANNEL_FEATURE_GLOBAL,
								DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
								DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
						else
							nnet::copy_features_BRAM2DDR<global_feature_config, DDR_feature_{conv_layer_output}_config>(global_feature[{conv_layer_name}_allocate_global_out_feature_start_idx], {conv_layer_output},
								{conv_layer_name}_out_channel_DDR_offset + DDR_block_out_feature_c_start_idx + global_out_feature_idx * CHANNEL_FEATURE_GLOBAL, DDR_block_out_feature_c_num - global_out_feature_idx * CHANNEL_FEATURE_GLOBAL,
								DDR_block_out_feature_h_start_idx, DDR_block_out_feature_h_num,
								DDR_block_out_feature_w_start_idx, DDR_block_out_feature_w_num);
					{right_bracket}
				{right_bracket}//end copy out feature from BRAM to DRAM
			{right_bracket}// end outer_ic loop
		{right_bracket}
	{right_bracket}
{right_bracket}
/////////////top_function/////////////

/////////////template_config/////////////
///{conv_layer_name}
struct DDR_feature_{conv_layer_output}_config : nnet::Feature_Memory {left_bracket}
	typedef FIX_INT20 feature_type;
	static const unsigned channel = {conv_layer_name}_out_channel;
	static const unsigned height = {conv_layer_name}_out_height;
	static const unsigned width = {conv_layer_name}_out_width;
{right_bracket};
/////////////template_config/////////////

/////////////allocate_config/////////////
//{conv_layer_name}
///configuration
const int {conv_layer_name}_allocate_global_in_feature_start_idx = 0;
const int {conv_layer_name}_allocate_global_in_feature_num = 2;
const int {conv_layer_name}_allocate_{global_weight_name}_start_idx = 0;
const int {conv_layer_name}_allocate_{global_weight_name}_num = 2;
const int {conv_layer_name}_allocate_global_out_feature_start_idx = 2;
const int {conv_layer_name}_allocate_global_out_feature_num = 2;
const int {conv_layer_name}_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int {conv_layer_name}_block_overlap_height = KERNEL_HEIGHT_{conv_layer_type} - 1;
const int {conv_layer_name}_block_overlap_width = KERNEL_WIDTH_{conv_layer_type} - 1;
///number of blocks(the dims of the outer loop)
const int {conv_layer_name}_outer_in_channel = DIV_CEIL({conv_layer_name}_in_channel, {conv_layer_name}_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_{global_weight_type} ? {conv_layer_name}_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_{global_weight_type});
const int {conv_layer_name}_outer_height = DIV_CEIL({conv_layer_name}_in_height, HEIGHT_FEATURE_GLOBAL- {conv_layer_name}_block_overlap_height);
const int {conv_layer_name}_outer_width = DIV_CEIL({conv_layer_name}_in_width , WIDTH_FEATURE_GLOBAL - {conv_layer_name}_block_overlap_width);
///interval between blocks
const int {conv_layer_name}_block_interval_height = DIV_CEIL(DIV_CEIL({conv_layer_name}_in_height, {conv_layer_name}_outer_height), STRIDE_{conv_layer_type})*STRIDE_{conv_layer_type};//the spacing between blocks
const int {conv_layer_name}_block_interval_width = DIV_CEIL(DIV_CEIL({conv_layer_name}_in_width, {conv_layer_name}_outer_width), STRIDE_{conv_layer_type})*STRIDE_{conv_layer_type};
///dim of blocks
const int {conv_layer_name}_block_in_height = {conv_layer_name}_block_interval_height+ {conv_layer_name}_block_overlap_height;
const int {conv_layer_name}_block_in_width = {conv_layer_name}_block_interval_height + {conv_layer_name}_block_overlap_height;
const int {conv_layer_name}_block_in_channel = MIN({conv_layer_name}_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_{global_weight_type});
///set parallism
const int {conv_layer_name}_inner_pe_parallel = NUM_PE_{conv_layer_type};
///dim of kernels
const int {conv_layer_name}_block_out_channel = MIN({conv_layer_name}_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , {conv_layer_name}_allocate_{global_weight_name}_num*OUT_CHANNEL_{global_weight_type});
const int {conv_layer_name}_outer_out_channel = DIV_CEIL({conv_layer_name}_kernel_num, {conv_layer_name}_block_out_channel);//outer loop
/////////////allocate_config/////////////