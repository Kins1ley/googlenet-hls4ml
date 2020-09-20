#ifndef SCHEDULE_CONFIG_H_
#define SCHEDULE_CONFIG_H_

#define CALC_BLOCK_H_W(shape,num_block,kernel_shape,stride) ((DIV_CEIL(DIV_CEIL(shape,num_block),stride)-1)*(stride)+kernel_shape)

#include "header.h"
//conv1_7x7_s2
///number of blocks(the dims of the outer loop)
const int conv1_7x7_s2_outer_height = DIV_CEIL(conv1_7x7_s2_in_height, HEIGHT_FEATURE_GLOBAL - STRIDE_CONV7x7_S2 + 1);
const int conv1_7x7_s2_outer_width = DIV_CEIL(conv1_7x7_s2_in_width, WIDTH_FEATURE_GLOBAL - STRIDE_CONV7x7_S2 + 1);
const int conv1_7x7_s2_outer_out_channel = DIV_CEIL(conv1_7x7_s2_kernel_num, OUT_CHANNEL_WEIGHT_GLOBAL_7x7);
const int conv1_7x7_s2_outer_in_channel = DIV_CEIL(conv1_7x7_s2_in_channel, CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_7x7 ? CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_7x7);
///dim of blocks
const int conv1_7x7_s2_block_in_height = DIV_CEIL(conv1_7x7_s2_in_height, conv1_7x7_s2_outer_height) - 1 + KERNEL_HEIGHT_CONV7x7_S2;
const int conv1_7x7_s2_block_in_width = DIV_CEIL(conv1_7x7_s2_in_width, conv1_7x7_s2_outer_width) - 1 + KERNEL_WIDTH_CONV7x7_S2;
const int conv1_7x7_s2_block_in_channel = CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_7x7 ? CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_7x7;
///interval between blocks
const int conv1_7x7_s2_block_interval_height = DIV_CEIL(conv1_7x7_s2_in_height, conv1_7x7_s2_outer_height);//the spacing between blocks
const int conv1_7x7_s2_block_interval_width = DIV_CEIL(conv1_7x7_s2_in_width, conv1_7x7_s2_outer_width);
///set parallism
const int conv1_7x7_s2_inner_pe_parallel = NUM_PE_CONV7x7_S2;


//pool1_3x3_s2
const int pool1_3x3_s2_out_height_per_block = ((HEIGHT_FEATURE_GLOBAL - pool1_3x3_s2_kernel_height) / pool1_3x3_s2_stride + 1);
const int pool1_3x3_s2_out_width_per_block = ((WIDTH_FEATURE_GLOBAL - pool1_3x3_s2_kernel_width) / pool1_3x3_s2_stride + 1);

const int pool1_3x3_s2_outer_height = (pool1_3x3_s2_out_height / pool1_3x3_s2_out_height_per_block) +
((pool1_3x3_s2_out_height % pool1_3x3_s2_out_height_per_block) == 0 ? 0 : 1);
const int pool1_3x3_s2_outer_width = (pool1_3x3_s2_out_width / pool1_3x3_s2_out_width_per_block) +
((pool1_3x3_s2_out_width % pool1_3x3_s2_out_width_per_block) == 0 ? 0 : 1);
const int pool1_3x3_s2_outer_channel = (pool1_3x3_s2_in_channel / CHANNEL_FEATURE_GLOBAL) +
((pool1_3x3_s2_in_channel % CHANNEL_FEATURE_GLOBAL) == 0 ? 0 : 1);


const int pool1_3x3_s2_inner_height = (pool1_3x3_s2_out_height_per_block / OUT_HEIGHT_CONV7x7_S2) +
((pool1_3x3_s2_out_height_per_block % OUT_HEIGHT_MAXPOOL3x3_S2) == 0 ? 0 : 1);
const int pool1_3x3_s2_inner_width = (pool1_3x3_s2_out_width_per_block / OUT_WIDTH_MAXPOOL3x3_S2) +
((pool1_3x3_s2_out_width_per_block % OUT_WIDTH_MAXPOOL3x3_S2) == 0 ? 0 : 1);
const int pool1_3x3_s2_inner_channel = (CHANNEL_FEATURE_GLOBAL / N_CHAN_MAXPOOL3x3_S2) + ((CHANNEL_FEATURE_GLOBAL % N_CHAN_MAXPOOL3x3_S2) == 0 ? 0 : 1);

const int pool1_3x3_s2_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S2;



#endif