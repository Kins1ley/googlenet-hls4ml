#ifndef SCHEDULE_CONFIG_H_
#define SCHEDULE_CONFIG_H_

#include "header.h"
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

//pool1_3x3_s2
///configuration
const int pool1_3x3_s2_allocate_global_in_feature_start_idx = 0;
const int pool1_3x3_s2_allocate_global_in_feature_num = 4;
const int pool1_3x3_s2_allocate_global_out_feature_start_idx = 4;
const int pool1_3x3_s2_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int pool1_3x3_s2_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int pool1_3x3_s2_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int pool1_3x3_s2_outer_in_channel = DIV_CEIL(pool1_3x3_s2_in_channel, pool1_3x3_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int pool1_3x3_s2_outer_height = DIV_CEIL(pool1_3x3_s2_in_height, HEIGHT_FEATURE_GLOBAL - pool1_3x3_s2_block_overlap_height);
const int pool1_3x3_s2_outer_width = DIV_CEIL(pool1_3x3_s2_in_width, WIDTH_FEATURE_GLOBAL - pool1_3x3_s2_block_overlap_width);
///interval between blocks
const int pool1_3x3_s2_block_interval_height = DIV_CEIL(DIV_CEIL(pool1_3x3_s2_in_height, pool1_3x3_s2_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int pool1_3x3_s2_block_interval_width = DIV_CEIL(DIV_CEIL(pool1_3x3_s2_in_width, pool1_3x3_s2_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int pool1_3x3_s2_block_in_height = pool1_3x3_s2_block_interval_height + pool1_3x3_s2_block_overlap_height;
const int pool1_3x3_s2_block_in_width = pool1_3x3_s2_block_interval_height + pool1_3x3_s2_block_overlap_height;
const int pool1_3x3_s2_block_in_channel = pool1_3x3_s2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool1_3x3_s2_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int pool1_3x3_s2_block_out_channel = pool1_3x3_s2_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;



#endif