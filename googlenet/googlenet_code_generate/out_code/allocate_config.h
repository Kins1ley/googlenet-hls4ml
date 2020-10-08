#ifndef ALLOCATE_CONFIG_H_
#define ALLOCATE_CONFIG_H_

#define CALC_BLOCK_H_W(shape,num_block,kernel_shape,stride) ((DIV_CEIL(DIV_CEIL(shape,num_block),stride)-1)*(stride)+kernel_shape)

#include "header.h"
/////allocate_config_insert/////
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
//pool1_norm1_1
///configuration
const int pool1_norm1_allocate_global_in_feature_start_idx = 0;
const int pool1_norm1_allocate_global_in_feature_num = 1;//multi global BRAM is not supportted yet
const int pool1_norm1_allocate_global_out_feature_start_idx = 1;
const int pool1_norm1_allocate_global_out_feature_num = 1;
///overlapped features between blocks
const int pool1_norm1_block_overlap_channel = 2 * pool1_norm1_deepth_radius;
///number of blocks(the dims of the outer loop)
const int pool1_norm1_outer_in_channel = DIV_CEIL(pool1_norm1_in_channel, pool1_norm1_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * pool1_norm1_deepth_radius));
const int pool1_norm1_outer_height = DIV_CEIL(pool1_norm1_in_height, HEIGHT_FEATURE_GLOBAL);
const int pool1_norm1_outer_width = DIV_CEIL(pool1_norm1_in_width, WIDTH_FEATURE_GLOBAL);
///interval between blocks
const int pool1_norm1_block_interval_channel = pool1_norm1_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * pool1_norm1_deepth_radius);//the spacing between blocks
const int pool1_norm1_block_interval_height = HEIGHT_FEATURE_GLOBAL;
const int pool1_norm1_block_interval_width = WIDTH_FEATURE_GLOBAL;
///dim of blocks
const int pool1_norm1_block_in_height = pool1_norm1_block_interval_height;
const int pool1_norm1_block_in_width = pool1_norm1_block_interval_height;
const int pool1_norm1_block_in_channel = pool1_norm1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool1_norm1_inner_pe_parallel = NUM_PE_LRN;//prallelism not supported
//conv2_3x3_reduce
///configuration
const int conv2_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int conv2_3x3_reduce_allocate_global_in_feature_num = 2;
const int conv2_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int conv2_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int conv2_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int conv2_3x3_reduce_allocate_global_out_feature_num = 2;
const int conv2_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int conv2_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int conv2_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int conv2_3x3_reduce_outer_in_channel = DIV_CEIL(conv2_3x3_reduce_in_channel, conv2_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? conv2_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int conv2_3x3_reduce_outer_height = DIV_CEIL(conv2_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- conv2_3x3_reduce_block_overlap_height);
const int conv2_3x3_reduce_outer_width = DIV_CEIL(conv2_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - conv2_3x3_reduce_block_overlap_width);
///interval between blocks
const int conv2_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(conv2_3x3_reduce_in_height, conv2_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int conv2_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(conv2_3x3_reduce_in_width, conv2_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int conv2_3x3_reduce_block_in_height = conv2_3x3_reduce_block_interval_height+ conv2_3x3_reduce_block_overlap_height;
const int conv2_3x3_reduce_block_in_width = conv2_3x3_reduce_block_interval_height + conv2_3x3_reduce_block_overlap_height;
const int conv2_3x3_reduce_block_in_channel = MIN(conv2_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int conv2_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int conv2_3x3_reduce_block_out_channel = MIN(conv2_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , conv2_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int conv2_3x3_reduce_outer_out_channel = DIV_CEIL(conv2_3x3_reduce_kernel_num, conv2_3x3_reduce_block_out_channel);//outer loop
//conv2_3x3
///configuration
const int conv2_3x3_allocate_global_in_feature_start_idx = 0;
const int conv2_3x3_allocate_global_in_feature_num = 2;
const int conv2_3x3_allocate_global_weight_3x3_start_idx = 0;
const int conv2_3x3_allocate_global_weight_3x3_num = 2;
const int conv2_3x3_allocate_global_out_feature_start_idx = 2;
const int conv2_3x3_allocate_global_out_feature_num = 2;
const int conv2_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int conv2_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int conv2_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int conv2_3x3_outer_in_channel = DIV_CEIL(conv2_3x3_in_channel, conv2_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? conv2_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int conv2_3x3_outer_height = DIV_CEIL(conv2_3x3_in_height, HEIGHT_FEATURE_GLOBAL- conv2_3x3_block_overlap_height);
const int conv2_3x3_outer_width = DIV_CEIL(conv2_3x3_in_width , WIDTH_FEATURE_GLOBAL - conv2_3x3_block_overlap_width);
///interval between blocks
const int conv2_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(conv2_3x3_in_height, conv2_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int conv2_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(conv2_3x3_in_width, conv2_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int conv2_3x3_block_in_height = conv2_3x3_block_interval_height+ conv2_3x3_block_overlap_height;
const int conv2_3x3_block_in_width = conv2_3x3_block_interval_height + conv2_3x3_block_overlap_height;
const int conv2_3x3_block_in_channel = MIN(conv2_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int conv2_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int conv2_3x3_block_out_channel = MIN(conv2_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , conv2_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int conv2_3x3_outer_out_channel = DIV_CEIL(conv2_3x3_kernel_num, conv2_3x3_block_out_channel);//outer loop
//conv2_norm2_1
///configuration
const int conv2_norm2_allocate_global_in_feature_start_idx = 0;
const int conv2_norm2_allocate_global_in_feature_num = 1;//multi global BRAM is not supportted yet
const int conv2_norm2_allocate_global_out_feature_start_idx = 1;
const int conv2_norm2_allocate_global_out_feature_num = 1;
///overlapped features between blocks
const int conv2_norm2_block_overlap_channel = 2 * conv2_norm2_deepth_radius;
///number of blocks(the dims of the outer loop)
const int conv2_norm2_outer_in_channel = DIV_CEIL(conv2_norm2_in_channel, conv2_norm2_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * conv2_norm2_deepth_radius));
const int conv2_norm2_outer_height = DIV_CEIL(conv2_norm2_in_height, HEIGHT_FEATURE_GLOBAL);
const int conv2_norm2_outer_width = DIV_CEIL(conv2_norm2_in_width, WIDTH_FEATURE_GLOBAL);
///interval between blocks
const int conv2_norm2_block_interval_channel = conv2_norm2_allocate_global_in_feature_num * (CHANNEL_FEATURE_GLOBAL - 2 * conv2_norm2_deepth_radius);//the spacing between blocks
const int conv2_norm2_block_interval_height = HEIGHT_FEATURE_GLOBAL;
const int conv2_norm2_block_interval_width = WIDTH_FEATURE_GLOBAL;
///dim of blocks
const int conv2_norm2_block_in_height = conv2_norm2_block_interval_height;
const int conv2_norm2_block_in_width = conv2_norm2_block_interval_height;
const int conv2_norm2_block_in_channel = conv2_norm2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int conv2_norm2_inner_pe_parallel = NUM_PE_LRN;//prallelism not supported
//pool2_3x3_s2
///configuration
const int pool2_3x3_s2_allocate_global_in_feature_start_idx = 0;
const int pool2_3x3_s2_allocate_global_in_feature_num = 4;
const int pool2_3x3_s2_allocate_global_out_feature_start_idx = 4;
const int pool2_3x3_s2_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int pool2_3x3_s2_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S2 - 1;
const int pool2_3x3_s2_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S2 - 1;
///number of blocks(the dims of the outer loop)
const int pool2_3x3_s2_outer_in_channel = DIV_CEIL(pool2_3x3_s2_in_channel, pool2_3x3_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int pool2_3x3_s2_outer_height = DIV_CEIL(pool2_3x3_s2_in_height, HEIGHT_FEATURE_GLOBAL - pool2_3x3_s2_block_overlap_height);
const int pool2_3x3_s2_outer_width = DIV_CEIL(pool2_3x3_s2_in_width, WIDTH_FEATURE_GLOBAL - pool2_3x3_s2_block_overlap_width);
///interval between blocks
const int pool2_3x3_s2_block_interval_height = DIV_CEIL(DIV_CEIL(pool2_3x3_s2_in_height, pool2_3x3_s2_outer_height), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;//the spacing between blocks
const int pool2_3x3_s2_block_interval_width = DIV_CEIL(DIV_CEIL(pool2_3x3_s2_in_width, pool2_3x3_s2_outer_width), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;
///dim of blocks
const int pool2_3x3_s2_block_in_height = pool2_3x3_s2_block_interval_height + pool2_3x3_s2_block_overlap_height;
const int pool2_3x3_s2_block_in_width = pool2_3x3_s2_block_interval_height + pool2_3x3_s2_block_overlap_height;
const int pool2_3x3_s2_block_in_channel = pool2_3x3_s2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool2_3x3_s2_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S2;
///dim of kernels
const int pool2_3x3_s2_block_out_channel = pool2_3x3_s2_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_3a_1x1
///configuration
const int inception_3a_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_3a_1x1_allocate_global_in_feature_num = 2;
const int inception_3a_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_3a_1x1_allocate_global_weight_1x1_num = 2;
const int inception_3a_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_3a_1x1_allocate_global_out_feature_num = 2;
const int inception_3a_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3a_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3a_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_1x1_outer_in_channel = DIV_CEIL(inception_3a_1x1_in_channel, inception_3a_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3a_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_1x1_outer_height = DIV_CEIL(inception_3a_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_3a_1x1_block_overlap_height);
const int inception_3a_1x1_outer_width = DIV_CEIL(inception_3a_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_3a_1x1_block_overlap_width);
///interval between blocks
const int inception_3a_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_1x1_in_height, inception_3a_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3a_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_1x1_in_width, inception_3a_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3a_1x1_block_in_height = inception_3a_1x1_block_interval_height+ inception_3a_1x1_block_overlap_height;
const int inception_3a_1x1_block_in_width = inception_3a_1x1_block_interval_height + inception_3a_1x1_block_overlap_height;
const int inception_3a_1x1_block_in_channel = MIN(inception_3a_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3a_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3a_1x1_block_out_channel = MIN(inception_3a_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3a_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_1x1_outer_out_channel = DIV_CEIL(inception_3a_1x1_kernel_num, inception_3a_1x1_block_out_channel);//outer loop
//inception_3a_3x3_reduce
///configuration
const int inception_3a_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_3a_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_3a_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_3a_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_3a_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_3a_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_3a_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3a_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3a_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_3x3_reduce_outer_in_channel = DIV_CEIL(inception_3a_3x3_reduce_in_channel, inception_3a_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3a_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_3x3_reduce_outer_height = DIV_CEIL(inception_3a_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_3a_3x3_reduce_block_overlap_height);
const int inception_3a_3x3_reduce_outer_width = DIV_CEIL(inception_3a_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_3a_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_3a_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_3x3_reduce_in_height, inception_3a_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3a_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_3x3_reduce_in_width, inception_3a_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3a_3x3_reduce_block_in_height = inception_3a_3x3_reduce_block_interval_height+ inception_3a_3x3_reduce_block_overlap_height;
const int inception_3a_3x3_reduce_block_in_width = inception_3a_3x3_reduce_block_interval_height + inception_3a_3x3_reduce_block_overlap_height;
const int inception_3a_3x3_reduce_block_in_channel = MIN(inception_3a_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3a_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3a_3x3_reduce_block_out_channel = MIN(inception_3a_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3a_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_3x3_reduce_outer_out_channel = DIV_CEIL(inception_3a_3x3_reduce_kernel_num, inception_3a_3x3_reduce_block_out_channel);//outer loop
//inception_3a_3x3
///configuration
const int inception_3a_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_3a_3x3_allocate_global_in_feature_num = 2;
const int inception_3a_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_3a_3x3_allocate_global_weight_3x3_num = 2;
const int inception_3a_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_3a_3x3_allocate_global_out_feature_num = 2;
const int inception_3a_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3a_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_3a_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_3x3_outer_in_channel = DIV_CEIL(inception_3a_3x3_in_channel, inception_3a_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_3a_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_3a_3x3_outer_height = DIV_CEIL(inception_3a_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_3a_3x3_block_overlap_height);
const int inception_3a_3x3_outer_width = DIV_CEIL(inception_3a_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_3a_3x3_block_overlap_width);
///interval between blocks
const int inception_3a_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_3x3_in_height, inception_3a_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_3a_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_3x3_in_width, inception_3a_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_3a_3x3_block_in_height = inception_3a_3x3_block_interval_height+ inception_3a_3x3_block_overlap_height;
const int inception_3a_3x3_block_in_width = inception_3a_3x3_block_interval_height + inception_3a_3x3_block_overlap_height;
const int inception_3a_3x3_block_in_channel = MIN(inception_3a_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_3a_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_3a_3x3_block_out_channel = MIN(inception_3a_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3a_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_3a_3x3_outer_out_channel = DIV_CEIL(inception_3a_3x3_kernel_num, inception_3a_3x3_block_out_channel);//outer loop
//inception_3a_5x5_reduce
///configuration
const int inception_3a_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_3a_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_3a_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_3a_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_3a_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_3a_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_3a_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3a_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3a_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_5x5_reduce_outer_in_channel = DIV_CEIL(inception_3a_5x5_reduce_in_channel, inception_3a_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3a_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_5x5_reduce_outer_height = DIV_CEIL(inception_3a_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_3a_5x5_reduce_block_overlap_height);
const int inception_3a_5x5_reduce_outer_width = DIV_CEIL(inception_3a_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_3a_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_3a_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_5x5_reduce_in_height, inception_3a_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3a_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_5x5_reduce_in_width, inception_3a_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3a_5x5_reduce_block_in_height = inception_3a_5x5_reduce_block_interval_height+ inception_3a_5x5_reduce_block_overlap_height;
const int inception_3a_5x5_reduce_block_in_width = inception_3a_5x5_reduce_block_interval_height + inception_3a_5x5_reduce_block_overlap_height;
const int inception_3a_5x5_reduce_block_in_channel = MIN(inception_3a_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3a_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3a_5x5_reduce_block_out_channel = MIN(inception_3a_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3a_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_5x5_reduce_outer_out_channel = DIV_CEIL(inception_3a_5x5_reduce_kernel_num, inception_3a_5x5_reduce_block_out_channel);//outer loop
//inception_3a_5x5
///configuration
const int inception_3a_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_3a_5x5_allocate_global_in_feature_num = 2;
const int inception_3a_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_3a_5x5_allocate_global_weight_5x5_num = 2;
const int inception_3a_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_3a_5x5_allocate_global_out_feature_num = 2;
const int inception_3a_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3a_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_3a_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_5x5_outer_in_channel = DIV_CEIL(inception_3a_5x5_in_channel, inception_3a_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_3a_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_3a_5x5_outer_height = DIV_CEIL(inception_3a_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_3a_5x5_block_overlap_height);
const int inception_3a_5x5_outer_width = DIV_CEIL(inception_3a_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_3a_5x5_block_overlap_width);
///interval between blocks
const int inception_3a_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_5x5_in_height, inception_3a_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_3a_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_5x5_in_width, inception_3a_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_3a_5x5_block_in_height = inception_3a_5x5_block_interval_height+ inception_3a_5x5_block_overlap_height;
const int inception_3a_5x5_block_in_width = inception_3a_5x5_block_interval_height + inception_3a_5x5_block_overlap_height;
const int inception_3a_5x5_block_in_channel = MIN(inception_3a_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_3a_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_3a_5x5_block_out_channel = MIN(inception_3a_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3a_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_3a_5x5_outer_out_channel = DIV_CEIL(inception_3a_5x5_kernel_num, inception_3a_5x5_block_out_channel);//outer loop
//inception_3a_pool
///configuration
const int inception_3a_pool_allocate_global_in_feature_start_idx = 0;
const int inception_3a_pool_allocate_global_in_feature_num = 4;
const int inception_3a_pool_allocate_global_out_feature_start_idx = 4;
const int inception_3a_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_3a_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_3a_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_pool_outer_in_channel = DIV_CEIL(inception_3a_pool_in_channel, inception_3a_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_3a_pool_outer_height = DIV_CEIL(inception_3a_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_3a_pool_block_overlap_height);
const int inception_3a_pool_outer_width = DIV_CEIL(inception_3a_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_3a_pool_block_overlap_width);
///interval between blocks
const int inception_3a_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_pool_in_height, inception_3a_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_3a_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_pool_in_width, inception_3a_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_3a_pool_block_in_height = inception_3a_pool_block_interval_height + inception_3a_pool_block_overlap_height;
const int inception_3a_pool_block_in_width = inception_3a_pool_block_interval_height + inception_3a_pool_block_overlap_height;
const int inception_3a_pool_block_in_channel = inception_3a_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_3a_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_3a_pool_block_out_channel = inception_3a_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_3a_pool_proj
///configuration
const int inception_3a_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_3a_pool_proj_allocate_global_in_feature_num = 2;
const int inception_3a_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_3a_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_3a_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_3a_pool_proj_allocate_global_out_feature_num = 2;
const int inception_3a_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3a_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3a_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3a_pool_proj_outer_in_channel = DIV_CEIL(inception_3a_pool_proj_in_channel, inception_3a_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3a_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_pool_proj_outer_height = DIV_CEIL(inception_3a_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_3a_pool_proj_block_overlap_height);
const int inception_3a_pool_proj_outer_width = DIV_CEIL(inception_3a_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_3a_pool_proj_block_overlap_width);
///interval between blocks
const int inception_3a_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3a_pool_proj_in_height, inception_3a_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3a_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3a_pool_proj_in_width, inception_3a_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3a_pool_proj_block_in_height = inception_3a_pool_proj_block_interval_height+ inception_3a_pool_proj_block_overlap_height;
const int inception_3a_pool_proj_block_in_width = inception_3a_pool_proj_block_interval_height + inception_3a_pool_proj_block_overlap_height;
const int inception_3a_pool_proj_block_in_channel = MIN(inception_3a_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3a_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3a_pool_proj_block_out_channel = MIN(inception_3a_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3a_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3a_pool_proj_outer_out_channel = DIV_CEIL(inception_3a_pool_proj_kernel_num, inception_3a_pool_proj_block_out_channel);//outer loop
//inception_3b_1x1
///configuration
const int inception_3b_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_3b_1x1_allocate_global_in_feature_num = 2;
const int inception_3b_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_3b_1x1_allocate_global_weight_1x1_num = 2;
const int inception_3b_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_3b_1x1_allocate_global_out_feature_num = 2;
const int inception_3b_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3b_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3b_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_1x1_outer_in_channel = DIV_CEIL(inception_3b_1x1_in_channel, inception_3b_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3b_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_1x1_outer_height = DIV_CEIL(inception_3b_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_3b_1x1_block_overlap_height);
const int inception_3b_1x1_outer_width = DIV_CEIL(inception_3b_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_3b_1x1_block_overlap_width);
///interval between blocks
const int inception_3b_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_1x1_in_height, inception_3b_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3b_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_1x1_in_width, inception_3b_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3b_1x1_block_in_height = inception_3b_1x1_block_interval_height+ inception_3b_1x1_block_overlap_height;
const int inception_3b_1x1_block_in_width = inception_3b_1x1_block_interval_height + inception_3b_1x1_block_overlap_height;
const int inception_3b_1x1_block_in_channel = MIN(inception_3b_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3b_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3b_1x1_block_out_channel = MIN(inception_3b_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3b_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_1x1_outer_out_channel = DIV_CEIL(inception_3b_1x1_kernel_num, inception_3b_1x1_block_out_channel);//outer loop
//inception_3b_3x3_reduce
///configuration
const int inception_3b_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_3b_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_3b_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_3b_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_3b_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_3b_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_3b_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3b_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3b_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_3x3_reduce_outer_in_channel = DIV_CEIL(inception_3b_3x3_reduce_in_channel, inception_3b_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3b_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_3x3_reduce_outer_height = DIV_CEIL(inception_3b_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_3b_3x3_reduce_block_overlap_height);
const int inception_3b_3x3_reduce_outer_width = DIV_CEIL(inception_3b_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_3b_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_3b_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_3x3_reduce_in_height, inception_3b_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3b_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_3x3_reduce_in_width, inception_3b_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3b_3x3_reduce_block_in_height = inception_3b_3x3_reduce_block_interval_height+ inception_3b_3x3_reduce_block_overlap_height;
const int inception_3b_3x3_reduce_block_in_width = inception_3b_3x3_reduce_block_interval_height + inception_3b_3x3_reduce_block_overlap_height;
const int inception_3b_3x3_reduce_block_in_channel = MIN(inception_3b_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3b_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3b_3x3_reduce_block_out_channel = MIN(inception_3b_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3b_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_3x3_reduce_outer_out_channel = DIV_CEIL(inception_3b_3x3_reduce_kernel_num, inception_3b_3x3_reduce_block_out_channel);//outer loop
//inception_3b_3x3
///configuration
const int inception_3b_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_3b_3x3_allocate_global_in_feature_num = 2;
const int inception_3b_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_3b_3x3_allocate_global_weight_3x3_num = 2;
const int inception_3b_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_3b_3x3_allocate_global_out_feature_num = 2;
const int inception_3b_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3b_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_3b_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_3x3_outer_in_channel = DIV_CEIL(inception_3b_3x3_in_channel, inception_3b_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_3b_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_3b_3x3_outer_height = DIV_CEIL(inception_3b_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_3b_3x3_block_overlap_height);
const int inception_3b_3x3_outer_width = DIV_CEIL(inception_3b_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_3b_3x3_block_overlap_width);
///interval between blocks
const int inception_3b_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_3x3_in_height, inception_3b_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_3b_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_3x3_in_width, inception_3b_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_3b_3x3_block_in_height = inception_3b_3x3_block_interval_height+ inception_3b_3x3_block_overlap_height;
const int inception_3b_3x3_block_in_width = inception_3b_3x3_block_interval_height + inception_3b_3x3_block_overlap_height;
const int inception_3b_3x3_block_in_channel = MIN(inception_3b_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_3b_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_3b_3x3_block_out_channel = MIN(inception_3b_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3b_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_3b_3x3_outer_out_channel = DIV_CEIL(inception_3b_3x3_kernel_num, inception_3b_3x3_block_out_channel);//outer loop
//inception_3b_5x5_reduce
///configuration
const int inception_3b_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_3b_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_3b_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_3b_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_3b_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_3b_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_3b_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3b_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3b_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_5x5_reduce_outer_in_channel = DIV_CEIL(inception_3b_5x5_reduce_in_channel, inception_3b_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3b_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_5x5_reduce_outer_height = DIV_CEIL(inception_3b_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_3b_5x5_reduce_block_overlap_height);
const int inception_3b_5x5_reduce_outer_width = DIV_CEIL(inception_3b_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_3b_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_3b_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_5x5_reduce_in_height, inception_3b_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3b_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_5x5_reduce_in_width, inception_3b_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3b_5x5_reduce_block_in_height = inception_3b_5x5_reduce_block_interval_height+ inception_3b_5x5_reduce_block_overlap_height;
const int inception_3b_5x5_reduce_block_in_width = inception_3b_5x5_reduce_block_interval_height + inception_3b_5x5_reduce_block_overlap_height;
const int inception_3b_5x5_reduce_block_in_channel = MIN(inception_3b_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3b_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3b_5x5_reduce_block_out_channel = MIN(inception_3b_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3b_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_5x5_reduce_outer_out_channel = DIV_CEIL(inception_3b_5x5_reduce_kernel_num, inception_3b_5x5_reduce_block_out_channel);//outer loop
//inception_3b_5x5
///configuration
const int inception_3b_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_3b_5x5_allocate_global_in_feature_num = 2;
const int inception_3b_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_3b_5x5_allocate_global_weight_5x5_num = 2;
const int inception_3b_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_3b_5x5_allocate_global_out_feature_num = 2;
const int inception_3b_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3b_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_3b_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_5x5_outer_in_channel = DIV_CEIL(inception_3b_5x5_in_channel, inception_3b_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_3b_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_3b_5x5_outer_height = DIV_CEIL(inception_3b_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_3b_5x5_block_overlap_height);
const int inception_3b_5x5_outer_width = DIV_CEIL(inception_3b_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_3b_5x5_block_overlap_width);
///interval between blocks
const int inception_3b_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_5x5_in_height, inception_3b_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_3b_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_5x5_in_width, inception_3b_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_3b_5x5_block_in_height = inception_3b_5x5_block_interval_height+ inception_3b_5x5_block_overlap_height;
const int inception_3b_5x5_block_in_width = inception_3b_5x5_block_interval_height + inception_3b_5x5_block_overlap_height;
const int inception_3b_5x5_block_in_channel = MIN(inception_3b_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_3b_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_3b_5x5_block_out_channel = MIN(inception_3b_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3b_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_3b_5x5_outer_out_channel = DIV_CEIL(inception_3b_5x5_kernel_num, inception_3b_5x5_block_out_channel);//outer loop
//inception_3b_pool
///configuration
const int inception_3b_pool_allocate_global_in_feature_start_idx = 0;
const int inception_3b_pool_allocate_global_in_feature_num = 4;
const int inception_3b_pool_allocate_global_out_feature_start_idx = 4;
const int inception_3b_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_3b_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_3b_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_pool_outer_in_channel = DIV_CEIL(inception_3b_pool_in_channel, inception_3b_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_3b_pool_outer_height = DIV_CEIL(inception_3b_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_3b_pool_block_overlap_height);
const int inception_3b_pool_outer_width = DIV_CEIL(inception_3b_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_3b_pool_block_overlap_width);
///interval between blocks
const int inception_3b_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_pool_in_height, inception_3b_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_3b_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_pool_in_width, inception_3b_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_3b_pool_block_in_height = inception_3b_pool_block_interval_height + inception_3b_pool_block_overlap_height;
const int inception_3b_pool_block_in_width = inception_3b_pool_block_interval_height + inception_3b_pool_block_overlap_height;
const int inception_3b_pool_block_in_channel = inception_3b_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_3b_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_3b_pool_block_out_channel = inception_3b_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_3b_pool_proj
///configuration
const int inception_3b_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_3b_pool_proj_allocate_global_in_feature_num = 2;
const int inception_3b_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_3b_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_3b_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_3b_pool_proj_allocate_global_out_feature_num = 2;
const int inception_3b_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_3b_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_3b_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_3b_pool_proj_outer_in_channel = DIV_CEIL(inception_3b_pool_proj_in_channel, inception_3b_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_3b_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_pool_proj_outer_height = DIV_CEIL(inception_3b_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_3b_pool_proj_block_overlap_height);
const int inception_3b_pool_proj_outer_width = DIV_CEIL(inception_3b_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_3b_pool_proj_block_overlap_width);
///interval between blocks
const int inception_3b_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_3b_pool_proj_in_height, inception_3b_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_3b_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_3b_pool_proj_in_width, inception_3b_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_3b_pool_proj_block_in_height = inception_3b_pool_proj_block_interval_height+ inception_3b_pool_proj_block_overlap_height;
const int inception_3b_pool_proj_block_in_width = inception_3b_pool_proj_block_interval_height + inception_3b_pool_proj_block_overlap_height;
const int inception_3b_pool_proj_block_in_channel = MIN(inception_3b_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_3b_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_3b_pool_proj_block_out_channel = MIN(inception_3b_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_3b_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_3b_pool_proj_outer_out_channel = DIV_CEIL(inception_3b_pool_proj_kernel_num, inception_3b_pool_proj_block_out_channel);//outer loop
//pool3_3x3_s2
///configuration
const int pool3_3x3_s2_allocate_global_in_feature_start_idx = 0;
const int pool3_3x3_s2_allocate_global_in_feature_num = 4;
const int pool3_3x3_s2_allocate_global_out_feature_start_idx = 4;
const int pool3_3x3_s2_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int pool3_3x3_s2_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S2 - 1;
const int pool3_3x3_s2_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S2 - 1;
///number of blocks(the dims of the outer loop)
const int pool3_3x3_s2_outer_in_channel = DIV_CEIL(pool3_3x3_s2_in_channel, pool3_3x3_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int pool3_3x3_s2_outer_height = DIV_CEIL(pool3_3x3_s2_in_height, HEIGHT_FEATURE_GLOBAL - pool3_3x3_s2_block_overlap_height);
const int pool3_3x3_s2_outer_width = DIV_CEIL(pool3_3x3_s2_in_width, WIDTH_FEATURE_GLOBAL - pool3_3x3_s2_block_overlap_width);
///interval between blocks
const int pool3_3x3_s2_block_interval_height = DIV_CEIL(DIV_CEIL(pool3_3x3_s2_in_height, pool3_3x3_s2_outer_height), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;//the spacing between blocks
const int pool3_3x3_s2_block_interval_width = DIV_CEIL(DIV_CEIL(pool3_3x3_s2_in_width, pool3_3x3_s2_outer_width), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;
///dim of blocks
const int pool3_3x3_s2_block_in_height = pool3_3x3_s2_block_interval_height + pool3_3x3_s2_block_overlap_height;
const int pool3_3x3_s2_block_in_width = pool3_3x3_s2_block_interval_height + pool3_3x3_s2_block_overlap_height;
const int pool3_3x3_s2_block_in_channel = pool3_3x3_s2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool3_3x3_s2_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S2;
///dim of kernels
const int pool3_3x3_s2_block_out_channel = pool3_3x3_s2_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_4a_1x1
///configuration
const int inception_4a_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_4a_1x1_allocate_global_in_feature_num = 2;
const int inception_4a_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_4a_1x1_allocate_global_weight_1x1_num = 2;
const int inception_4a_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_4a_1x1_allocate_global_out_feature_num = 2;
const int inception_4a_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4a_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4a_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_1x1_outer_in_channel = DIV_CEIL(inception_4a_1x1_in_channel, inception_4a_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4a_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_1x1_outer_height = DIV_CEIL(inception_4a_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_4a_1x1_block_overlap_height);
const int inception_4a_1x1_outer_width = DIV_CEIL(inception_4a_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_4a_1x1_block_overlap_width);
///interval between blocks
const int inception_4a_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_1x1_in_height, inception_4a_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4a_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_1x1_in_width, inception_4a_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4a_1x1_block_in_height = inception_4a_1x1_block_interval_height+ inception_4a_1x1_block_overlap_height;
const int inception_4a_1x1_block_in_width = inception_4a_1x1_block_interval_height + inception_4a_1x1_block_overlap_height;
const int inception_4a_1x1_block_in_channel = MIN(inception_4a_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4a_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4a_1x1_block_out_channel = MIN(inception_4a_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4a_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_1x1_outer_out_channel = DIV_CEIL(inception_4a_1x1_kernel_num, inception_4a_1x1_block_out_channel);//outer loop
//inception_4a_3x3_reduce
///configuration
const int inception_4a_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4a_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_4a_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4a_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4a_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4a_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_4a_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4a_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4a_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_3x3_reduce_outer_in_channel = DIV_CEIL(inception_4a_3x3_reduce_in_channel, inception_4a_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4a_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_3x3_reduce_outer_height = DIV_CEIL(inception_4a_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4a_3x3_reduce_block_overlap_height);
const int inception_4a_3x3_reduce_outer_width = DIV_CEIL(inception_4a_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4a_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_4a_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_3x3_reduce_in_height, inception_4a_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4a_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_3x3_reduce_in_width, inception_4a_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4a_3x3_reduce_block_in_height = inception_4a_3x3_reduce_block_interval_height+ inception_4a_3x3_reduce_block_overlap_height;
const int inception_4a_3x3_reduce_block_in_width = inception_4a_3x3_reduce_block_interval_height + inception_4a_3x3_reduce_block_overlap_height;
const int inception_4a_3x3_reduce_block_in_channel = MIN(inception_4a_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4a_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4a_3x3_reduce_block_out_channel = MIN(inception_4a_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4a_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_3x3_reduce_outer_out_channel = DIV_CEIL(inception_4a_3x3_reduce_kernel_num, inception_4a_3x3_reduce_block_out_channel);//outer loop
//inception_4a_3x3
///configuration
const int inception_4a_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_4a_3x3_allocate_global_in_feature_num = 2;
const int inception_4a_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_4a_3x3_allocate_global_weight_3x3_num = 2;
const int inception_4a_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_4a_3x3_allocate_global_out_feature_num = 2;
const int inception_4a_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4a_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_4a_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_3x3_outer_in_channel = DIV_CEIL(inception_4a_3x3_in_channel, inception_4a_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_4a_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4a_3x3_outer_height = DIV_CEIL(inception_4a_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_4a_3x3_block_overlap_height);
const int inception_4a_3x3_outer_width = DIV_CEIL(inception_4a_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_4a_3x3_block_overlap_width);
///interval between blocks
const int inception_4a_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_3x3_in_height, inception_4a_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_4a_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_3x3_in_width, inception_4a_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_4a_3x3_block_in_height = inception_4a_3x3_block_interval_height+ inception_4a_3x3_block_overlap_height;
const int inception_4a_3x3_block_in_width = inception_4a_3x3_block_interval_height + inception_4a_3x3_block_overlap_height;
const int inception_4a_3x3_block_in_channel = MIN(inception_4a_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_4a_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_4a_3x3_block_out_channel = MIN(inception_4a_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4a_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4a_3x3_outer_out_channel = DIV_CEIL(inception_4a_3x3_kernel_num, inception_4a_3x3_block_out_channel);//outer loop
//inception_4a_5x5_reduce
///configuration
const int inception_4a_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4a_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_4a_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4a_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4a_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4a_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_4a_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4a_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4a_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_5x5_reduce_outer_in_channel = DIV_CEIL(inception_4a_5x5_reduce_in_channel, inception_4a_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4a_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_5x5_reduce_outer_height = DIV_CEIL(inception_4a_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4a_5x5_reduce_block_overlap_height);
const int inception_4a_5x5_reduce_outer_width = DIV_CEIL(inception_4a_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4a_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_4a_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_5x5_reduce_in_height, inception_4a_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4a_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_5x5_reduce_in_width, inception_4a_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4a_5x5_reduce_block_in_height = inception_4a_5x5_reduce_block_interval_height+ inception_4a_5x5_reduce_block_overlap_height;
const int inception_4a_5x5_reduce_block_in_width = inception_4a_5x5_reduce_block_interval_height + inception_4a_5x5_reduce_block_overlap_height;
const int inception_4a_5x5_reduce_block_in_channel = MIN(inception_4a_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4a_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4a_5x5_reduce_block_out_channel = MIN(inception_4a_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4a_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_5x5_reduce_outer_out_channel = DIV_CEIL(inception_4a_5x5_reduce_kernel_num, inception_4a_5x5_reduce_block_out_channel);//outer loop
//inception_4a_5x5
///configuration
const int inception_4a_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_4a_5x5_allocate_global_in_feature_num = 2;
const int inception_4a_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_4a_5x5_allocate_global_weight_5x5_num = 2;
const int inception_4a_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_4a_5x5_allocate_global_out_feature_num = 2;
const int inception_4a_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4a_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_4a_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_5x5_outer_in_channel = DIV_CEIL(inception_4a_5x5_in_channel, inception_4a_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_4a_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4a_5x5_outer_height = DIV_CEIL(inception_4a_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_4a_5x5_block_overlap_height);
const int inception_4a_5x5_outer_width = DIV_CEIL(inception_4a_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_4a_5x5_block_overlap_width);
///interval between blocks
const int inception_4a_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_5x5_in_height, inception_4a_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_4a_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_5x5_in_width, inception_4a_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_4a_5x5_block_in_height = inception_4a_5x5_block_interval_height+ inception_4a_5x5_block_overlap_height;
const int inception_4a_5x5_block_in_width = inception_4a_5x5_block_interval_height + inception_4a_5x5_block_overlap_height;
const int inception_4a_5x5_block_in_channel = MIN(inception_4a_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_4a_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_4a_5x5_block_out_channel = MIN(inception_4a_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4a_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4a_5x5_outer_out_channel = DIV_CEIL(inception_4a_5x5_kernel_num, inception_4a_5x5_block_out_channel);//outer loop
//inception_4a_pool
///configuration
const int inception_4a_pool_allocate_global_in_feature_start_idx = 0;
const int inception_4a_pool_allocate_global_in_feature_num = 4;
const int inception_4a_pool_allocate_global_out_feature_start_idx = 4;
const int inception_4a_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_4a_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_4a_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_pool_outer_in_channel = DIV_CEIL(inception_4a_pool_in_channel, inception_4a_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_4a_pool_outer_height = DIV_CEIL(inception_4a_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_4a_pool_block_overlap_height);
const int inception_4a_pool_outer_width = DIV_CEIL(inception_4a_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_4a_pool_block_overlap_width);
///interval between blocks
const int inception_4a_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_pool_in_height, inception_4a_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_4a_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_pool_in_width, inception_4a_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_4a_pool_block_in_height = inception_4a_pool_block_interval_height + inception_4a_pool_block_overlap_height;
const int inception_4a_pool_block_in_width = inception_4a_pool_block_interval_height + inception_4a_pool_block_overlap_height;
const int inception_4a_pool_block_in_channel = inception_4a_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_4a_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_4a_pool_block_out_channel = inception_4a_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_4a_pool_proj
///configuration
const int inception_4a_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_4a_pool_proj_allocate_global_in_feature_num = 2;
const int inception_4a_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_4a_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_4a_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_4a_pool_proj_allocate_global_out_feature_num = 2;
const int inception_4a_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4a_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4a_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4a_pool_proj_outer_in_channel = DIV_CEIL(inception_4a_pool_proj_in_channel, inception_4a_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4a_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_pool_proj_outer_height = DIV_CEIL(inception_4a_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_4a_pool_proj_block_overlap_height);
const int inception_4a_pool_proj_outer_width = DIV_CEIL(inception_4a_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_4a_pool_proj_block_overlap_width);
///interval between blocks
const int inception_4a_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4a_pool_proj_in_height, inception_4a_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4a_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4a_pool_proj_in_width, inception_4a_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4a_pool_proj_block_in_height = inception_4a_pool_proj_block_interval_height+ inception_4a_pool_proj_block_overlap_height;
const int inception_4a_pool_proj_block_in_width = inception_4a_pool_proj_block_interval_height + inception_4a_pool_proj_block_overlap_height;
const int inception_4a_pool_proj_block_in_channel = MIN(inception_4a_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4a_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4a_pool_proj_block_out_channel = MIN(inception_4a_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4a_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4a_pool_proj_outer_out_channel = DIV_CEIL(inception_4a_pool_proj_kernel_num, inception_4a_pool_proj_block_out_channel);//outer loop
//inception_4b_1x1
///configuration
const int inception_4b_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_4b_1x1_allocate_global_in_feature_num = 2;
const int inception_4b_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_4b_1x1_allocate_global_weight_1x1_num = 2;
const int inception_4b_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_4b_1x1_allocate_global_out_feature_num = 2;
const int inception_4b_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4b_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4b_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_1x1_outer_in_channel = DIV_CEIL(inception_4b_1x1_in_channel, inception_4b_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4b_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_1x1_outer_height = DIV_CEIL(inception_4b_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_4b_1x1_block_overlap_height);
const int inception_4b_1x1_outer_width = DIV_CEIL(inception_4b_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_4b_1x1_block_overlap_width);
///interval between blocks
const int inception_4b_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_1x1_in_height, inception_4b_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4b_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_1x1_in_width, inception_4b_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4b_1x1_block_in_height = inception_4b_1x1_block_interval_height+ inception_4b_1x1_block_overlap_height;
const int inception_4b_1x1_block_in_width = inception_4b_1x1_block_interval_height + inception_4b_1x1_block_overlap_height;
const int inception_4b_1x1_block_in_channel = MIN(inception_4b_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4b_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4b_1x1_block_out_channel = MIN(inception_4b_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4b_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_1x1_outer_out_channel = DIV_CEIL(inception_4b_1x1_kernel_num, inception_4b_1x1_block_out_channel);//outer loop
//inception_4b_3x3_reduce
///configuration
const int inception_4b_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4b_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_4b_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4b_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4b_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4b_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_4b_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4b_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4b_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_3x3_reduce_outer_in_channel = DIV_CEIL(inception_4b_3x3_reduce_in_channel, inception_4b_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4b_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_3x3_reduce_outer_height = DIV_CEIL(inception_4b_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4b_3x3_reduce_block_overlap_height);
const int inception_4b_3x3_reduce_outer_width = DIV_CEIL(inception_4b_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4b_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_4b_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_3x3_reduce_in_height, inception_4b_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4b_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_3x3_reduce_in_width, inception_4b_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4b_3x3_reduce_block_in_height = inception_4b_3x3_reduce_block_interval_height+ inception_4b_3x3_reduce_block_overlap_height;
const int inception_4b_3x3_reduce_block_in_width = inception_4b_3x3_reduce_block_interval_height + inception_4b_3x3_reduce_block_overlap_height;
const int inception_4b_3x3_reduce_block_in_channel = MIN(inception_4b_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4b_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4b_3x3_reduce_block_out_channel = MIN(inception_4b_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4b_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_3x3_reduce_outer_out_channel = DIV_CEIL(inception_4b_3x3_reduce_kernel_num, inception_4b_3x3_reduce_block_out_channel);//outer loop
//inception_4b_3x3
///configuration
const int inception_4b_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_4b_3x3_allocate_global_in_feature_num = 2;
const int inception_4b_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_4b_3x3_allocate_global_weight_3x3_num = 2;
const int inception_4b_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_4b_3x3_allocate_global_out_feature_num = 2;
const int inception_4b_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4b_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_4b_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_3x3_outer_in_channel = DIV_CEIL(inception_4b_3x3_in_channel, inception_4b_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_4b_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4b_3x3_outer_height = DIV_CEIL(inception_4b_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_4b_3x3_block_overlap_height);
const int inception_4b_3x3_outer_width = DIV_CEIL(inception_4b_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_4b_3x3_block_overlap_width);
///interval between blocks
const int inception_4b_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_3x3_in_height, inception_4b_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_4b_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_3x3_in_width, inception_4b_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_4b_3x3_block_in_height = inception_4b_3x3_block_interval_height+ inception_4b_3x3_block_overlap_height;
const int inception_4b_3x3_block_in_width = inception_4b_3x3_block_interval_height + inception_4b_3x3_block_overlap_height;
const int inception_4b_3x3_block_in_channel = MIN(inception_4b_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_4b_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_4b_3x3_block_out_channel = MIN(inception_4b_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4b_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4b_3x3_outer_out_channel = DIV_CEIL(inception_4b_3x3_kernel_num, inception_4b_3x3_block_out_channel);//outer loop
//inception_4b_5x5_reduce
///configuration
const int inception_4b_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4b_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_4b_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4b_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4b_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4b_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_4b_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4b_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4b_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_5x5_reduce_outer_in_channel = DIV_CEIL(inception_4b_5x5_reduce_in_channel, inception_4b_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4b_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_5x5_reduce_outer_height = DIV_CEIL(inception_4b_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4b_5x5_reduce_block_overlap_height);
const int inception_4b_5x5_reduce_outer_width = DIV_CEIL(inception_4b_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4b_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_4b_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_5x5_reduce_in_height, inception_4b_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4b_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_5x5_reduce_in_width, inception_4b_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4b_5x5_reduce_block_in_height = inception_4b_5x5_reduce_block_interval_height+ inception_4b_5x5_reduce_block_overlap_height;
const int inception_4b_5x5_reduce_block_in_width = inception_4b_5x5_reduce_block_interval_height + inception_4b_5x5_reduce_block_overlap_height;
const int inception_4b_5x5_reduce_block_in_channel = MIN(inception_4b_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4b_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4b_5x5_reduce_block_out_channel = MIN(inception_4b_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4b_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_5x5_reduce_outer_out_channel = DIV_CEIL(inception_4b_5x5_reduce_kernel_num, inception_4b_5x5_reduce_block_out_channel);//outer loop
//inception_4b_5x5
///configuration
const int inception_4b_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_4b_5x5_allocate_global_in_feature_num = 2;
const int inception_4b_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_4b_5x5_allocate_global_weight_5x5_num = 2;
const int inception_4b_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_4b_5x5_allocate_global_out_feature_num = 2;
const int inception_4b_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4b_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_4b_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_5x5_outer_in_channel = DIV_CEIL(inception_4b_5x5_in_channel, inception_4b_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_4b_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4b_5x5_outer_height = DIV_CEIL(inception_4b_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_4b_5x5_block_overlap_height);
const int inception_4b_5x5_outer_width = DIV_CEIL(inception_4b_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_4b_5x5_block_overlap_width);
///interval between blocks
const int inception_4b_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_5x5_in_height, inception_4b_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_4b_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_5x5_in_width, inception_4b_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_4b_5x5_block_in_height = inception_4b_5x5_block_interval_height+ inception_4b_5x5_block_overlap_height;
const int inception_4b_5x5_block_in_width = inception_4b_5x5_block_interval_height + inception_4b_5x5_block_overlap_height;
const int inception_4b_5x5_block_in_channel = MIN(inception_4b_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_4b_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_4b_5x5_block_out_channel = MIN(inception_4b_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4b_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4b_5x5_outer_out_channel = DIV_CEIL(inception_4b_5x5_kernel_num, inception_4b_5x5_block_out_channel);//outer loop
//inception_4b_pool
///configuration
const int inception_4b_pool_allocate_global_in_feature_start_idx = 0;
const int inception_4b_pool_allocate_global_in_feature_num = 4;
const int inception_4b_pool_allocate_global_out_feature_start_idx = 4;
const int inception_4b_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_4b_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_4b_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_pool_outer_in_channel = DIV_CEIL(inception_4b_pool_in_channel, inception_4b_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_4b_pool_outer_height = DIV_CEIL(inception_4b_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_4b_pool_block_overlap_height);
const int inception_4b_pool_outer_width = DIV_CEIL(inception_4b_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_4b_pool_block_overlap_width);
///interval between blocks
const int inception_4b_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_pool_in_height, inception_4b_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_4b_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_pool_in_width, inception_4b_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_4b_pool_block_in_height = inception_4b_pool_block_interval_height + inception_4b_pool_block_overlap_height;
const int inception_4b_pool_block_in_width = inception_4b_pool_block_interval_height + inception_4b_pool_block_overlap_height;
const int inception_4b_pool_block_in_channel = inception_4b_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_4b_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_4b_pool_block_out_channel = inception_4b_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_4b_pool_proj
///configuration
const int inception_4b_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_4b_pool_proj_allocate_global_in_feature_num = 2;
const int inception_4b_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_4b_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_4b_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_4b_pool_proj_allocate_global_out_feature_num = 2;
const int inception_4b_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4b_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4b_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4b_pool_proj_outer_in_channel = DIV_CEIL(inception_4b_pool_proj_in_channel, inception_4b_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4b_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_pool_proj_outer_height = DIV_CEIL(inception_4b_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_4b_pool_proj_block_overlap_height);
const int inception_4b_pool_proj_outer_width = DIV_CEIL(inception_4b_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_4b_pool_proj_block_overlap_width);
///interval between blocks
const int inception_4b_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4b_pool_proj_in_height, inception_4b_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4b_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4b_pool_proj_in_width, inception_4b_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4b_pool_proj_block_in_height = inception_4b_pool_proj_block_interval_height+ inception_4b_pool_proj_block_overlap_height;
const int inception_4b_pool_proj_block_in_width = inception_4b_pool_proj_block_interval_height + inception_4b_pool_proj_block_overlap_height;
const int inception_4b_pool_proj_block_in_channel = MIN(inception_4b_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4b_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4b_pool_proj_block_out_channel = MIN(inception_4b_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4b_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4b_pool_proj_outer_out_channel = DIV_CEIL(inception_4b_pool_proj_kernel_num, inception_4b_pool_proj_block_out_channel);//outer loop
//inception_4c_1x1
///configuration
const int inception_4c_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_4c_1x1_allocate_global_in_feature_num = 2;
const int inception_4c_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_4c_1x1_allocate_global_weight_1x1_num = 2;
const int inception_4c_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_4c_1x1_allocate_global_out_feature_num = 2;
const int inception_4c_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4c_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4c_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_1x1_outer_in_channel = DIV_CEIL(inception_4c_1x1_in_channel, inception_4c_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4c_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_1x1_outer_height = DIV_CEIL(inception_4c_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_4c_1x1_block_overlap_height);
const int inception_4c_1x1_outer_width = DIV_CEIL(inception_4c_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_4c_1x1_block_overlap_width);
///interval between blocks
const int inception_4c_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_1x1_in_height, inception_4c_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4c_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_1x1_in_width, inception_4c_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4c_1x1_block_in_height = inception_4c_1x1_block_interval_height+ inception_4c_1x1_block_overlap_height;
const int inception_4c_1x1_block_in_width = inception_4c_1x1_block_interval_height + inception_4c_1x1_block_overlap_height;
const int inception_4c_1x1_block_in_channel = MIN(inception_4c_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4c_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4c_1x1_block_out_channel = MIN(inception_4c_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4c_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_1x1_outer_out_channel = DIV_CEIL(inception_4c_1x1_kernel_num, inception_4c_1x1_block_out_channel);//outer loop
//inception_4c_3x3_reduce
///configuration
const int inception_4c_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4c_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_4c_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4c_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4c_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4c_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_4c_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4c_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4c_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_3x3_reduce_outer_in_channel = DIV_CEIL(inception_4c_3x3_reduce_in_channel, inception_4c_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4c_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_3x3_reduce_outer_height = DIV_CEIL(inception_4c_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4c_3x3_reduce_block_overlap_height);
const int inception_4c_3x3_reduce_outer_width = DIV_CEIL(inception_4c_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4c_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_4c_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_3x3_reduce_in_height, inception_4c_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4c_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_3x3_reduce_in_width, inception_4c_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4c_3x3_reduce_block_in_height = inception_4c_3x3_reduce_block_interval_height+ inception_4c_3x3_reduce_block_overlap_height;
const int inception_4c_3x3_reduce_block_in_width = inception_4c_3x3_reduce_block_interval_height + inception_4c_3x3_reduce_block_overlap_height;
const int inception_4c_3x3_reduce_block_in_channel = MIN(inception_4c_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4c_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4c_3x3_reduce_block_out_channel = MIN(inception_4c_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4c_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_3x3_reduce_outer_out_channel = DIV_CEIL(inception_4c_3x3_reduce_kernel_num, inception_4c_3x3_reduce_block_out_channel);//outer loop
//inception_4c_3x3
///configuration
const int inception_4c_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_4c_3x3_allocate_global_in_feature_num = 2;
const int inception_4c_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_4c_3x3_allocate_global_weight_3x3_num = 2;
const int inception_4c_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_4c_3x3_allocate_global_out_feature_num = 2;
const int inception_4c_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4c_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_4c_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_3x3_outer_in_channel = DIV_CEIL(inception_4c_3x3_in_channel, inception_4c_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_4c_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4c_3x3_outer_height = DIV_CEIL(inception_4c_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_4c_3x3_block_overlap_height);
const int inception_4c_3x3_outer_width = DIV_CEIL(inception_4c_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_4c_3x3_block_overlap_width);
///interval between blocks
const int inception_4c_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_3x3_in_height, inception_4c_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_4c_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_3x3_in_width, inception_4c_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_4c_3x3_block_in_height = inception_4c_3x3_block_interval_height+ inception_4c_3x3_block_overlap_height;
const int inception_4c_3x3_block_in_width = inception_4c_3x3_block_interval_height + inception_4c_3x3_block_overlap_height;
const int inception_4c_3x3_block_in_channel = MIN(inception_4c_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_4c_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_4c_3x3_block_out_channel = MIN(inception_4c_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4c_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4c_3x3_outer_out_channel = DIV_CEIL(inception_4c_3x3_kernel_num, inception_4c_3x3_block_out_channel);//outer loop
//inception_4c_5x5_reduce
///configuration
const int inception_4c_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4c_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_4c_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4c_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4c_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4c_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_4c_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4c_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4c_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_5x5_reduce_outer_in_channel = DIV_CEIL(inception_4c_5x5_reduce_in_channel, inception_4c_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4c_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_5x5_reduce_outer_height = DIV_CEIL(inception_4c_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4c_5x5_reduce_block_overlap_height);
const int inception_4c_5x5_reduce_outer_width = DIV_CEIL(inception_4c_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4c_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_4c_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_5x5_reduce_in_height, inception_4c_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4c_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_5x5_reduce_in_width, inception_4c_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4c_5x5_reduce_block_in_height = inception_4c_5x5_reduce_block_interval_height+ inception_4c_5x5_reduce_block_overlap_height;
const int inception_4c_5x5_reduce_block_in_width = inception_4c_5x5_reduce_block_interval_height + inception_4c_5x5_reduce_block_overlap_height;
const int inception_4c_5x5_reduce_block_in_channel = MIN(inception_4c_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4c_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4c_5x5_reduce_block_out_channel = MIN(inception_4c_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4c_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_5x5_reduce_outer_out_channel = DIV_CEIL(inception_4c_5x5_reduce_kernel_num, inception_4c_5x5_reduce_block_out_channel);//outer loop
//inception_4c_5x5
///configuration
const int inception_4c_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_4c_5x5_allocate_global_in_feature_num = 2;
const int inception_4c_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_4c_5x5_allocate_global_weight_5x5_num = 2;
const int inception_4c_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_4c_5x5_allocate_global_out_feature_num = 2;
const int inception_4c_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4c_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_4c_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_5x5_outer_in_channel = DIV_CEIL(inception_4c_5x5_in_channel, inception_4c_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_4c_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4c_5x5_outer_height = DIV_CEIL(inception_4c_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_4c_5x5_block_overlap_height);
const int inception_4c_5x5_outer_width = DIV_CEIL(inception_4c_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_4c_5x5_block_overlap_width);
///interval between blocks
const int inception_4c_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_5x5_in_height, inception_4c_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_4c_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_5x5_in_width, inception_4c_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_4c_5x5_block_in_height = inception_4c_5x5_block_interval_height+ inception_4c_5x5_block_overlap_height;
const int inception_4c_5x5_block_in_width = inception_4c_5x5_block_interval_height + inception_4c_5x5_block_overlap_height;
const int inception_4c_5x5_block_in_channel = MIN(inception_4c_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_4c_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_4c_5x5_block_out_channel = MIN(inception_4c_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4c_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4c_5x5_outer_out_channel = DIV_CEIL(inception_4c_5x5_kernel_num, inception_4c_5x5_block_out_channel);//outer loop
//inception_4c_pool
///configuration
const int inception_4c_pool_allocate_global_in_feature_start_idx = 0;
const int inception_4c_pool_allocate_global_in_feature_num = 4;
const int inception_4c_pool_allocate_global_out_feature_start_idx = 4;
const int inception_4c_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_4c_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_4c_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_pool_outer_in_channel = DIV_CEIL(inception_4c_pool_in_channel, inception_4c_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_4c_pool_outer_height = DIV_CEIL(inception_4c_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_4c_pool_block_overlap_height);
const int inception_4c_pool_outer_width = DIV_CEIL(inception_4c_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_4c_pool_block_overlap_width);
///interval between blocks
const int inception_4c_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_pool_in_height, inception_4c_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_4c_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_pool_in_width, inception_4c_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_4c_pool_block_in_height = inception_4c_pool_block_interval_height + inception_4c_pool_block_overlap_height;
const int inception_4c_pool_block_in_width = inception_4c_pool_block_interval_height + inception_4c_pool_block_overlap_height;
const int inception_4c_pool_block_in_channel = inception_4c_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_4c_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_4c_pool_block_out_channel = inception_4c_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_4c_pool_proj
///configuration
const int inception_4c_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_4c_pool_proj_allocate_global_in_feature_num = 2;
const int inception_4c_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_4c_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_4c_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_4c_pool_proj_allocate_global_out_feature_num = 2;
const int inception_4c_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4c_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4c_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4c_pool_proj_outer_in_channel = DIV_CEIL(inception_4c_pool_proj_in_channel, inception_4c_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4c_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_pool_proj_outer_height = DIV_CEIL(inception_4c_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_4c_pool_proj_block_overlap_height);
const int inception_4c_pool_proj_outer_width = DIV_CEIL(inception_4c_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_4c_pool_proj_block_overlap_width);
///interval between blocks
const int inception_4c_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4c_pool_proj_in_height, inception_4c_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4c_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4c_pool_proj_in_width, inception_4c_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4c_pool_proj_block_in_height = inception_4c_pool_proj_block_interval_height+ inception_4c_pool_proj_block_overlap_height;
const int inception_4c_pool_proj_block_in_width = inception_4c_pool_proj_block_interval_height + inception_4c_pool_proj_block_overlap_height;
const int inception_4c_pool_proj_block_in_channel = MIN(inception_4c_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4c_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4c_pool_proj_block_out_channel = MIN(inception_4c_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4c_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4c_pool_proj_outer_out_channel = DIV_CEIL(inception_4c_pool_proj_kernel_num, inception_4c_pool_proj_block_out_channel);//outer loop
//inception_4d_1x1
///configuration
const int inception_4d_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_4d_1x1_allocate_global_in_feature_num = 2;
const int inception_4d_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_4d_1x1_allocate_global_weight_1x1_num = 2;
const int inception_4d_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_4d_1x1_allocate_global_out_feature_num = 2;
const int inception_4d_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4d_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4d_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_1x1_outer_in_channel = DIV_CEIL(inception_4d_1x1_in_channel, inception_4d_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4d_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_1x1_outer_height = DIV_CEIL(inception_4d_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_4d_1x1_block_overlap_height);
const int inception_4d_1x1_outer_width = DIV_CEIL(inception_4d_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_4d_1x1_block_overlap_width);
///interval between blocks
const int inception_4d_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_1x1_in_height, inception_4d_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4d_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_1x1_in_width, inception_4d_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4d_1x1_block_in_height = inception_4d_1x1_block_interval_height+ inception_4d_1x1_block_overlap_height;
const int inception_4d_1x1_block_in_width = inception_4d_1x1_block_interval_height + inception_4d_1x1_block_overlap_height;
const int inception_4d_1x1_block_in_channel = MIN(inception_4d_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4d_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4d_1x1_block_out_channel = MIN(inception_4d_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4d_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_1x1_outer_out_channel = DIV_CEIL(inception_4d_1x1_kernel_num, inception_4d_1x1_block_out_channel);//outer loop
//inception_4d_3x3_reduce
///configuration
const int inception_4d_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4d_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_4d_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4d_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4d_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4d_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_4d_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4d_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4d_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_3x3_reduce_outer_in_channel = DIV_CEIL(inception_4d_3x3_reduce_in_channel, inception_4d_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4d_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_3x3_reduce_outer_height = DIV_CEIL(inception_4d_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4d_3x3_reduce_block_overlap_height);
const int inception_4d_3x3_reduce_outer_width = DIV_CEIL(inception_4d_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4d_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_4d_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_3x3_reduce_in_height, inception_4d_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4d_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_3x3_reduce_in_width, inception_4d_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4d_3x3_reduce_block_in_height = inception_4d_3x3_reduce_block_interval_height+ inception_4d_3x3_reduce_block_overlap_height;
const int inception_4d_3x3_reduce_block_in_width = inception_4d_3x3_reduce_block_interval_height + inception_4d_3x3_reduce_block_overlap_height;
const int inception_4d_3x3_reduce_block_in_channel = MIN(inception_4d_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4d_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4d_3x3_reduce_block_out_channel = MIN(inception_4d_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4d_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_3x3_reduce_outer_out_channel = DIV_CEIL(inception_4d_3x3_reduce_kernel_num, inception_4d_3x3_reduce_block_out_channel);//outer loop
//inception_4d_3x3
///configuration
const int inception_4d_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_4d_3x3_allocate_global_in_feature_num = 2;
const int inception_4d_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_4d_3x3_allocate_global_weight_3x3_num = 2;
const int inception_4d_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_4d_3x3_allocate_global_out_feature_num = 2;
const int inception_4d_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4d_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_4d_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_3x3_outer_in_channel = DIV_CEIL(inception_4d_3x3_in_channel, inception_4d_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_4d_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4d_3x3_outer_height = DIV_CEIL(inception_4d_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_4d_3x3_block_overlap_height);
const int inception_4d_3x3_outer_width = DIV_CEIL(inception_4d_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_4d_3x3_block_overlap_width);
///interval between blocks
const int inception_4d_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_3x3_in_height, inception_4d_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_4d_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_3x3_in_width, inception_4d_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_4d_3x3_block_in_height = inception_4d_3x3_block_interval_height+ inception_4d_3x3_block_overlap_height;
const int inception_4d_3x3_block_in_width = inception_4d_3x3_block_interval_height + inception_4d_3x3_block_overlap_height;
const int inception_4d_3x3_block_in_channel = MIN(inception_4d_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_4d_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_4d_3x3_block_out_channel = MIN(inception_4d_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4d_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4d_3x3_outer_out_channel = DIV_CEIL(inception_4d_3x3_kernel_num, inception_4d_3x3_block_out_channel);//outer loop
//inception_4d_5x5_reduce
///configuration
const int inception_4d_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4d_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_4d_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4d_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4d_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4d_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_4d_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4d_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4d_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_5x5_reduce_outer_in_channel = DIV_CEIL(inception_4d_5x5_reduce_in_channel, inception_4d_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4d_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_5x5_reduce_outer_height = DIV_CEIL(inception_4d_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4d_5x5_reduce_block_overlap_height);
const int inception_4d_5x5_reduce_outer_width = DIV_CEIL(inception_4d_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4d_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_4d_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_5x5_reduce_in_height, inception_4d_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4d_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_5x5_reduce_in_width, inception_4d_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4d_5x5_reduce_block_in_height = inception_4d_5x5_reduce_block_interval_height+ inception_4d_5x5_reduce_block_overlap_height;
const int inception_4d_5x5_reduce_block_in_width = inception_4d_5x5_reduce_block_interval_height + inception_4d_5x5_reduce_block_overlap_height;
const int inception_4d_5x5_reduce_block_in_channel = MIN(inception_4d_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4d_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4d_5x5_reduce_block_out_channel = MIN(inception_4d_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4d_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_5x5_reduce_outer_out_channel = DIV_CEIL(inception_4d_5x5_reduce_kernel_num, inception_4d_5x5_reduce_block_out_channel);//outer loop
//inception_4d_5x5
///configuration
const int inception_4d_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_4d_5x5_allocate_global_in_feature_num = 2;
const int inception_4d_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_4d_5x5_allocate_global_weight_5x5_num = 2;
const int inception_4d_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_4d_5x5_allocate_global_out_feature_num = 2;
const int inception_4d_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4d_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_4d_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_5x5_outer_in_channel = DIV_CEIL(inception_4d_5x5_in_channel, inception_4d_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_4d_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4d_5x5_outer_height = DIV_CEIL(inception_4d_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_4d_5x5_block_overlap_height);
const int inception_4d_5x5_outer_width = DIV_CEIL(inception_4d_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_4d_5x5_block_overlap_width);
///interval between blocks
const int inception_4d_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_5x5_in_height, inception_4d_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_4d_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_5x5_in_width, inception_4d_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_4d_5x5_block_in_height = inception_4d_5x5_block_interval_height+ inception_4d_5x5_block_overlap_height;
const int inception_4d_5x5_block_in_width = inception_4d_5x5_block_interval_height + inception_4d_5x5_block_overlap_height;
const int inception_4d_5x5_block_in_channel = MIN(inception_4d_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_4d_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_4d_5x5_block_out_channel = MIN(inception_4d_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4d_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4d_5x5_outer_out_channel = DIV_CEIL(inception_4d_5x5_kernel_num, inception_4d_5x5_block_out_channel);//outer loop
//inception_4d_pool
///configuration
const int inception_4d_pool_allocate_global_in_feature_start_idx = 0;
const int inception_4d_pool_allocate_global_in_feature_num = 4;
const int inception_4d_pool_allocate_global_out_feature_start_idx = 4;
const int inception_4d_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_4d_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_4d_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_pool_outer_in_channel = DIV_CEIL(inception_4d_pool_in_channel, inception_4d_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_4d_pool_outer_height = DIV_CEIL(inception_4d_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_4d_pool_block_overlap_height);
const int inception_4d_pool_outer_width = DIV_CEIL(inception_4d_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_4d_pool_block_overlap_width);
///interval between blocks
const int inception_4d_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_pool_in_height, inception_4d_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_4d_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_pool_in_width, inception_4d_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_4d_pool_block_in_height = inception_4d_pool_block_interval_height + inception_4d_pool_block_overlap_height;
const int inception_4d_pool_block_in_width = inception_4d_pool_block_interval_height + inception_4d_pool_block_overlap_height;
const int inception_4d_pool_block_in_channel = inception_4d_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_4d_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_4d_pool_block_out_channel = inception_4d_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_4d_pool_proj
///configuration
const int inception_4d_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_4d_pool_proj_allocate_global_in_feature_num = 2;
const int inception_4d_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_4d_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_4d_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_4d_pool_proj_allocate_global_out_feature_num = 2;
const int inception_4d_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4d_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4d_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4d_pool_proj_outer_in_channel = DIV_CEIL(inception_4d_pool_proj_in_channel, inception_4d_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4d_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_pool_proj_outer_height = DIV_CEIL(inception_4d_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_4d_pool_proj_block_overlap_height);
const int inception_4d_pool_proj_outer_width = DIV_CEIL(inception_4d_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_4d_pool_proj_block_overlap_width);
///interval between blocks
const int inception_4d_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4d_pool_proj_in_height, inception_4d_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4d_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4d_pool_proj_in_width, inception_4d_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4d_pool_proj_block_in_height = inception_4d_pool_proj_block_interval_height+ inception_4d_pool_proj_block_overlap_height;
const int inception_4d_pool_proj_block_in_width = inception_4d_pool_proj_block_interval_height + inception_4d_pool_proj_block_overlap_height;
const int inception_4d_pool_proj_block_in_channel = MIN(inception_4d_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4d_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4d_pool_proj_block_out_channel = MIN(inception_4d_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4d_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4d_pool_proj_outer_out_channel = DIV_CEIL(inception_4d_pool_proj_kernel_num, inception_4d_pool_proj_block_out_channel);//outer loop
//inception_4e_1x1
///configuration
const int inception_4e_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_4e_1x1_allocate_global_in_feature_num = 2;
const int inception_4e_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_4e_1x1_allocate_global_weight_1x1_num = 2;
const int inception_4e_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_4e_1x1_allocate_global_out_feature_num = 2;
const int inception_4e_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4e_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4e_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_1x1_outer_in_channel = DIV_CEIL(inception_4e_1x1_in_channel, inception_4e_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4e_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_1x1_outer_height = DIV_CEIL(inception_4e_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_4e_1x1_block_overlap_height);
const int inception_4e_1x1_outer_width = DIV_CEIL(inception_4e_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_4e_1x1_block_overlap_width);
///interval between blocks
const int inception_4e_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_1x1_in_height, inception_4e_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4e_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_1x1_in_width, inception_4e_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4e_1x1_block_in_height = inception_4e_1x1_block_interval_height+ inception_4e_1x1_block_overlap_height;
const int inception_4e_1x1_block_in_width = inception_4e_1x1_block_interval_height + inception_4e_1x1_block_overlap_height;
const int inception_4e_1x1_block_in_channel = MIN(inception_4e_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4e_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4e_1x1_block_out_channel = MIN(inception_4e_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4e_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_1x1_outer_out_channel = DIV_CEIL(inception_4e_1x1_kernel_num, inception_4e_1x1_block_out_channel);//outer loop
//inception_4e_3x3_reduce
///configuration
const int inception_4e_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4e_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_4e_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4e_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4e_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4e_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_4e_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4e_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4e_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_3x3_reduce_outer_in_channel = DIV_CEIL(inception_4e_3x3_reduce_in_channel, inception_4e_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4e_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_3x3_reduce_outer_height = DIV_CEIL(inception_4e_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4e_3x3_reduce_block_overlap_height);
const int inception_4e_3x3_reduce_outer_width = DIV_CEIL(inception_4e_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4e_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_4e_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_3x3_reduce_in_height, inception_4e_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4e_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_3x3_reduce_in_width, inception_4e_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4e_3x3_reduce_block_in_height = inception_4e_3x3_reduce_block_interval_height+ inception_4e_3x3_reduce_block_overlap_height;
const int inception_4e_3x3_reduce_block_in_width = inception_4e_3x3_reduce_block_interval_height + inception_4e_3x3_reduce_block_overlap_height;
const int inception_4e_3x3_reduce_block_in_channel = MIN(inception_4e_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4e_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4e_3x3_reduce_block_out_channel = MIN(inception_4e_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4e_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_3x3_reduce_outer_out_channel = DIV_CEIL(inception_4e_3x3_reduce_kernel_num, inception_4e_3x3_reduce_block_out_channel);//outer loop
//inception_4e_3x3
///configuration
const int inception_4e_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_4e_3x3_allocate_global_in_feature_num = 2;
const int inception_4e_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_4e_3x3_allocate_global_weight_3x3_num = 2;
const int inception_4e_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_4e_3x3_allocate_global_out_feature_num = 2;
const int inception_4e_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4e_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_4e_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_3x3_outer_in_channel = DIV_CEIL(inception_4e_3x3_in_channel, inception_4e_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_4e_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4e_3x3_outer_height = DIV_CEIL(inception_4e_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_4e_3x3_block_overlap_height);
const int inception_4e_3x3_outer_width = DIV_CEIL(inception_4e_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_4e_3x3_block_overlap_width);
///interval between blocks
const int inception_4e_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_3x3_in_height, inception_4e_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_4e_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_3x3_in_width, inception_4e_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_4e_3x3_block_in_height = inception_4e_3x3_block_interval_height+ inception_4e_3x3_block_overlap_height;
const int inception_4e_3x3_block_in_width = inception_4e_3x3_block_interval_height + inception_4e_3x3_block_overlap_height;
const int inception_4e_3x3_block_in_channel = MIN(inception_4e_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_4e_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_4e_3x3_block_out_channel = MIN(inception_4e_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4e_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_4e_3x3_outer_out_channel = DIV_CEIL(inception_4e_3x3_kernel_num, inception_4e_3x3_block_out_channel);//outer loop
//inception_4e_5x5_reduce
///configuration
const int inception_4e_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_4e_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_4e_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_4e_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_4e_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_4e_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_4e_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4e_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4e_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_5x5_reduce_outer_in_channel = DIV_CEIL(inception_4e_5x5_reduce_in_channel, inception_4e_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4e_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_5x5_reduce_outer_height = DIV_CEIL(inception_4e_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_4e_5x5_reduce_block_overlap_height);
const int inception_4e_5x5_reduce_outer_width = DIV_CEIL(inception_4e_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_4e_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_4e_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_5x5_reduce_in_height, inception_4e_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4e_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_5x5_reduce_in_width, inception_4e_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4e_5x5_reduce_block_in_height = inception_4e_5x5_reduce_block_interval_height+ inception_4e_5x5_reduce_block_overlap_height;
const int inception_4e_5x5_reduce_block_in_width = inception_4e_5x5_reduce_block_interval_height + inception_4e_5x5_reduce_block_overlap_height;
const int inception_4e_5x5_reduce_block_in_channel = MIN(inception_4e_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4e_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4e_5x5_reduce_block_out_channel = MIN(inception_4e_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4e_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_5x5_reduce_outer_out_channel = DIV_CEIL(inception_4e_5x5_reduce_kernel_num, inception_4e_5x5_reduce_block_out_channel);//outer loop
//inception_4e_5x5
///configuration
const int inception_4e_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_4e_5x5_allocate_global_in_feature_num = 2;
const int inception_4e_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_4e_5x5_allocate_global_weight_5x5_num = 2;
const int inception_4e_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_4e_5x5_allocate_global_out_feature_num = 2;
const int inception_4e_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4e_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_4e_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_5x5_outer_in_channel = DIV_CEIL(inception_4e_5x5_in_channel, inception_4e_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_4e_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4e_5x5_outer_height = DIV_CEIL(inception_4e_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_4e_5x5_block_overlap_height);
const int inception_4e_5x5_outer_width = DIV_CEIL(inception_4e_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_4e_5x5_block_overlap_width);
///interval between blocks
const int inception_4e_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_5x5_in_height, inception_4e_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_4e_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_5x5_in_width, inception_4e_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_4e_5x5_block_in_height = inception_4e_5x5_block_interval_height+ inception_4e_5x5_block_overlap_height;
const int inception_4e_5x5_block_in_width = inception_4e_5x5_block_interval_height + inception_4e_5x5_block_overlap_height;
const int inception_4e_5x5_block_in_channel = MIN(inception_4e_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_4e_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_4e_5x5_block_out_channel = MIN(inception_4e_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4e_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_4e_5x5_outer_out_channel = DIV_CEIL(inception_4e_5x5_kernel_num, inception_4e_5x5_block_out_channel);//outer loop
//inception_4e_pool
///configuration
const int inception_4e_pool_allocate_global_in_feature_start_idx = 0;
const int inception_4e_pool_allocate_global_in_feature_num = 4;
const int inception_4e_pool_allocate_global_out_feature_start_idx = 4;
const int inception_4e_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_4e_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_4e_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_pool_outer_in_channel = DIV_CEIL(inception_4e_pool_in_channel, inception_4e_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_4e_pool_outer_height = DIV_CEIL(inception_4e_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_4e_pool_block_overlap_height);
const int inception_4e_pool_outer_width = DIV_CEIL(inception_4e_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_4e_pool_block_overlap_width);
///interval between blocks
const int inception_4e_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_pool_in_height, inception_4e_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_4e_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_pool_in_width, inception_4e_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_4e_pool_block_in_height = inception_4e_pool_block_interval_height + inception_4e_pool_block_overlap_height;
const int inception_4e_pool_block_in_width = inception_4e_pool_block_interval_height + inception_4e_pool_block_overlap_height;
const int inception_4e_pool_block_in_channel = inception_4e_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_4e_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_4e_pool_block_out_channel = inception_4e_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_4e_pool_proj
///configuration
const int inception_4e_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_4e_pool_proj_allocate_global_in_feature_num = 2;
const int inception_4e_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_4e_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_4e_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_4e_pool_proj_allocate_global_out_feature_num = 2;
const int inception_4e_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_4e_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_4e_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_4e_pool_proj_outer_in_channel = DIV_CEIL(inception_4e_pool_proj_in_channel, inception_4e_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_4e_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_pool_proj_outer_height = DIV_CEIL(inception_4e_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_4e_pool_proj_block_overlap_height);
const int inception_4e_pool_proj_outer_width = DIV_CEIL(inception_4e_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_4e_pool_proj_block_overlap_width);
///interval between blocks
const int inception_4e_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_4e_pool_proj_in_height, inception_4e_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_4e_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_4e_pool_proj_in_width, inception_4e_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_4e_pool_proj_block_in_height = inception_4e_pool_proj_block_interval_height+ inception_4e_pool_proj_block_overlap_height;
const int inception_4e_pool_proj_block_in_width = inception_4e_pool_proj_block_interval_height + inception_4e_pool_proj_block_overlap_height;
const int inception_4e_pool_proj_block_in_channel = MIN(inception_4e_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_4e_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_4e_pool_proj_block_out_channel = MIN(inception_4e_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_4e_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_4e_pool_proj_outer_out_channel = DIV_CEIL(inception_4e_pool_proj_kernel_num, inception_4e_pool_proj_block_out_channel);//outer loop
//pool4_3x3_s2
///configuration
const int pool4_3x3_s2_allocate_global_in_feature_start_idx = 0;
const int pool4_3x3_s2_allocate_global_in_feature_num = 4;
const int pool4_3x3_s2_allocate_global_out_feature_start_idx = 4;
const int pool4_3x3_s2_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int pool4_3x3_s2_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S2 - 1;
const int pool4_3x3_s2_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S2 - 1;
///number of blocks(the dims of the outer loop)
const int pool4_3x3_s2_outer_in_channel = DIV_CEIL(pool4_3x3_s2_in_channel, pool4_3x3_s2_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int pool4_3x3_s2_outer_height = DIV_CEIL(pool4_3x3_s2_in_height, HEIGHT_FEATURE_GLOBAL - pool4_3x3_s2_block_overlap_height);
const int pool4_3x3_s2_outer_width = DIV_CEIL(pool4_3x3_s2_in_width, WIDTH_FEATURE_GLOBAL - pool4_3x3_s2_block_overlap_width);
///interval between blocks
const int pool4_3x3_s2_block_interval_height = DIV_CEIL(DIV_CEIL(pool4_3x3_s2_in_height, pool4_3x3_s2_outer_height), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;//the spacing between blocks
const int pool4_3x3_s2_block_interval_width = DIV_CEIL(DIV_CEIL(pool4_3x3_s2_in_width, pool4_3x3_s2_outer_width), STRIDE_MAXPOOL3x3_S2)*STRIDE_MAXPOOL3x3_S2;
///dim of blocks
const int pool4_3x3_s2_block_in_height = pool4_3x3_s2_block_interval_height + pool4_3x3_s2_block_overlap_height;
const int pool4_3x3_s2_block_in_width = pool4_3x3_s2_block_interval_height + pool4_3x3_s2_block_overlap_height;
const int pool4_3x3_s2_block_in_channel = pool4_3x3_s2_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool4_3x3_s2_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S2;
///dim of kernels
const int pool4_3x3_s2_block_out_channel = pool4_3x3_s2_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_5a_1x1
///configuration
const int inception_5a_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_5a_1x1_allocate_global_in_feature_num = 2;
const int inception_5a_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_5a_1x1_allocate_global_weight_1x1_num = 2;
const int inception_5a_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_5a_1x1_allocate_global_out_feature_num = 2;
const int inception_5a_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5a_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5a_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_1x1_outer_in_channel = DIV_CEIL(inception_5a_1x1_in_channel, inception_5a_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5a_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_1x1_outer_height = DIV_CEIL(inception_5a_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_5a_1x1_block_overlap_height);
const int inception_5a_1x1_outer_width = DIV_CEIL(inception_5a_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_5a_1x1_block_overlap_width);
///interval between blocks
const int inception_5a_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_1x1_in_height, inception_5a_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5a_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_1x1_in_width, inception_5a_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5a_1x1_block_in_height = inception_5a_1x1_block_interval_height+ inception_5a_1x1_block_overlap_height;
const int inception_5a_1x1_block_in_width = inception_5a_1x1_block_interval_height + inception_5a_1x1_block_overlap_height;
const int inception_5a_1x1_block_in_channel = MIN(inception_5a_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5a_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5a_1x1_block_out_channel = MIN(inception_5a_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5a_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_1x1_outer_out_channel = DIV_CEIL(inception_5a_1x1_kernel_num, inception_5a_1x1_block_out_channel);//outer loop
//inception_5a_3x3_reduce
///configuration
const int inception_5a_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_5a_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_5a_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_5a_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_5a_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_5a_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_5a_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5a_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5a_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_3x3_reduce_outer_in_channel = DIV_CEIL(inception_5a_3x3_reduce_in_channel, inception_5a_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5a_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_3x3_reduce_outer_height = DIV_CEIL(inception_5a_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_5a_3x3_reduce_block_overlap_height);
const int inception_5a_3x3_reduce_outer_width = DIV_CEIL(inception_5a_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_5a_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_5a_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_3x3_reduce_in_height, inception_5a_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5a_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_3x3_reduce_in_width, inception_5a_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5a_3x3_reduce_block_in_height = inception_5a_3x3_reduce_block_interval_height+ inception_5a_3x3_reduce_block_overlap_height;
const int inception_5a_3x3_reduce_block_in_width = inception_5a_3x3_reduce_block_interval_height + inception_5a_3x3_reduce_block_overlap_height;
const int inception_5a_3x3_reduce_block_in_channel = MIN(inception_5a_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5a_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5a_3x3_reduce_block_out_channel = MIN(inception_5a_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5a_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_3x3_reduce_outer_out_channel = DIV_CEIL(inception_5a_3x3_reduce_kernel_num, inception_5a_3x3_reduce_block_out_channel);//outer loop
//inception_5a_3x3
///configuration
const int inception_5a_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_5a_3x3_allocate_global_in_feature_num = 2;
const int inception_5a_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_5a_3x3_allocate_global_weight_3x3_num = 2;
const int inception_5a_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_5a_3x3_allocate_global_out_feature_num = 2;
const int inception_5a_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5a_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_5a_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_3x3_outer_in_channel = DIV_CEIL(inception_5a_3x3_in_channel, inception_5a_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_5a_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_5a_3x3_outer_height = DIV_CEIL(inception_5a_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_5a_3x3_block_overlap_height);
const int inception_5a_3x3_outer_width = DIV_CEIL(inception_5a_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_5a_3x3_block_overlap_width);
///interval between blocks
const int inception_5a_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_3x3_in_height, inception_5a_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_5a_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_3x3_in_width, inception_5a_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_5a_3x3_block_in_height = inception_5a_3x3_block_interval_height+ inception_5a_3x3_block_overlap_height;
const int inception_5a_3x3_block_in_width = inception_5a_3x3_block_interval_height + inception_5a_3x3_block_overlap_height;
const int inception_5a_3x3_block_in_channel = MIN(inception_5a_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_5a_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_5a_3x3_block_out_channel = MIN(inception_5a_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5a_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_5a_3x3_outer_out_channel = DIV_CEIL(inception_5a_3x3_kernel_num, inception_5a_3x3_block_out_channel);//outer loop
//inception_5a_5x5_reduce
///configuration
const int inception_5a_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_5a_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_5a_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_5a_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_5a_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_5a_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_5a_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5a_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5a_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_5x5_reduce_outer_in_channel = DIV_CEIL(inception_5a_5x5_reduce_in_channel, inception_5a_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5a_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_5x5_reduce_outer_height = DIV_CEIL(inception_5a_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_5a_5x5_reduce_block_overlap_height);
const int inception_5a_5x5_reduce_outer_width = DIV_CEIL(inception_5a_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_5a_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_5a_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_5x5_reduce_in_height, inception_5a_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5a_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_5x5_reduce_in_width, inception_5a_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5a_5x5_reduce_block_in_height = inception_5a_5x5_reduce_block_interval_height+ inception_5a_5x5_reduce_block_overlap_height;
const int inception_5a_5x5_reduce_block_in_width = inception_5a_5x5_reduce_block_interval_height + inception_5a_5x5_reduce_block_overlap_height;
const int inception_5a_5x5_reduce_block_in_channel = MIN(inception_5a_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5a_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5a_5x5_reduce_block_out_channel = MIN(inception_5a_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5a_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_5x5_reduce_outer_out_channel = DIV_CEIL(inception_5a_5x5_reduce_kernel_num, inception_5a_5x5_reduce_block_out_channel);//outer loop
//inception_5a_5x5
///configuration
const int inception_5a_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_5a_5x5_allocate_global_in_feature_num = 2;
const int inception_5a_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_5a_5x5_allocate_global_weight_5x5_num = 2;
const int inception_5a_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_5a_5x5_allocate_global_out_feature_num = 2;
const int inception_5a_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5a_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_5a_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_5x5_outer_in_channel = DIV_CEIL(inception_5a_5x5_in_channel, inception_5a_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_5a_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_5a_5x5_outer_height = DIV_CEIL(inception_5a_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_5a_5x5_block_overlap_height);
const int inception_5a_5x5_outer_width = DIV_CEIL(inception_5a_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_5a_5x5_block_overlap_width);
///interval between blocks
const int inception_5a_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_5x5_in_height, inception_5a_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_5a_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_5x5_in_width, inception_5a_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_5a_5x5_block_in_height = inception_5a_5x5_block_interval_height+ inception_5a_5x5_block_overlap_height;
const int inception_5a_5x5_block_in_width = inception_5a_5x5_block_interval_height + inception_5a_5x5_block_overlap_height;
const int inception_5a_5x5_block_in_channel = MIN(inception_5a_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_5a_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_5a_5x5_block_out_channel = MIN(inception_5a_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5a_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_5a_5x5_outer_out_channel = DIV_CEIL(inception_5a_5x5_kernel_num, inception_5a_5x5_block_out_channel);//outer loop
//inception_5a_pool
///configuration
const int inception_5a_pool_allocate_global_in_feature_start_idx = 0;
const int inception_5a_pool_allocate_global_in_feature_num = 4;
const int inception_5a_pool_allocate_global_out_feature_start_idx = 4;
const int inception_5a_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_5a_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_5a_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_pool_outer_in_channel = DIV_CEIL(inception_5a_pool_in_channel, inception_5a_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_5a_pool_outer_height = DIV_CEIL(inception_5a_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_5a_pool_block_overlap_height);
const int inception_5a_pool_outer_width = DIV_CEIL(inception_5a_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_5a_pool_block_overlap_width);
///interval between blocks
const int inception_5a_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_pool_in_height, inception_5a_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_5a_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_pool_in_width, inception_5a_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_5a_pool_block_in_height = inception_5a_pool_block_interval_height + inception_5a_pool_block_overlap_height;
const int inception_5a_pool_block_in_width = inception_5a_pool_block_interval_height + inception_5a_pool_block_overlap_height;
const int inception_5a_pool_block_in_channel = inception_5a_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_5a_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_5a_pool_block_out_channel = inception_5a_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_5a_pool_proj
///configuration
const int inception_5a_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_5a_pool_proj_allocate_global_in_feature_num = 2;
const int inception_5a_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_5a_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_5a_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_5a_pool_proj_allocate_global_out_feature_num = 2;
const int inception_5a_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5a_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5a_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5a_pool_proj_outer_in_channel = DIV_CEIL(inception_5a_pool_proj_in_channel, inception_5a_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5a_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_pool_proj_outer_height = DIV_CEIL(inception_5a_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_5a_pool_proj_block_overlap_height);
const int inception_5a_pool_proj_outer_width = DIV_CEIL(inception_5a_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_5a_pool_proj_block_overlap_width);
///interval between blocks
const int inception_5a_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5a_pool_proj_in_height, inception_5a_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5a_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5a_pool_proj_in_width, inception_5a_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5a_pool_proj_block_in_height = inception_5a_pool_proj_block_interval_height+ inception_5a_pool_proj_block_overlap_height;
const int inception_5a_pool_proj_block_in_width = inception_5a_pool_proj_block_interval_height + inception_5a_pool_proj_block_overlap_height;
const int inception_5a_pool_proj_block_in_channel = MIN(inception_5a_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5a_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5a_pool_proj_block_out_channel = MIN(inception_5a_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5a_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5a_pool_proj_outer_out_channel = DIV_CEIL(inception_5a_pool_proj_kernel_num, inception_5a_pool_proj_block_out_channel);//outer loop
//inception_5b_1x1
///configuration
const int inception_5b_1x1_allocate_global_in_feature_start_idx = 0;
const int inception_5b_1x1_allocate_global_in_feature_num = 2;
const int inception_5b_1x1_allocate_global_weight_1x1_start_idx = 0;
const int inception_5b_1x1_allocate_global_weight_1x1_num = 2;
const int inception_5b_1x1_allocate_global_out_feature_start_idx = 2;
const int inception_5b_1x1_allocate_global_out_feature_num = 2;
const int inception_5b_1x1_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5b_1x1_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5b_1x1_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_1x1_outer_in_channel = DIV_CEIL(inception_5b_1x1_in_channel, inception_5b_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5b_1x1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_1x1_outer_height = DIV_CEIL(inception_5b_1x1_in_height, HEIGHT_FEATURE_GLOBAL- inception_5b_1x1_block_overlap_height);
const int inception_5b_1x1_outer_width = DIV_CEIL(inception_5b_1x1_in_width , WIDTH_FEATURE_GLOBAL - inception_5b_1x1_block_overlap_width);
///interval between blocks
const int inception_5b_1x1_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_1x1_in_height, inception_5b_1x1_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5b_1x1_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_1x1_in_width, inception_5b_1x1_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5b_1x1_block_in_height = inception_5b_1x1_block_interval_height+ inception_5b_1x1_block_overlap_height;
const int inception_5b_1x1_block_in_width = inception_5b_1x1_block_interval_height + inception_5b_1x1_block_overlap_height;
const int inception_5b_1x1_block_in_channel = MIN(inception_5b_1x1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5b_1x1_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5b_1x1_block_out_channel = MIN(inception_5b_1x1_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5b_1x1_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_1x1_outer_out_channel = DIV_CEIL(inception_5b_1x1_kernel_num, inception_5b_1x1_block_out_channel);//outer loop
//inception_5b_3x3_reduce
///configuration
const int inception_5b_3x3_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_5b_3x3_reduce_allocate_global_in_feature_num = 2;
const int inception_5b_3x3_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_5b_3x3_reduce_allocate_global_weight_1x1_num = 2;
const int inception_5b_3x3_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_5b_3x3_reduce_allocate_global_out_feature_num = 2;
const int inception_5b_3x3_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5b_3x3_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5b_3x3_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_3x3_reduce_outer_in_channel = DIV_CEIL(inception_5b_3x3_reduce_in_channel, inception_5b_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5b_3x3_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_3x3_reduce_outer_height = DIV_CEIL(inception_5b_3x3_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_5b_3x3_reduce_block_overlap_height);
const int inception_5b_3x3_reduce_outer_width = DIV_CEIL(inception_5b_3x3_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_5b_3x3_reduce_block_overlap_width);
///interval between blocks
const int inception_5b_3x3_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_3x3_reduce_in_height, inception_5b_3x3_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5b_3x3_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_3x3_reduce_in_width, inception_5b_3x3_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5b_3x3_reduce_block_in_height = inception_5b_3x3_reduce_block_interval_height+ inception_5b_3x3_reduce_block_overlap_height;
const int inception_5b_3x3_reduce_block_in_width = inception_5b_3x3_reduce_block_interval_height + inception_5b_3x3_reduce_block_overlap_height;
const int inception_5b_3x3_reduce_block_in_channel = MIN(inception_5b_3x3_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5b_3x3_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5b_3x3_reduce_block_out_channel = MIN(inception_5b_3x3_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5b_3x3_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_3x3_reduce_outer_out_channel = DIV_CEIL(inception_5b_3x3_reduce_kernel_num, inception_5b_3x3_reduce_block_out_channel);//outer loop
//inception_5b_3x3
///configuration
const int inception_5b_3x3_allocate_global_in_feature_start_idx = 0;
const int inception_5b_3x3_allocate_global_in_feature_num = 2;
const int inception_5b_3x3_allocate_global_weight_3x3_start_idx = 0;
const int inception_5b_3x3_allocate_global_weight_3x3_num = 2;
const int inception_5b_3x3_allocate_global_out_feature_start_idx = 2;
const int inception_5b_3x3_allocate_global_out_feature_num = 2;
const int inception_5b_3x3_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5b_3x3_block_overlap_height = KERNEL_HEIGHT_CONV3x3_S1 - 1;
const int inception_5b_3x3_block_overlap_width = KERNEL_WIDTH_CONV3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_3x3_outer_in_channel = DIV_CEIL(inception_5b_3x3_in_channel, inception_5b_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_3x3 ? inception_5b_3x3_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_5b_3x3_outer_height = DIV_CEIL(inception_5b_3x3_in_height, HEIGHT_FEATURE_GLOBAL- inception_5b_3x3_block_overlap_height);
const int inception_5b_3x3_outer_width = DIV_CEIL(inception_5b_3x3_in_width , WIDTH_FEATURE_GLOBAL - inception_5b_3x3_block_overlap_width);
///interval between blocks
const int inception_5b_3x3_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_3x3_in_height, inception_5b_3x3_outer_height), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;//the spacing between blocks
const int inception_5b_3x3_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_3x3_in_width, inception_5b_3x3_outer_width), STRIDE_CONV3x3_S1)*STRIDE_CONV3x3_S1;
///dim of blocks
const int inception_5b_3x3_block_in_height = inception_5b_3x3_block_interval_height+ inception_5b_3x3_block_overlap_height;
const int inception_5b_3x3_block_in_width = inception_5b_3x3_block_interval_height + inception_5b_3x3_block_overlap_height;
const int inception_5b_3x3_block_in_channel = MIN(inception_5b_3x3_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_3x3);
///set parallism
const int inception_5b_3x3_inner_pe_parallel = NUM_PE_CONV3x3_S1;
///dim of kernels
const int inception_5b_3x3_block_out_channel = MIN(inception_5b_3x3_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5b_3x3_allocate_global_weight_3x3_num*OUT_CHANNEL_WEIGHT_GLOBAL_3x3);
const int inception_5b_3x3_outer_out_channel = DIV_CEIL(inception_5b_3x3_kernel_num, inception_5b_3x3_block_out_channel);//outer loop
//inception_5b_5x5_reduce
///configuration
const int inception_5b_5x5_reduce_allocate_global_in_feature_start_idx = 0;
const int inception_5b_5x5_reduce_allocate_global_in_feature_num = 2;
const int inception_5b_5x5_reduce_allocate_global_weight_1x1_start_idx = 0;
const int inception_5b_5x5_reduce_allocate_global_weight_1x1_num = 2;
const int inception_5b_5x5_reduce_allocate_global_out_feature_start_idx = 2;
const int inception_5b_5x5_reduce_allocate_global_out_feature_num = 2;
const int inception_5b_5x5_reduce_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5b_5x5_reduce_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5b_5x5_reduce_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_5x5_reduce_outer_in_channel = DIV_CEIL(inception_5b_5x5_reduce_in_channel, inception_5b_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5b_5x5_reduce_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_5x5_reduce_outer_height = DIV_CEIL(inception_5b_5x5_reduce_in_height, HEIGHT_FEATURE_GLOBAL- inception_5b_5x5_reduce_block_overlap_height);
const int inception_5b_5x5_reduce_outer_width = DIV_CEIL(inception_5b_5x5_reduce_in_width , WIDTH_FEATURE_GLOBAL - inception_5b_5x5_reduce_block_overlap_width);
///interval between blocks
const int inception_5b_5x5_reduce_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_5x5_reduce_in_height, inception_5b_5x5_reduce_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5b_5x5_reduce_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_5x5_reduce_in_width, inception_5b_5x5_reduce_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5b_5x5_reduce_block_in_height = inception_5b_5x5_reduce_block_interval_height+ inception_5b_5x5_reduce_block_overlap_height;
const int inception_5b_5x5_reduce_block_in_width = inception_5b_5x5_reduce_block_interval_height + inception_5b_5x5_reduce_block_overlap_height;
const int inception_5b_5x5_reduce_block_in_channel = MIN(inception_5b_5x5_reduce_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5b_5x5_reduce_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5b_5x5_reduce_block_out_channel = MIN(inception_5b_5x5_reduce_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5b_5x5_reduce_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_5x5_reduce_outer_out_channel = DIV_CEIL(inception_5b_5x5_reduce_kernel_num, inception_5b_5x5_reduce_block_out_channel);//outer loop
//inception_5b_5x5
///configuration
const int inception_5b_5x5_allocate_global_in_feature_start_idx = 0;
const int inception_5b_5x5_allocate_global_in_feature_num = 2;
const int inception_5b_5x5_allocate_global_weight_5x5_start_idx = 0;
const int inception_5b_5x5_allocate_global_weight_5x5_num = 2;
const int inception_5b_5x5_allocate_global_out_feature_start_idx = 2;
const int inception_5b_5x5_allocate_global_out_feature_num = 2;
const int inception_5b_5x5_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5b_5x5_block_overlap_height = KERNEL_HEIGHT_CONV5x5_S1 - 1;
const int inception_5b_5x5_block_overlap_width = KERNEL_WIDTH_CONV5x5_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_5x5_outer_in_channel = DIV_CEIL(inception_5b_5x5_in_channel, inception_5b_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_5x5 ? inception_5b_5x5_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_5b_5x5_outer_height = DIV_CEIL(inception_5b_5x5_in_height, HEIGHT_FEATURE_GLOBAL- inception_5b_5x5_block_overlap_height);
const int inception_5b_5x5_outer_width = DIV_CEIL(inception_5b_5x5_in_width , WIDTH_FEATURE_GLOBAL - inception_5b_5x5_block_overlap_width);
///interval between blocks
const int inception_5b_5x5_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_5x5_in_height, inception_5b_5x5_outer_height), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;//the spacing between blocks
const int inception_5b_5x5_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_5x5_in_width, inception_5b_5x5_outer_width), STRIDE_CONV5x5_S1)*STRIDE_CONV5x5_S1;
///dim of blocks
const int inception_5b_5x5_block_in_height = inception_5b_5x5_block_interval_height+ inception_5b_5x5_block_overlap_height;
const int inception_5b_5x5_block_in_width = inception_5b_5x5_block_interval_height + inception_5b_5x5_block_overlap_height;
const int inception_5b_5x5_block_in_channel = MIN(inception_5b_5x5_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_5x5);
///set parallism
const int inception_5b_5x5_inner_pe_parallel = NUM_PE_CONV5x5_S1;
///dim of kernels
const int inception_5b_5x5_block_out_channel = MIN(inception_5b_5x5_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5b_5x5_allocate_global_weight_5x5_num*OUT_CHANNEL_WEIGHT_GLOBAL_5x5);
const int inception_5b_5x5_outer_out_channel = DIV_CEIL(inception_5b_5x5_kernel_num, inception_5b_5x5_block_out_channel);//outer loop
//inception_5b_pool
///configuration
const int inception_5b_pool_allocate_global_in_feature_start_idx = 0;
const int inception_5b_pool_allocate_global_in_feature_num = 4;
const int inception_5b_pool_allocate_global_out_feature_start_idx = 4;
const int inception_5b_pool_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int inception_5b_pool_block_overlap_height = KERNEL_HEIGHT_MAXPOOL3x3_S1 - 1;
const int inception_5b_pool_block_overlap_width = KERNEL_WIDTH_MAXPOOL3x3_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_pool_outer_in_channel = DIV_CEIL(inception_5b_pool_in_channel, inception_5b_pool_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int inception_5b_pool_outer_height = DIV_CEIL(inception_5b_pool_in_height, HEIGHT_FEATURE_GLOBAL - inception_5b_pool_block_overlap_height);
const int inception_5b_pool_outer_width = DIV_CEIL(inception_5b_pool_in_width, WIDTH_FEATURE_GLOBAL - inception_5b_pool_block_overlap_width);
///interval between blocks
const int inception_5b_pool_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_pool_in_height, inception_5b_pool_outer_height), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;//the spacing between blocks
const int inception_5b_pool_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_pool_in_width, inception_5b_pool_outer_width), STRIDE_MAXPOOL3x3_S1)*STRIDE_MAXPOOL3x3_S1;
///dim of blocks
const int inception_5b_pool_block_in_height = inception_5b_pool_block_interval_height + inception_5b_pool_block_overlap_height;
const int inception_5b_pool_block_in_width = inception_5b_pool_block_interval_height + inception_5b_pool_block_overlap_height;
const int inception_5b_pool_block_in_channel = inception_5b_pool_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int inception_5b_pool_inner_pe_parallel = NUM_PE_MAXPOOL3x3_S1;
///dim of kernels
const int inception_5b_pool_block_out_channel = inception_5b_pool_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;
//inception_5b_pool_proj
///configuration
const int inception_5b_pool_proj_allocate_global_in_feature_start_idx = 0;
const int inception_5b_pool_proj_allocate_global_in_feature_num = 2;
const int inception_5b_pool_proj_allocate_global_weight_1x1_start_idx = 0;
const int inception_5b_pool_proj_allocate_global_weight_1x1_num = 2;
const int inception_5b_pool_proj_allocate_global_out_feature_start_idx = 2;
const int inception_5b_pool_proj_allocate_global_out_feature_num = 2;
const int inception_5b_pool_proj_allocate_bias_start_idx = 0;
///overlapped features between blocks
const int inception_5b_pool_proj_block_overlap_height = KERNEL_HEIGHT_CONV1x1_S1 - 1;
const int inception_5b_pool_proj_block_overlap_width = KERNEL_WIDTH_CONV1x1_S1 - 1;
///number of blocks(the dims of the outer loop)
const int inception_5b_pool_proj_outer_in_channel = DIV_CEIL(inception_5b_pool_proj_in_channel, inception_5b_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL < IN_CHANNEL_WEIGHT_GLOBAL_1x1 ? inception_5b_pool_proj_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL : IN_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_pool_proj_outer_height = DIV_CEIL(inception_5b_pool_proj_in_height, HEIGHT_FEATURE_GLOBAL- inception_5b_pool_proj_block_overlap_height);
const int inception_5b_pool_proj_outer_width = DIV_CEIL(inception_5b_pool_proj_in_width , WIDTH_FEATURE_GLOBAL - inception_5b_pool_proj_block_overlap_width);
///interval between blocks
const int inception_5b_pool_proj_block_interval_height = DIV_CEIL(DIV_CEIL(inception_5b_pool_proj_in_height, inception_5b_pool_proj_outer_height), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;//the spacing between blocks
const int inception_5b_pool_proj_block_interval_width = DIV_CEIL(DIV_CEIL(inception_5b_pool_proj_in_width, inception_5b_pool_proj_outer_width), STRIDE_CONV1x1_S1)*STRIDE_CONV1x1_S1;
///dim of blocks
const int inception_5b_pool_proj_block_in_height = inception_5b_pool_proj_block_interval_height+ inception_5b_pool_proj_block_overlap_height;
const int inception_5b_pool_proj_block_in_width = inception_5b_pool_proj_block_interval_height + inception_5b_pool_proj_block_overlap_height;
const int inception_5b_pool_proj_block_in_channel = MIN(inception_5b_pool_proj_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL , IN_CHANNEL_WEIGHT_GLOBAL_1x1);
///set parallism
const int inception_5b_pool_proj_inner_pe_parallel = NUM_PE_CONV1x1_S1;
///dim of kernels
const int inception_5b_pool_proj_block_out_channel = MIN(inception_5b_pool_proj_allocate_global_out_feature_num*CHANNEL_FEATURE_GLOBAL , inception_5b_pool_proj_allocate_global_weight_1x1_num*OUT_CHANNEL_WEIGHT_GLOBAL_1x1);
const int inception_5b_pool_proj_outer_out_channel = DIV_CEIL(inception_5b_pool_proj_kernel_num, inception_5b_pool_proj_block_out_channel);//outer loop
//pool5_7x7_s1
///configuration
const int pool5_7x7_s1_allocate_global_in_feature_start_idx = 0;
const int pool5_7x7_s1_allocate_global_in_feature_num = 4;
const int pool5_7x7_s1_allocate_global_out_feature_start_idx = 4;
const int pool5_7x7_s1_allocate_global_out_feature_num = 4;
///overlapped features between blocks
const int pool5_7x7_s1_block_overlap_height = KERNEL_HEIGHT_AVGPOOL7x7_S1 - 1;
const int pool5_7x7_s1_block_overlap_width = KERNEL_WIDTH_AVGPOOL7x7_S1 - 1;
///number of blocks(the dims of the outer loop)
const int pool5_7x7_s1_outer_in_channel = DIV_CEIL(pool5_7x7_s1_in_channel, pool5_7x7_s1_allocate_global_in_feature_num*CHANNEL_FEATURE_GLOBAL);
const int pool5_7x7_s1_outer_height = DIV_CEIL(pool5_7x7_s1_in_height, HEIGHT_FEATURE_GLOBAL - pool5_7x7_s1_block_overlap_height);
const int pool5_7x7_s1_outer_width = DIV_CEIL(pool5_7x7_s1_in_width, WIDTH_FEATURE_GLOBAL - pool5_7x7_s1_block_overlap_width);
///interval between blocks
const int pool5_7x7_s1_block_interval_height = DIV_CEIL(DIV_CEIL(pool5_7x7_s1_in_height, pool5_7x7_s1_outer_height), STRIDE_AVGPOOL7x7_S1)*STRIDE_AVGPOOL7x7_S1;//the spacing between blocks
const int pool5_7x7_s1_block_interval_width = DIV_CEIL(DIV_CEIL(pool5_7x7_s1_in_width, pool5_7x7_s1_outer_width), STRIDE_AVGPOOL7x7_S1)*STRIDE_AVGPOOL7x7_S1;
///dim of blocks
const int pool5_7x7_s1_block_in_height = pool5_7x7_s1_block_interval_height + pool5_7x7_s1_block_overlap_height;
const int pool5_7x7_s1_block_in_width = pool5_7x7_s1_block_interval_height + pool5_7x7_s1_block_overlap_height;
const int pool5_7x7_s1_block_in_channel = pool5_7x7_s1_allocate_global_in_feature_num * CHANNEL_FEATURE_GLOBAL;
///set parallism
const int pool5_7x7_s1_inner_pe_parallel = NUM_PE_AVGPOOL7x7_S1;
///dim of kernels
const int pool5_7x7_s1_block_out_channel = pool5_7x7_s1_allocate_global_out_feature_num * CHANNEL_FEATURE_GLOBAL;


#endif