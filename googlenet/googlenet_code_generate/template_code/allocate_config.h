#ifndef ALLOCATE_CONFIG_H_
#define ALLOCATE_CONFIG_H_

#define CALC_BLOCK_H_W(shape,num_block,kernel_shape,stride) ((DIV_CEIL(DIV_CEIL(shape,num_block),stride)-1)*(stride)+kernel_shape)

#include "header.h"
/////allocate_config_insert/////


#endif