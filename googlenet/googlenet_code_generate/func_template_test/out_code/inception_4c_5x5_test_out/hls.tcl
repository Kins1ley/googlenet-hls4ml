#################################
# @author	Qi SUN, CSE, CUHK   #
# @date		Feb. 2020           #
#################################
open_project -reset hls_project
add_files ./allocate_config.h
add_files ./googlenet.h
add_files ./nnet_batchnorm.h
add_files ./nnet_bias.h
add_files ./nnet_buffer.h
add_files ./nnet_common.h
add_files ./nnet_conv_input_reuse.h
add_files ./nnet_conv_output_reuse.h
add_files ./nnet_linear.h
add_files ./nnet_lrn.h
add_files ./nnet_mac.h
add_files ./nnet_pooling.h
add_files ./nnet_relu.h
add_files ./template_config.h
add_files ./googlenet_conv.cpp
add_files -tb ./googlenet_conv.cpp
set_top googlenet
open_solution -reset solution
set_part {xc7z020clg400-1} -tool vivado
create_clock -period 10
csynth_design
# export_design -format ip_catalog
cosim_design

exit
