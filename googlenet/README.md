# HLS4ML-googlenet
## conv
### outer loop
根据global BRAM的大小将卷积层切分为若干个block，具体为：
1. 根据global_BRAM for weight的out_channel维度将output channel切分
2. 根据global_BRAM for feature的channel维度和global_BRAM for weight的in_feature维度将in_channel切分
3. 根据global_BRAM for feature的height和width维度将卷积层的in_height和in_width维度切分

可以使用多块global_BRAM，对于存放input_feature和output_feature的global_BRAM，各块global_BRAM在channel维度堆叠。对于存放weight的global_BRAM，各块global_BRAM在output_channel维度堆叠。

outer_loop不考虑padding

### inner loop
循环调用PE对一个block的in_feature进行处理:
1. 从global BRAM拷贝in_feature到local BRAM。在此处考虑padding。
2. 从global BRAM拷贝weight到local BRAM。
3. 用bias或者partial_sum初始化local BRAM。
4. 调用PE进行计算。
5. 在PE处理完整个block的in_channel之后，将结果从local BRAM拷贝回global BRAM。若所有的in_channel处理完毕，在拷贝回global BRAM之前还需要进行relu
6. 将结果从global BRAM拷贝回DRAM
