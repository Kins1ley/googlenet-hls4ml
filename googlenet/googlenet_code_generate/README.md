# googlenet-HLS4ML code generator
用法
> python generate_googlenet.py

生成的代码在./out_code/
## generate_conv.py
1. layer_config_dict的创建参照开头的几行。
2. 把layer_config_dict作为参数传入generate_conv函数，函数就可以根据function_template文件夹里的模板生成对应的卷积代码。
## generate_pool.py & generate_lrn.py 
与generate_conv.py相同
## onnx_info.py
读取googlenet-7.onnx文件里的卷积层，生成传入generate_conv.py等文件的layer_config_dict。
### Concat层的处理
输入concat层的层的输出变为concat层的输出，在global_BRAM拷贝回DRAM时实现concat操作。
## generate_googlenet.py
依据onnx文件填充卷积层，池化层和LRN层的模板，并写入googlenet.h，allocate_config.h和template_config.h。

## TODO
1. 复用DRAM的feature (已完成)
2. 读取权重 
3. 合并onnx_info.py中的几个读onnx文件的函数
4. 细节完善(lrn的其他参数，maxpooling的最小值设置等)
5. 为整个googlenet做csim

