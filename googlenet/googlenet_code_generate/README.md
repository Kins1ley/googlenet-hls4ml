# googlenet-HLS4ML code generator
## generate_conv.py
1. 参照开头的几行，为对应层创建layer_config_dict。
2. 把layer_config_dict作为参数传入generate_conv函数，函数就可以根据模板生成对应的卷积代码。
## generate_pool.py
与generate_conv.py相同

## TODO
1. 三个文件，googlenet.h，allocate_config.h和template_config.h都需要写入。目前三个文件需要的代码段的模板保存在一个文件"./templates/conv_template.h"里，generate_conv函数需要能够分别把代码段写入到三个文件里。
2. 调用generate_conv函数自动写入googlenet.h等文件生成整个googlenet，尽量减少需要手动填充的部分。
3. 除了conv和pool，还有一些其他的函数，由于计算量不大，可能不需要考虑切分，所以等我们把googlenet的大框架写完再把这些函数补上。