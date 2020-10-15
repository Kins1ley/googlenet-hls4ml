# 模板测试代码
测试函数模板能否正确进行计算

## 卷积层和池化层测试代码
根据写的卷积层和池化层的模板代码，改里面的参数，验证googlenet所有卷积层和卷积层。

输出结果在./conv_out_code和./pool_outcode里

### 用法
> cd 到out_code里的文件夹里
> make Makefile cpp
> make Makefile hls

# TODO
1. cpp验证(已完成)
2. hls验证，包括不同的数据类型，如ap_fixed和ap_int，以及不同的BRAM大小和不同的parallism。