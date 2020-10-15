import os
import shutil
import onnx_info

onnx_file_name="googlenet-7.onnx"
template_path="./pool_template_code/"
files= os.listdir(template_path)
out_path="./pool_test_out/"
def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    print("cleaning dir {}".format(filepath))
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
layer_list=onnx_info.get_layer_list(onnx_file_name,"maxpool")
layer_list+=onnx_info.get_layer_list(onnx_file_name,"avgpool")
def write_googlenet(template_file_name,out_file_name,layer_config):
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        text=text.replace("MAXPOOL3x3_S2","{pool_type}{kernel_height}x{kernel_width}_S{stride}".format(**layer_config))
        text=text.replace("nnet::pool3x3","nnet::pool{kernel_height}x{kernel_width}".format(**layer_config))

        print(text,file=out_file,end="")

def write_header(template_file_name,out_file_name,layer_config):
    insert_tag_1="/////header_insert_1/////"
    insert_code_1="const int conv1_7x7_s2_in_channel = IMAGE_CH;\n\
const int conv1_7x7_s2_in_height = IMAGE_H;\n\
const int conv1_7x7_s2_in_width = IMAGE_W;\n\
const int conv1_7x7_s2_pad_top = 3;\n\
const int conv1_7x7_s2_pad_bottom = 3;\n\
const int conv1_7x7_s2_pad_left = 3;\n\
const int conv1_7x7_s2_pad_right = 3;\n\
const int conv1_7x7_s2_kernel_num = {in_channel};\n\
const int conv1_7x7_s2_kernel_channel = conv1_7x7_s2_in_channel;\n\
const int conv1_7x7_s2_kernel_height = 7;\n\
const int conv1_7x7_s2_kernel_width = 7;\n\
const int conv1_7x7_s2_out_channel = conv1_7x7_s2_kernel_num;\n\
const int conv1_7x7_s2_out_height = {in_height};\n\
const int conv1_7x7_s2_out_width = {in_width};\n\
///layer pool1_3x3_s2\n\
const int pool1_3x3_s2_in_channel = {in_channel};\n\
const int pool1_3x3_s2_in_height = {in_height};\n\
const int pool1_3x3_s2_in_width = {in_width};\n\
const int pool1_3x3_s2_pad_top = {pad_top};\n\
const int pool1_3x3_s2_pad_bottom = {pad_bottom};\n\
const int pool1_3x3_s2_pad_left = {pad_left};\n\
const int pool1_3x3_s2_pad_right = {pad_right};\n\
const int pool1_3x3_s2_out_channel = pool1_3x3_s2_in_channel;\n\
const int pool1_3x3_s2_kernel_height = {kernel_height};\n\
const int pool1_3x3_s2_kernel_width = {kernel_width};\n\
const int pool1_3x3_s2_out_height = {out_height};\n\
const int pool1_3x3_s2_out_width = {out_width};\n\
const int pool1_3x3_s2_stride = {stride};\n\
const int pool1_3x3_s2_out_channel_DDR_offset = 0;"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag_1 in text:
            print(insert_code_1.format(**layer_config),file=out_file)


def write_other(template_file_name,out_file_name):
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")

def generate_pool_test(template_path,out_path,layer_config):
    #print(layer_config)
    for fn in files:
        template_file_name = os.path.join(template_path, fn)
        out_file_name = os.path.join(out_path, fn)
        if "googlenet.h" in fn:
            write_googlenet(template_file_name, out_file_name, layer_config)
        elif "googlenet_pool.cpp" in fn:
            write_googlenet(template_file_name, out_file_name, layer_config)
        elif "allocate_config.h" in fn:
            write_googlenet(template_file_name, out_file_name, layer_config)
        elif "header.h" in fn:
            write_header(template_file_name, out_file_name, layer_config)
        else:
            write_other(template_file_name, out_file_name)
if __name__ == "__main__":
    for pool_layer in layer_list:
        out_path="./pool_out_code/{}_test_out".format(pool_layer["layer_name"])
        setDir(out_path)
        generate_pool_test(template_path,out_path,pool_layer)