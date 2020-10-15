import os
import shutil
import onnx_info

onnx_file_name="googlenet-7.onnx"
template_path="./conv_template_code/"
files= os.listdir(template_path)
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
layer_list=onnx_info.get_layer_list(onnx_file_name,"conv")

def write_googlenet(template_file_name,out_file_name,layer_config):
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        text=text.replace("CONV7x7_S2","CONV{kernel_height}x{kernel_width}_S{stride}".format(**layer_config))
        text=text.replace("conv_output_reuse7x7","conv_output_reuse{kernel_height}x{kernel_width}".format(**layer_config))
        text=text.replace("WEIGHT_GLOBAL_7x7","WEIGHT_GLOBAL_{kernel_height}x{kernel_width}".format(**layer_config))

        print(text,file=out_file,end="")

def write_header(template_file_name,out_file_name,layer_config):
    insert_tag_1="/////header_insert_1/////"
    insert_code_1="const int IMAGE_CH = {in_channel}; // image channel\n\
const int IMAGE_H = {in_height};// image height\n\
const int IMAGE_W = {in_width};// image width"
    insert_tag_2 = "/////header_insert_2/////"
    insert_code_2 = "const int conv1_7x7_s2_in_channel = IMAGE_CH;\n\
const int conv1_7x7_s2_in_height = IMAGE_H;\n\
const int conv1_7x7_s2_in_width = IMAGE_W;\n\
const int conv1_7x7_s2_pad_top = {pad_top};\n\
const int conv1_7x7_s2_pad_bottom = {pad_bottom};\n\
const int conv1_7x7_s2_pad_left = {pad_left};\n\
const int conv1_7x7_s2_pad_right = {pad_right};\n\
const int conv1_7x7_s2_kernel_num = {kernel_num};\n\
const int conv1_7x7_s2_kernel_channel = conv1_7x7_s2_in_channel;\n\
const int conv1_7x7_s2_kernel_channel_DDR_offset = 0;\n\
const int conv1_7x7_s2_kernel_height = {kernel_height};\n\
const int conv1_7x7_s2_kernel_width = {kernel_width};\n\
const int conv1_7x7_s2_bias_num = conv1_7x7_s2_kernel_num;\n\
const int conv1_7x7_s2_bias_DDR_offset = 0;\n\
const int conv1_7x7_s2_out_channel = conv1_7x7_s2_kernel_num;\n\
const int conv1_7x7_s2_out_height = {out_height};\n\
const int conv1_7x7_s2_out_width = {out_width};\n\
const int conv1_7x7_s2_out_channel_DDR_offset = 0;"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag_1 in text:
            print(insert_code_1.format(**layer_config),file=out_file)
        if insert_tag_2 in text:
            print(insert_code_2.format(**layer_config),file=out_file)


def write_other(template_file_name,out_file_name):
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")

def generate_conv_test(template_path,out_path,layer_config):
    #print(layer_config)
    for fn in files:
        template_file_name = os.path.join(template_path, fn)
        out_file_name = os.path.join(out_path, fn)
        if "googlenet.h" in fn:
            write_googlenet(template_file_name, out_file_name, layer_config)
        elif "googlenet_conv.cpp" in fn:
            write_googlenet(template_file_name, out_file_name, layer_config)
        elif "allocate_config.h" in fn:
            write_googlenet(template_file_name, out_file_name, layer_config)
        elif "header.h" in fn:
            write_header(template_file_name, out_file_name, layer_config)
        else:
            write_other(template_file_name, out_file_name)
if __name__ == "__main__":
    for conv_layer in layer_list:
        out_path="./conv_out_code/{}_test_out".format(conv_layer["layer_name"])
        setDir(out_path)
        generate_conv_test(template_path,out_path,conv_layer)