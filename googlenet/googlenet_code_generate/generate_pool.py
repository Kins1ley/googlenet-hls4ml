import os
import shutil


layer_config_dict={"left_bracket":"{","right_bracket":"}"}
layer_config_dict["pool_layer_name"]="pool1_3x3_s2"
layer_config_dict["pool_layer_type"]="MAXPOOL3x3_S2"
layer_config_dict["pool_pe_name"]="pool3x3"
layer_config_dict["pool_layer_input"]="conv1_7x7_s2_1"
layer_config_dict["pool_layer_output"]="pool1_3x3_s2_1"

def generate_pool_template():
    """
    generate the template file
    :return:
    """
    with open("./origin_code/pool_template_origin.h") as f:
        tmp_file=open("./origin_code/pool_template_tmp.h","w")
        while True:
            text = f.readline()
            if not text:
                break
            text = text.replace("{","{left_bracket")
            text = text.replace("}","right_bracket}")
            text = text.replace("{left_bracket", "{left_bracket}")
            text = text.replace("right_bracket}", "{right_bracket}")
            text = text.replace("pool1_3x3_s2_1", "{pool_layer_output}")
            text = text.replace("conv1_7x7_s2_1", "{pool_layer_input}")
            text = text.replace("pool1_3x3_s2","{pool_layer_name}")
            text = text.replace("MAXPOOL3x3_S2","{pool_layer_type}")
            text = text.replace("pool3x3", "{pool_pe_name}")

            print(text,file=tmp_file,end="")
            #print(text)
        tmp_file.close()
    #os.remove("pool_template.h")
    os.rename("./origin_code/pool_template_tmp.h", "./templates/pool_template.h")

def generate_pool(layer_config_dict,template_name="./templates/pool_template.h",out_file=None):
    """
    generate code according to the template file
    :param layer_config_dict:
    :param template_name:
    :param out_file: if None, show the result on the screen
    :return:
    """
    with open(template_name) as f:
        while True:
            text = f.readline()
            if not text:
                break
            if out_file is not None:
                print(text.format(left_bracket=layer_config_dict["left_bracket"],
                                  right_bracket=layer_config_dict["right_bracket"],
                                  pool_layer_name=layer_config_dict["pool_layer_name"],
                                  pool_layer_type=layer_config_dict["pool_layer_type"],
                                  pool_pe_name=layer_config_dict["pool_pe_name"],
                                  pool_layer_input=layer_config_dict["pool_layer_input"],
                                  pool_layer_output=layer_config_dict["pool_layer_output"]
                                  ), end="",file=out_file)
            else:
                print(text.format(left_bracket=layer_config_dict["left_bracket"],
                                  right_bracket=layer_config_dict["right_bracket"],
                                  pool_layer_name=layer_config_dict["pool_layer_name"],
                                  pool_layer_type=layer_config_dict["pool_layer_type"],
                                  pool_pe_name=layer_config_dict["pool_pe_name"],
                                  pool_layer_input=layer_config_dict["pool_layer_input"],
                                  pool_layer_output=layer_config_dict["pool_layer_output"]
                                  ), end="")
#generate_pool_template()
generate_pool(layer_config_dict)
