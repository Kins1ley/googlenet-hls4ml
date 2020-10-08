import os
import shutil
top_function_mark=   "/////////////top_function/////////////"
template_config_mark="/////////////template_config/////////////"
allocate_config_mark="/////////////allocate_config/////////////"


layer_config_dict={"left_bracket":"{","right_bracket":"}"}
layer_config_dict["conv_layer_name"]="conv1_7x7_s2"
layer_config_dict["conv_layer_type"]="CONV7x7_S2"
layer_config_dict["global_weight_name"]="global_weight_7x7"
layer_config_dict["global_weight_type"]="WEIGHT_GLOBAL_7x7"
layer_config_dict["DDR_weight_type"]="DDR_weight_7x7"
layer_config_dict["conv_pe_name"]="conv_output_reuse7x7"
layer_config_dict["conv_layer_input"]="image_in"
layer_config_dict["conv_layer_output"]="conv1_7x7_s2_1"

def generate_conv_template():
    """
    generate the template file
    :return:
    """
    with open("./origin_code/conv_template_origin.h") as f:
        tmp_file=open("./origin_code/conv_template_tmp.h","w")
        while True:
            text = f.readline()
            if not text:
                break
            text = text.replace("{","{left_bracket")
            text = text.replace("}","right_bracket}")
            text = text.replace("{left_bracket", "{left_bracket}")
            text = text.replace("right_bracket}", "{right_bracket}")
            text = text.replace("conv1_7x7_s2_1", "{conv_layer_output}")
            text = text.replace("image_in", "{conv_layer_input}")
            text = text.replace("conv1_7x7_s2","{conv_layer_name}")
            text = text.replace("CONV7x7_S2","{conv_layer_type}")
            text = text.replace("conv_output_reuse7x7", "{conv_pe_name}")
            text = text.replace("global_weight_7x7", "{global_weight_name}")
            text = text.replace("WEIGHT_GLOBAL_7x7", "{global_weight_type}")
            text = text.replace("DDR_weight_7x7", "{DDR_weight_type}")



            print(text,file=tmp_file,end="")
            #print(text)
        tmp_file.close()
    #os.remove("conv_template.h")
    os.rename("./origin_code/conv_template_tmp.h", "./function_templates/conv_template.h")

def generate_conv(layer_config_dict,template_name="./function_templates/conv_template.h",out_file=None,part="top_function"):
    """
    generate code according to the template file
    :param layer_config_dict:
    :param template_name:
    :param out_file: if None, show the result on the screen
    :param part: the part to write in the file.
    :return:
    """
    found_top_function=False
    found_template_config=False
    found_allocate_config=False
    with open(template_name) as f:
        while True:
            text = f.readline()
            if not text:
                break
            #print(text)
            if top_function_mark in text:
                if not found_top_function:
                    found_top_function=True
                else:
                    found_top_function=False
                continue

            if template_config_mark in text:
                if not found_template_config:
                    found_template_config=True
                else:
                    found_template_config=False
                continue

            if allocate_config_mark in text:
                if not found_allocate_config:
                    found_allocate_config=True
                else:
                    found_allocate_config=False
                continue

            if (part=="top_function" and found_top_function) or \
                (part=="template_config" and found_template_config) or \
                (part=="allocate_config" and found_allocate_config):
                if out_file is not None:
                    print(text.format(left_bracket=layer_config_dict["left_bracket"],
                                      right_bracket=layer_config_dict["right_bracket"],
                                      conv_layer_name=layer_config_dict["conv_layer_name"],
                                      conv_layer_type=layer_config_dict["conv_layer_type"],
                                      global_weight_name=layer_config_dict["global_weight_name"],
                                      global_weight_type=layer_config_dict["global_weight_type"],
                                      conv_pe_name=layer_config_dict["conv_pe_name"],
                                      conv_layer_input=layer_config_dict["conv_layer_input"],
                                      conv_layer_output=layer_config_dict["conv_layer_output"],
                                      DDR_weight_type=layer_config_dict["DDR_weight_type"]
                                      ), end="",file=out_file)
                else:
                    print(text.format(left_bracket=layer_config_dict["left_bracket"],
                                      right_bracket=layer_config_dict["right_bracket"],
                                      conv_layer_name=layer_config_dict["conv_layer_name"],
                                      conv_layer_type=layer_config_dict["conv_layer_type"],
                                      global_weight_name=layer_config_dict["global_weight_name"],
                                      global_weight_type=layer_config_dict["global_weight_type"],
                                      conv_pe_name=layer_config_dict["conv_pe_name"],
                                      conv_layer_input=layer_config_dict["conv_layer_input"],
                                      conv_layer_output=layer_config_dict["conv_layer_output"],
                                      DDR_weight_type = layer_config_dict["DDR_weight_type"]
                                      ), end="")
if __name__ == "__main__":
    generate_conv_template()
    generate_conv(layer_config_dict)
