import os
import shutil
top_function_mark=   "/////////////top_function/////////////"
template_config_mark="/////////////template_config/////////////"
allocate_config_mark="/////////////allocate_config/////////////"


layer_config_dict={"left_bracket":"{","right_bracket":"}"}
layer_config_dict["lrn_layer_name"]="pool1_norm1"
layer_config_dict["lrn_layer_type"]="LRN"
layer_config_dict["lrn_pe_name"]="LRN"
layer_config_dict["lrn_layer_input"]="pool1_3x3_s2_1"
layer_config_dict["lrn_layer_output"]="pool1_norm1_1"

def generate_lrn_template():
    """
    generate the template file
    :return:
    """
    with open("./origin_code/lrn_template_origin.h") as f:
        tmp_file=open("./origin_code/lrn_template_tmp.h","w")
        while True:
            text = f.readline()
            if not text:
                break
            text = text.replace("{","{left_bracket")
            text = text.replace("}","right_bracket}")
            text = text.replace("{left_bracket", "{left_bracket}")
            text = text.replace("right_bracket}", "{right_bracket}")
            text = text.replace("pool1_norm1_1", "{lrn_layer_output}")
            text = text.replace("pool1_3x3_s2_1", "{lrn_layer_input}")
            text = text.replace("pool1_norm1","{lrn_layer_name}")
            text = text.replace("LRN","{lrn_layer_type}")
            #text = text.replace("LRN", "{lrn_pe_name}")


            print(text,file=tmp_file,end="")
            #print(text)
        tmp_file.close()
    #os.remove("lrn_template.h")
    os.rename("./origin_code/lrn_template_tmp.h", "./function_templates/lrn_template.h")

def generate_lrn(layer_config_dict,template_name="./function_templates/lrn_template.h",out_file=None,part="top_function"):
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
                                      lrn_layer_name=layer_config_dict["lrn_layer_name"],
                                      lrn_layer_type=layer_config_dict["lrn_layer_type"],
                                      lrn_pe_name=layer_config_dict["lrn_pe_name"],
                                      lrn_layer_input=layer_config_dict["lrn_layer_input"],
                                      lrn_layer_output=layer_config_dict["lrn_layer_output"]
                                      ), end="", file=out_file)
                else:
                    print(text.format(left_bracket=layer_config_dict["left_bracket"],
                                      right_bracket=layer_config_dict["right_bracket"],
                                      lrn_layer_name=layer_config_dict["lrn_layer_name"],
                                      lrn_layer_type=layer_config_dict["lrn_layer_type"],
                                      lrn_pe_name=layer_config_dict["lrn_pe_name"],
                                      lrn_layer_input=layer_config_dict["lrn_layer_input"],
                                      lrn_layer_output=layer_config_dict["lrn_layer_output"]
                                      ), end="")



if __name__ == "__main__":
    generate_lrn_template()
    generate_lrn(layer_config_dict)
