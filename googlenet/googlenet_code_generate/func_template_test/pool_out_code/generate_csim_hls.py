import os
files=os.listdir(".")
print(files)
with open("csim.sh","w") as f:
    print("#! /bin/bash",file=f)
    for file in files:
        if ".sh" not in file and ".py" not in file:
            print("cd {file_name}\npwd\nmake Makefile cpp\nrm cppExe\ncd ..".format(file_name=file),file=f)
with open("hls.sh","w") as f:
    print("#! /bin/bash",file=f)
    for file in files:
        if ".sh" not in file and ".py" not in file:
            print("cd {file_name}\npwd\nmake Makefile hls\ncd ..".format(file_name=file),file=f)
