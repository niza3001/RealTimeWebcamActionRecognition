import os
import pickle

root_folder = os.getcwd()
content = {}

for root, dirs, files in os.walk(root_folder):
    for subdir in dirs:
    	print subdir
        # content[subdir] = len(files)

# with open(os.getcwd()+'/dataloader/dic/frame_count.pickle','rb') as file:
#     dic_frame = pickle.load(file)
#     with open("output.txt", "w") as txt_file:
# 	    for key in dic_frame:
# 		    txt_file.write("".join(key) + " : ")
# 		    val = dic_frame[key]
# 		    txt_file.write(str(val) + "\n")
# file.close()

