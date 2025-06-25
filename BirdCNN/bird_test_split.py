# Removes test images from training folder
# Creates new folder with those images 
# Goes by an 80-20 split
# PERMANENTLY ALTERS DATA FOLDERS (so keep a back up)

# NOTE: assumes equal img number for each class

import os
import shutil

root_src = r"bird_data/birds_train"
root_dest = r"bird_data/birds_test"
    
# go through each folder in train, take some imgs, create corresponding test folder

def split_folder(folder_name, num_test_files): 

    folder_path = os.path.join(root_src, folder_name)

    img_names = [img.name for img in os.scandir(folder_path)]
    num_imgs = len(img_names)

    test_folder_path = os.path.join(root_dest, folder)
    os.mkdir(test_folder_path)

    # After creating the test folder, perform the split
    # Removes from the end of each folder (counts backward)

    for img_index in range(num_test_files):
        curr_img = img_names[num_imgs - img_index - 1]
        
        src = os.path.join(folder_path, curr_img)
        dest = os.path.join(test_folder_path, curr_img)
        shutil.move(src, dest)

# MAIN:

if (not os.path.exists(root_dest)):
    os.mkdir(root_dest)

all_folders = [folder.name for folder in os.scandir(root_src) 
               if folder.is_dir()]

for folder in all_folders:
    split_folder(folder, 8) # .2 of 40