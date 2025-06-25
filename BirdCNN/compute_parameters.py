# This is literally just to find the mean and standard deviation of the data
# (Lord have mercy on us all)

import time
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import dask

def calc_for_dir(folder):
    
    # Calculations are done for each image:
    pixel_sum = np.zeros(3)     # [mean_red, mean_green, mean_blue]
    pixel_sq_sum = np.zeros(3)
    num_imgs = 0

    paths = [img.path for img in os.scandir(folder)]
    for img_path in paths:
        img = Image.open(img_path)
        img = np.array(img) / 255.0  # pixel range, scale to [0,1]
        pixel_sum += img.mean(axis = (0,1)) # Per-Channel E[X]
        pixel_sq_sum += (img ** 2).mean(axis = (0,1)) # Per-Channel E[X^2]
        num_imgs += 1
    
    return pixel_sum, pixel_sq_sum, num_imgs

# for timing the process (just counts each second)

def start_time():
    curr_secs = 0
    while True:
        mins, secs = divmod(curr_secs, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end = '\r')
        time.sleep(1)
        curr_secs += 1

all_folders = [folder.path for folder in os.scandir(r"bird_data\birds_train")
               if folder.is_dir()]

blank_arr = np.zeros(3)
sums = [blank_arr, blank_arr, 0]

# Done to avoid Dask recursively creating subprocesses within its subprocesses
# NOTE: says it processes crash after, but it returns that it went through all 8000 images...so do I win???

if __name__ == "__main__":
    from dask.distributed import Client, as_completed
    dask.config.set(scheduler='processes')

    client = Client(processes = True)

    print("Running processes: ")

    timer = client.submit(start_time)
    futures = client.map(calc_for_dir, all_folders)

    for future in as_completed(futures):
        dir_sum = future.result()
        
        sums[0] += dir_sum[0]
        sums[1] += dir_sum[1]
        sums[2] += dir_sum[2]

    timer.cancel()
    print("Done; looked at %d images" % (sums[2]))

    mean = sums[0]/sums[2]
    std = np.sqrt(sums[1] / sums[2] - mean ** 2)

    with open("image_stats.txt", "w") as file:
        file.write(str(mean) + "\n" + str(std))

    print("Mean: ", mean)
    print("Standard deviation: ", std)