# What is this Project?

This is a convulutional neural network that is purposed to classify around 200 different species of birds.

## Bird_Data

The data is from Kaggle: [https://www.kaggle.com/datasets/dmcgow/birds-200](url)  
Each subdirectory is organized in a similar fashion:

\\---Subdirectory  
    ###
    \\---BirdSpecies1  
        |---img1  
        |---img2  
        |---img3  
        ...  
    ###
    \\---BirdSpecies2  
    \\---BirdSpecies3  
    ...  

There are around 8000 (between both training and test data). There are 200 BirdSpecies folders in each subdirectory.

## Model.py

The model is implemented using Pytorch.  
The model currently is not accurate; this is partly due to my inexperience choosing layer dimensions. Additionally, as of June 27, 2025 (the time I'm writing this), I lack the hardware to efficiently train the model (I am forced to use my CPU, which is quite slow).  
All data preparation, model configuration, training and validation functions, and user IO is located on this file. This will likely change in the future.

## Bird_Test_Split.py

This file splits the data into training and validation groups.  
This needed to be done because, unfortunately, the original testing folder from Kaggle does not have labeled images.  
This file should only have to be run once.

## Compute_Parameters.py

This file computes the channel means and standard deviations needed to normalize the data.  
It uses Dask to (hopefully) make computing more efficient.  
This file should only run once, in which it writes the calculations to image_stats.txt.

## Saved_Checkpoint.pth

This file stores the model's current state.


