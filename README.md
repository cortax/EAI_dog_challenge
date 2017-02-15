# Classification on Stanford Dogs Dataset

## Data preparation
The "Stanford Dogs Dataset" contains over 20,000 pictures of dogs grouped into subfolders, one per breed. To avoid loading the whole dataset in memory, we use a flow_from_directory as our image generator. We need one flow_from_directory generator for the training set and one flow_from_directory generator for the test set. This requires spliting the data into training and testing beforehand. The function doing this work only requires the spliting ratio as input and will deterministically the first files for training and the rest for testing. One it's done, we can used flow_from_directory generators easily. 

The images comes in different sizes with either portrait or landscape aspect. The first preprocessing step is to resize all images to 300x300 pixels, which is roughly the smallest image width/heigth we can find in the dataset. We do not want to go too small, since classifying dog breeds might require observing subtle features at a very small scale. To retain this information, we begin with this resizing. 

The second preprocessing involves transforming the training images for data augmentation. We use full rotations of the images and flips in both directions since this won't affect the label on the image. A little zoom is used, however, this could generate erroneous labels if the dog is out of range, but the risk is small here. Also, all images are rescaled to [0,1]. An addition to the preprocessing could be to normalize to the data to 0 mean and 1 std. 

## Model architecture





