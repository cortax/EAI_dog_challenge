# Classification on Stanford Dogs Dataset

## Data preparation
The "Stanford Dogs Dataset" contains over 20,000 pictures of dogs grouped into subfolders, one per breed. To avoid loading the whole dataset in memory, we use a flow_from_directory as our image generator. We need one flow_from_directory generator for the training set and one flow_from_directory generator for the test set. This requires spliting the data into training and testing beforehand. The function doing this work only requires the spliting ratio as input and will deterministically the first files for training and the rest for testing. Once it's done, we can use flow_from_directory generators easily. 

The images comes in different sizes with either portrait or landscape aspect. The first preprocessing step is to resize all images to 299x299 pixels, which is roughly the smallest image width/heigth found in the dataset. We do not want to go too small, since classifying dog breeds might require observing subtle features at a very small scale such as fur. 

The second preprocessing involves transforming the training images for data augmentation. We use full rotations of the images and flips in both directions since this won't affect the label on the image. A little zoom is used, however, this could generate erroneous labels if the dog is out of range, but the risk is small here. Also, all images are rescaled to [0,1]. An addition to the preprocessing could be to normalize to the data to 0 mean and 1 std. 

## Model architecture
The main constrain of the architecture is the hardware, a GTX 1050 Ti with 4Go of memory, which is an entry-level GPU. This led us to make conservative choice on the architecture. We use the pretrained model Inception-v3 and retrain the model as suggested by Keras. First, the fully connected layers controlling the classification is retrained. Then, we go a little deeper and retrain a few layers below the fully connected layers. The input layer remains as is, since we resize all images down to the default image input of Inception-v3, 299x299. 

Using a pretrained network is a safe bet for someone new to Keras, which is my case. Moreover, it can save a lot of time, since the training is already initiated. Here, the bottom layers of Inception-v3 essentially becomes a feature extractor and we can train a classifier using the Inception-v3 feature space. The approach consists to fine-tune the parameters of the network with a small learning rate and to add a fully connected layer to achieve the new classification task. 

The performance on the Stanford dogs dataset were immediate. Some tests were made on training a CNN from scratch, but the performance could hardly get above random within 30 minutes of training. Retraining Inception-v3 using the proposed method in the documentation easily led to 40% validation accuracy only with a new FC layer, this is about half of the state-of-the-art performance in 30 minutes. 

The training approach we use here relies on learning rate ladder. Basically, a learning rate is used for 10 epochs and then divided by 10. The ladder contains 4 steps for a total of 40 epochs. This process is repeated several times by gradually unlocking more and more inception blocks. This is done 3 times, which means a total of 120 learning epochs. This decision of retraining deeper layers is motivated by the availability of a moderate amount of data, diminishing the risk of overfitting. 


