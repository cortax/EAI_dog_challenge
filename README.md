# Classification on Stanford Dogs Dataset

## Data preparation
The "Stanford Dogs Dataset" contains over 20,000 pictures of dogs grouped into subfolders, one per breed. To avoid loading the whole dataset in memory, we use a flow_from_directory as our image generator. We need one flow_from_directory generator for the training set and one flow_from_directory generator for the test set. This requires spliting the data into training and testing beforehand. The function doing this work only requires the spliting ratio as input and will deterministically the first files for training and the rest for testing. Once it's done, we can use flow_from_directory generators easily. 

The images comes in different sizes with either portrait or landscape aspect. The first preprocessing step is to resize all images to 299x299 pixels, which is roughly the smallest image width/heigth found in the dataset. We do not want to go too small, since classifying dog breeds might require observing subtle features at a very small scale such as fur. 

The second preprocessing involves transforming the training images for data augmentation. We use full rotations of the images and flips in both directions since this won't affect the label on the image. A little zoom is used, however, this could generate erroneous labels if the dog is out of range, but the risk is small here. Also, all images are rescaled to [0,1]. An addition to the preprocessing could be to normalize to the data to 0 mean and 1 std. 

## Model architecture
The main constrain of the architecture is the hardware, a GTX 1050 Ti with 4Go of memory, which is an entry-level GPU. This led us to make conservative choice on the architecture. We use the pretrained model Inception-v3 and retrain the model as suggested by Keras. First, the fully connected layers controlling the classification is retrained. Then, we go a little deeper and retrain a few layers below the fully connected layers. The input layer remains as is, since we resize all images down to the default image input of Inception-v3, 299x299. 

Using a pretrained network is a safe bet for someone new to Keras, which is my case. Moreover, it can save a lot of time, since the training is already initiated. Here, the bottom layers of Inception-v3 essentially becomes a feature extractor and we can train a classifier using the Inception-v3 feature space. The approach consists to fine-tune the parameters of the network with a small learning rate and to add a fully connected layer to achieve the new classification task. 

The performance on the Stanford dogs dataset were immediate. Some tests were made on training a CNN from scratch, but the performance could hardly get above random within 30 minutes of training. Retraining Inception-v3 using the proposed method in the documentation easily led to 40% validation accuracy only with a new FC layer, this is about half of the state-of-the-art performance in 30 minutes. 

The training approach we use here relies on learning rate ladder. Basically, a learning rate is used for 5 epochs and then divided by 10. The ladder contains 3 steps for a total of 15 epochs. This process is repeated several times by gradually unlocking more and more inception blocks. This is done 3 times, which means a total of 45 learning epochs. This decision of retraining deeper layers is motivated by the availability of a moderate amount of data, diminishing the risk of overfitting. Also, the number of epochs is chosen so that the overall training takes less than 4 hours. 

## Results

The network as been trained once and learning parameters were not fined-tuned. At this point, we should interpret the results and make modifications to the learning process to improve performance. We use the accuracy as our metric to determine the quality of a model, eventhough the optimization is done over a cross entropy loss. State-of-the-art performance on this dataset currently reach ~80-85% accuracy. The learned model does not achieve such good performance, but still performs quite well. The results are not averaged and taken from a single run. The dataset splitting (85-15) is also deterministic. 

Here is 
![accuracies](https://cloud.githubusercontent.com/assets/6197868/22986366/46267b04-f379-11e6-8fc4-5febac647bc0.png)
![losses](https://cloud.githubusercontent.com/assets/6197868/22986367/47620a24-f379-11e6-8be2-9e378e756ac9.png)




