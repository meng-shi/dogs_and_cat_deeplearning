# dogs_and_cat_deeplearning

Background¶
Image classification is one of the most fundamental problems in computer vision. It focuses on teaching machines to identify the object or category that appears in an image.

Visual recognition plays a crucial role in many advanced technologies. For example, autonomous driving relies heavily on a vehicle’s ability to accurately classify road signs, pedestrians, vehicles, and other objects in real time. Medical imaging systems use classification to detect abnormalities such as tumors or fractures. Security applications depend on visual recognition for facial identification, object detection, and monitoring. In nearly every area where computers interact with the physical world, image classification forms the first step that enables high-level reasoning and decision making.

To gain hands-on experience with this essential concept, I chose to experiment with a classic and widely used dataset: the Dogs vs. Cats image dataset.

Data
Data source:
Will Cukierski. Dogs vs. Cats Redux: Kernels Edition. https://kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition, 2016. Kaggle.

Overview:
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).

EDA
During the EDA, I inspected the raw images and found that their sizes are varied and are not consistently square. But their ratio difference is not huge. I resized all images to a uniform 150×150. I applied a series of augmentation techniques, including Random Horizontal Flip, Random Rotation, Color Jitter, and Random Affine transformations. These augmentations help the model become more robust.

Model
I evaluated three different convolutional neural network architectures to compare performance: BaselineCNN, ResNet18, and EfficientNet-B0.

BaselineCNN:
First one is a lightweight, 3-convolution-layer Baseline CNN model designed to provide a simple benchmark.

ResNet18:
ResNet is the second; it uses residual blocks with skip connections.

EfficientNet-B0:
EfficientNet-B0 is the final modern CNN that scales depth, width, and image resolution in a balanced way using a compound scaling method. It typically provides high accuracy while maintaining computational efficiency. With Fewer parameters, it is more efficient than ResNet18.

Optimization and Results

Fine-tuned the learning rate, decay, and epochs to optimize the model architecture.

After various optimization strategies were applied to improve model performance. If we checked the logloss result. The best result is from efficientnet_b0,learning_rate: 0.001, weight_decay: 0.0, epochs: 5. The logloss is 0.0684, less than the 0.036868. But the reason is probably that I set up the max step to 300. I tested with the entire training set later. The final result improves but not a lot.

Conclusion
Fine-tuning the custom CNN did not lead to significant improvements. Despite extensive hyperparameter tuning, its validation log loss remained high, and its performance was far worse than the pretrained transfer learning models.
Key Takeaways

This experiment confirms that transfer learning with pretrained models, combined with proper data preparation and hyperparameter tuning, can significantly improve performance on visual recognition tasks.

Reference¶
Dogs vs. Cats Redux: Will Cukierski. Dogs vs. Cats Redux: Kernels Edition. https://kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition, 2016. Kaggle.

ResNet18: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

Efficientnet_b0: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
