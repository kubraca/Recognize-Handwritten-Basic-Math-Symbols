# Recognize-Handwritten-Basic-Math-Symbols
Convoloution Neural Network SequNet for  Recognize Handwritten Basic Math Symbols
Mathematics has played a dominant role in each of us at least once in our lives, and 
perhaps continues to play a dominant role for some of us. When it comes to solving a 
mathematical expression, some of us have a hard time doing it. For example, imagine that a 6-
year-old is new to mathematics and you are trying to teach him the simplest mathematical 
symbols. Therefore, if we have an artificial intelligence model that knows and recognizes 
these symbols, the learning phase of the 6-year-old child will be much easier. Because every 
time he is forced, he will have a model that will tell him what the symbol is. In this study, an 
Artificial Neural Network model, SequNet, was created for the recognition of handwritten 
numbers, operators and symbols. The GUI interface is built for users to type their expressions 
and image manipulation is done by capturing an image from the canvas and converting it to a 
NumPy array and then converting it to binary array. These numbers and symbols are then sent 
to the neural network for predictions. The model gave 94.00% training accuracy and 94.00% 
test accuracy also valid accuracy 95%. Finally, the expression is evaluated and the translated
expression is printed on the screen.

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/69c0d114-8118-45a9-b69c-88427c4d5a35)
Related Work
a) Artifical Neural Network: 
The concept of Artificial Neural networks plays a very important role in the human 
body of neural networks. In science and engineering, neural networks are 
mathematical or computational models prone to machine learning, deep learning, and 
pattern recognition. In an artificial neural network, the term "network" refers to the 
interconnections between neurons in discrete layers of each system. These connections 
are represented by directed edges with weights. The neural network receives the input 
signal from the outside world in the form of a pattern and an image in the form of a 
vector.[4] The input layer receives input from the outside world. Real learning or 
recognition happens when you are at the input layer. The output layer contains the 
information in the system and the neurons that confirm whether the system has learned 
any task. Hidden layers are the layers between the input and output layers. Here, the 
units take a set of weighted inputs and generate an output via the activation function. 
The activation function is basically a set of transfer functions used to get the desired 
output.
b)Multilayer Perceptions:
Deep learning has many classifier methods like called Multi-Layer perception 
(MLP), is used to classify the handwritten digits. Multilayer perceptron 
includes of three different layers, input layer, hidden layer and output 
layer. Each of the layers can have certain number of nodes also called neurons and 
each node in a layer is connected to all other nodes to the next layer. Therefore it 
is also known as feed forward network. The number of nodes in the input layer 
contigent the number of attributes present in the dataset. The number of nodes in 
the output layer depends on the number of classes exist in the dataset. The 
connection between two nodes includes of a weight. For the trainig aim, it uses a 
supervised learning technique named as Back propagation algorithm.
As can be seen in the image containing the code snippet below, we created a multilayered architecture while creating our SequNet model.
c) BatchNormalization:
It is a method used to make the training of neural networks faster and more stable by 
normalizing the inputs of the layers through recentering and rescaling.
Thanks to batch normalization, layers in the network do not have to wait for the previous 
layer to learn. It allows simultaneous learning. It accelerates our education. If we use a 
high learning rate without using batch normalization, we may encounter the problem of 
gradients disappearing. However, with the batch norm, we can use higher learning rates 
since the change in one layer does not spread to the other layer. In addition, the batch 
norm makes the network more stable and organized.
If we look at the loss chart below, with Batch normalization, the losses are 
considerably reduced.
A loss function is a function that compares the target and predicted output values; 
measures how well the neural network models the training data. When training, we 
aim to minimize this loss between the predicted and target outputs. In other words, loss 
functions are a measurement of how good your model is in terms of predicting the 
expected outcome.

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/f5af2d33-1bcc-4c0f-9135-46cf32d44254)
 -Our Approach
In this section, it will be mentioned how to make simple mathematical symbols 
recognized with a simple model. Libraries and technologies used while developing this model 
are: Open-cv, keras, numpy, tensorflow, matploit, pandasAfterwards, some operations were 
performed on the data, for example, labeling and preproces the data.
![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/8f976672-f484-4ab3-b14b-b1f18eb52c1a)
Then we need to shuffle the data to get better results.Shuffling data serves the purpose 
of reducing variance and making sure that models remain general and overfit less. The 
obvious case where you'd shuffle your data is if your data is sorted by their class/target.

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/52068f06-374a-4b1b-884c-081f6e04c921)

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/302bdd30-4485-4df8-afd1-c4b6b4767458)
The train-test split is a technique for evaluating the performance of a machine learning 
algorithm.It can be used for classification or regression problems and can be used for any 
supervised learning algorithm.Train test discrimination is used to predict the performance of 
machine learning algorithms applicable to predictive algorithms. This method is a quick and 
easy procedure to perform such that we can compare our own machine learning model results 
with machine results.Also in this model random state is important, The random state 
hyperparameter is used to control any such randomness involved in machine
learning models to get consistent results. We can use cross-validation to mitigate the effect of 
randomness involved in machine learning models.

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/f55dd819-ae07-4ce3-b3be-a7113141b276)
Data normalization is an important step that ensures that each input parameter (pixel in 
this case) has a similar data distribution. This speeds up convergence when training the 
network.
The numbers will be small and the calculation will become easier and faster. Except 
for 0, the range is 255, as the pixel values range from 0 to 256. So dividing all values by 255 
will convert it from 0 to 1.

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/02ad59c8-0903-4c7f-bc71-4a3172c725fa)

Categorical Data is the data corresponding to the Categorical Variable. A Categorical 
Variable is a variable that takes a fixed, limited set of possible values. For example Gender, 
Blood type, a resident or non-resident in the country etc. Machine learning models require all 
input and output variables to be numeric. This means that if your data contains categorical 
data, you must encode it to numbers before you can fit and evaluate a model.

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/8475c28e-29c8-477a-9de7-bd54fdc7f8c9)

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/34c91b02-7d07-4a51-99ee-2bd083d3f7ed)

Maximum pooling or maxpooling is a pooling operation that calculates the 
maximum or largest value in each patch of each feature map. The results are downsampled or 
pooled feature maps that highlight the most available feature in the patch, not the average 
presence of the feature in the average pooling situation. In this section, the model structure 
will be discussed.
Flatten, The flattening step is a refreshingly simple step involved in building a 
convolutional neural network. It involves taking the pooled feature map that is generated in 
the pooling step and transforming it into a one-dimensional vector. Flattening is converting 
the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of 
the convolutional layers to create a single long feature vector. And it is connected to the final 
classification model, which is called a fully-connected layer. In other words, we put all the 
pixel data in one line and make connections with the final layer [11]
Model consists of 4 convolution layers, used 64 filters in the first two, 128 each in the 
other two.In this section applied batch normalization between each layer, the data is 
normalized before passing to the other layer. Also, maxpooled 2 layer so the number of 
entries was halved.
![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/be689d77-850c-4494-be8b-43c5942e1bac)

Dropout was applied after the Flatten layer and did not transfer five percent of the features to 
the other layer. So we avoided memorization and applied L2 regularization in the last dense 
layers and got 0.001 in regularization coefficient. The last layer has 16 outputs. Also, the 
output is of course not limited to this.
![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/785a71e5-0a62-4615-9ca7-1ad170877511)

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/82cbfadf-ca60-4ed6-a273-55604a4cf1cb)
Testing
In this section, model saved an loaded. Also, compare actual y values with prediction y values and print the screen classification report.
![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/db4b7636-0f8c-4c63-90cc-79650ed29d0e)
Object detection is a process of finding all the possible instances of real-world 
objects, such as human faces, flowers, cars, etc. in images or videos, in real-time with 
utmost accuracy. The object detection technique uses derived features and learning 
algorithms to recognize all the occurrences of an object category. Object detection technique 
helps in the recognition, detection, and localization of multiple visual instances of objects in 
an image or a video. It provides a much better understanding of the object as a whole, rather
than just basic object classification. This method can be used to count the number of instances 
of unique objects and mark their precise locations, along with labeling. With time, the 
performance of this process has also improved significantly, helping us with real-time use 
cases. All in all, it answers the question: “What object is where and how much of it is there?”.
The main concept behind this process is that every object will have its features. These 
features can help us to segregate objects from the other ones. Object detection methodology 
uses these features to classify the objects. The same concept is used for things like face 
detection, fingerprint detection, etc.
The machine learning approach requires the features to be defined by using various 
methods and then using any technique such as Support Vector Machines (SVMs) to do the 
classification. Whereas, the deep learning approach makes it possible to do the whole 
detection process without explicitly defining the features to do the classification. The deep 
learning approach is majorly based on Convolutional Neural Networks (CNNs).

![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/8d97bc03-a1c8-44a8-8c4c-28a041e33c4a)
In this model, is ready now we will use it on the equatio, first the picture is converted 
to gray format, then canny edge detection is done and the lines of the picture become clear 
with find contours, every single character is found and we order these characters from left to 
right.
Each character taken in turn and convert it to the format the model wants with the 
normalization and resize stages and give it to the model then we draw around the characters 
we found in the main picture and print the prediction result and add each result to the 
characters array then we read this array sequentially and print the expression.
![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/a7600e93-4a15-4a75-a8cd-ecd195e91b4d)
In this section, some sample pictures are determined and the result is obtained. SequNet 
model needs to be developed for some special images. For example, in this image the bracket 
did not match the bracket index in the category array. Instead of matching the bracket index, it 
gave wrong results like 6 or 0.
![image](https://github.com/kubraca/Recognize-Handwritten-Basic-Math-Symbols/assets/72304467/e0ca360c-e443-4ecb-b13c-4f2ed40ddb22)



