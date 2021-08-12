# Artist-Identification

The Task is to identify the artist of painting.
Return the prediction through simple Rest API

## Challenges
* Small data set
* Quick predictions

## Research approaches
* Use Pre trained network (VGG19), and adopt the loss from [style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
    * My Assumption is that every artist has is special style, and like that we can identify them.
* Use Pre trained network (Didn't decide yet) and fine tuned them. 
    * Will use siamese network and triplet loss to achieve good representations for each painter.
    * We will need to use a lot of augmentations because of the size of the data set  

## Data
6 artist, each artist has 9 painting

## How To Use
### Create Features
In order to create feature maps, you should run the following command.

``
python main.py --action train
``
This features will help to classify each image

### Evaluation
In order to eval our model on the test set, you sould run the following command.

``
python main.py --action eval
``

### Server
Will be published when the code will be done.