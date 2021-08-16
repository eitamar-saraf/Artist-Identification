# Artist Identification

The Task is to identify the artist of painting.

Return the prediction through simple Rest API

## Challenges

* Small data set
* Quick predictions

## Research approaches

* Use Pre trained network (VGG19), and adopt the loss
  from [style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
    * My Assumption is that every artist has is special style, and like that we can identify them.
* Use Pre trained network (Didn't decide yet) and fine tuned them.
    * Will use siamese network and triplet loss to achieve good representations for each painter.
    * We will need to use a lot of augmentations because of the size of the data set

#### I Implemented only the first option

## Data

6 artist, each artist has 9 painting

You can find the data in the next directory:

{Project Root}/data/raw_data

## How To Use

### K Fold

In Order to Find the Best hyper params, like:

* style loss weight
* content loss weight
* weight for each layer in the style loss

You should run the following code:

``
python main.py --action kfold
``

This Action Will preform the following steps:

* Splitting the data to test and train
* Splitting the train to 3 random folds
* extract features from 2 folds and predict on the third fold.

It's little different from the classic k fold, but mimics the same logic.

#### Be Aware, Because of the size of the dataset, the results will be noisy.

### Train

In order to Train our model on the Train set, you should run the following command.

``
python main.py --action train
``

This Action Will preform the following steps:

* Split the data to train and test
* Calculate the feature maps for each art painting
* Evaluate the test set
    * Extract feature maps the art painting
    * Calculate loss(style + content) from each artist
    * Convert loss to probabilities
    * Predict the artist

### Server
``
flask run
``
or
``
python -m flask run
``

The Server will run on localhost:5000

#### Requests

##### GET /

&nbsp;&nbsp;&nbsp;&nbsp;Home Page

##### GET /paintings

&nbsp;&nbsp;&nbsp;&nbsp;List of all painting in the storage

##### GET /predict

&nbsp;&nbsp;&nbsp;&nbsp;Painting to predict

###### BODY

{

"artwork": "{name from /painting response}"

}
