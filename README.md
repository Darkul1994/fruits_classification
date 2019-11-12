# Fruits images classification using Convolutional Neural Networks
> The aim of this project is to build a Convolutional Neural Networks (CNN) for fruits images classification. In 
> particular, the CNN developed for this project will attempt to predict a fruit or vegetable variety in a given photo.

## Table of contents
* [General info](#general-info)
* [Data set](#data-set)
* [Technologies](#technologies)
* [Setup](#setup)
* [Status](#status)

## General info
Add more general information about project. What the purpose of the project is? Motivation?

## Data set
Images used to train neural nets are obtained from the the website [@www.kaggle.com/moltean/fruits](https://www.kaggle.com/moltean/fruits).
The fruits 360 data set on Kaggle consists of 82213 images of 120 fruits. In this project are included the following 
fruits: Apple Red Yellow 2, Banana Red, Cherry 2, Grape Blue, Lemon, Peach 2, Plum 3, Strawberry Wedge, Tomato 1 and 
Walnut.

Training set size: 7225 images

Validation set size: 2359 images

Number of classes: 10 (fruits):
* 0 - Apple Red Yellow 2
* 1 - Banana Red
* 2 - Cherry 2
* 3 - Grape Blue
* 4 - Lemon
* 5 - Peach 2
* 6 - Plum 3
* 7 - Strawberry Wedge
* 8 - Tomato 1
* 9 - Walnut

Image size: 100x100 pixels

## Technologies
This project is created with:
* Python - version 3.7

## Setup
Create your virtualenv and install project requirements:
`pip install -r requirements.txt`
Run main program with:
`python mainy.py`
TODO add parameter to train
`python main.py --train=True`
`python main.py train`

## Status
Project is: _finished_

## Results
Przetrenowalas taka i taka siec, wykresy, wyniki, plot confusion matrix
