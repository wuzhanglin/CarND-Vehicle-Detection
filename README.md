# Self-Driving Car Project 5. Vehicle Detection

## Overview
In this project, my goal is to write a pipeline to detect vehicles in a video.

The test data includes two videos test_video.mp4 and a full project_video.mp4.  

## Project Details
The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector
* Note: for the first two steps I will try to normalize the features and randomize a selection for training and testing
* Implement a sliding-window technique and use my trained classifier to search for vehicles in images
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train a classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. We can also try to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment the training data.  

Some example images for testing the pipeline on single frames are located in the `test_images` folder. I will produce and save examples of the output from each stage of my pipeline in the folder called `ouput_images`. The video called `project_video.mp4` is the video my pipeline will work well on.

