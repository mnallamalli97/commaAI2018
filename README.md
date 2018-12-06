# commaAI2018
This is my solution and report for the 2018 internship at Comma AI. My entire solution is in the `generate_model.py` file and the `test.txt` file is made in the `generate_deliverable.py` file. 

If you would like only my deliverable, it is in `test.txt` as requested by the challenge description. 

Here is what I was given:

![alt text](https://github.com/mnallamalli97/commaAI2018/blob/master/pics_for_readme/original.png "given frame")

## High level Design: 

1. I am cropping image to remove black spots and crop out sky

2. I am taking each frame and augmenting it by
	+ edge detection
	+ adding noise
	+ adding brigthness to the 2nd channel layer of frame
	+ sharpening the image
  
  Here is my cleaned version:
  
  ![alt text](https://github.com/mnallamalli97/commaAI2018/blob/master/pics_for_readme/cleaned.jpg "cleaned frame")

3. I run these steps as my preprocessing and data cleansing

4. I made this clean data because I found a optical flow method that will calculate change in frame, so enhances pixels for more accurate pixel difference between frames. 

Here is my difference in frames:

  ![alt text](https://github.com/mnallamalli97/commaAI2018/blob/master/pics_for_readme/frame9459.jpg "optical flow")

5. Then I attached the speeds(Yaxis) given from the training video frames to the optical flow frames(Xaxis)

6. Then I trained a model I made (I am calling it `mehar_model()`) by fitting these points in n dimensional grid
	+ The model I made is a Keras model running on top of tensor flow. It is a CNN.
  
## Results: 
+ The model learned from the training data  has a MSE of <4.0 with a score of 0.95. My goal at the beginning of this project was to minimize the loss to be <1.0, and I am happy to say I was able to accomplish that. 
  
  First epoch: 
  
  ![alt text](https://github.com/mnallamalli97/commaAI2018/blob/master/pics_for_readme/first.png "see loss after first epoch")
  
+ As the number of epoches increased, the total loss decreased from 4,000 to 4.00. This is a 1000% improvement in the training data. 
  
  Last epoch:
  
  ![alt text](https://github.com/mnallamalli97/commaAI2018/blob/master/pics_for_readme/last.png "see loss after 150th")
  

## Improvements & Future Extensions:

1. I want to reduce the bias variance and prevent any overfitting by using a K-fold validation.
2. I want to train the model using more frames because the model will be able to learn more from more optical flow frames
3. I want to embedd this project solution into my **ISEF** winning project to track driver speed (rather than using GPS sensors).

## Important Dependencies: 

1. tensorflow, keras 

## How to Run and test my solution:

1. If you are just looking for the deliverable, it is under `test.txt`
2. Before running the command in step 3, uncomment the code in the main and comment the existing main. You need to first generate the optical flow frames as those are the inputs to the training model. 
3. If you would like to save my model weights in an h5 file:
	+ run the command `python2 generate_model`
4. If you would like to generate the `test.txt`:
	+ run the command `python2 generate_deliverable`
	Note: Once you generated the model weights, make sure to load in the correct name of the model generated in step 2.
