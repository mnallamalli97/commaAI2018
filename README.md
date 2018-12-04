# commaAI2018
This is my solution and report for the 2018 internship at Comma AI. 

ORIGINAL FRAME: 


1. I am cropping image to remove black spots and crop out sky

2. I am taking each frame and augmenting it by
	> edge detection
	> adding noise
	> adding brigthness to the 2nd channel layer of frame
	> sharpening the image
  
  INSERT CLEANED

3. I run these steps as my preprocessing and data cleansing

4. I made this clean data because I found a optical flow method that will calculate change in frame, so enhances pixels for more accurate pixel difference between frames. 

  INSERT OPTICAL FLOW

5. Then I attached the speeds(Yaxis) given from the training video frames to the optical flow frames(Xaxis)

6. Then I trained a model I made (I am calling it mehar_model()) by fitting these points in n dimensional grid
	> The model I made is a Keras model running on top of tensor flow. It is a CNN.
  
--
Results: 
  > The model learned from the training data  has a MSE of <4.0 with a score of 0.95. My goal at the beginning of this project was to minimize the loss to be <1.0, and I am happy to say I was able to accomplish that. 
  
  INSERT FIRST HERE
  
  As the number of epoches increased, the total loss decreased from 4,000 to 4.00. This is a 1000% improvement in the training data. 
  
  INSERT LAST HERE
  

Improvements & Future Extensions:

1. I want to reduce the bias variance and prevent any overfitting by using a K-fold validation.
2. I want to train the model using more frames because the model will be able to learn more from more optical flow frames
3. I want to embedd this project solution into my ISEF winning project to track driver speed (rather than using GPS sensors).
