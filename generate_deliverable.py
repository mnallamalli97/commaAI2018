import argparse
import tensorflow as tf
import numpy as np
import cv2
import imgaug as ig
from imgaug import augmenters as iga
from keras.models import  Sequential
from keras.layers.convolutional import  Convolution2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model
import pickle as pkl


'''constants'''
BASE_TRAINING_VIDEO = 'train_original.mp4'
BASE_TESTING_VIDEO = 'test.mp4'

PROCESSED_TRAINING = 'cleaned_data/clean_training_img'
PROCESSED_TESTING = 'cleaned_data/clean_testing_img'

TRAINING_FRAMES = 20400

NEW_FRAME_HEIGHT = 66
NEW_FRAME_WIDTH = 220
NEW_FRAME_CHANNELS = 3

'''
I want each frame to be as clear as possible during the optical flow step.
> I am using the imgaug module to detect edges, add noise, and sharpen each frame before brightening the 2nd channel
> This bit of code is taken from the imgaug python documentation from: https://github.com/aleju/imgaug
> The parameters were established using the paper: https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085
'''
sometimes = lambda aug: iga.Sometimes(0.5, aug)
aug = iga.SomeOf(1, [
    sometimes(iga.OneOf([
        iga.Dropout((0.01, 0.1)),
        iga.CoarseDropout((0.03, 0.07),
                          size_percent=(0.03, 0.15)),
    ])),
    sometimes(iga.DirectedEdgeDetect(0.4)),
    sometimes(iga.AdditiveGaussianNoise(loc=0,
                                        scale=(0.3, 0.7))),
    sometimes(iga.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
    sometimes(iga.Add((-7, 7)))])


'''Preprocessing, getting the frames ready for ML model so better info comes from them, more accurate'''

def resize_frame(frame):
    '''
    > The input image is 480 by 640 by 3, but this consists of a lot of extraneous data.
    > I want to resize this input to be 220 by 70 by 3.
        > I got this these values by setting a mask on the entire video in photoshop. I cropped the mask until only
          the valuable pixels needed were filtered (ie, removing black spots, crop out sky).
    '''
    image_cropped = frame[100:440, :-90]
    image = cv2.resize(image_cropped, (220, 66), interpolation=cv2.INTER_AREA)
    return image


def change_brightness_frame(image, amount):
    '''
    > Changes the pixel intensities of each frame.
    > After researching online, I found that by multiplying the saturation by some value,
      we can change the brightness of the 2nd channel of the image.
    > The output will be the same image, but augmented brightness.
    > Once we change the brightness on the 2nd channel, we need to reconvert the hsv image to a rgb
    '''

    bright_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_image[:,:,1] = bright_image[:,:,1] * amount

    converted_image = cv2.cvtColor(bright_image, cv2.COLOR_HSV2RGB)
    return converted_image

def optical_flow(image_current, image_next):
    '''
    > I am using the Farneback method to calculate the dense optical flow.
    > I am using Farneback specifically because I can calculate each pixel point in the current frame to each pixel point
      in the next frame.

    method taken from (https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/)
    '''
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    # flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow

def preprocess(path_to_video):
    cleaned_frames = []
    video_frames = cv2.VideoCapture(path_to_video)
    success, img = video_frames.read()
    count = 0
    print("preprocessing frames...")
    while success:
        amt = 0.2 + np.random.uniform()
        img = resize_frame(img)

        img = aug.augment_image(img)
        img = change_brightness_frame(img, amt)
        cv2.imwrite("cleaned_test_data/frame %d.jpg" % count, img)  # save frame as JPEG file
        cleaned_frames.append(img)
        count += 1
        success, img = video_frames.read()

    return cleaned_frames

def optical_flow_cleaned_frames_frames(cleaned_frames):
    count = 0
    optical_flow_frames = []
    print("computing optical flow between frames...")
    for i in range(len(cleaned_frames[:-1])):
        img = optical_flow(cleaned_frames[i], cleaned_frames[i+1])
        cv2.imwrite("optical_flow_frames_test_data/frame%d.jpg" % count, img)  # save frame as JPEG file
        optical_flow_frames.append(img)
        count+=1

    return optical_flow_frames

def optical_flow_cleaned_frames_speed(true_speeds):
    count = 0
    optical_flow_speeds = []
    print("computing optical flow speeds...")
    for i in range(len(true_speeds[:-1])):
        true_mean = np.mean([true_speeds[i],true_speeds[i+1]])
        optical_flow_speeds.append(true_mean)
        count+=1

    return optical_flow_speeds


'''this is the neural net part'''

def create_data_for_model_frames(optical_flow_frames, batch_size=32):
    print("Shuffling frame data")
    frame_batch = []

    randomize = np.arange(len(optical_flow_frames))
    np.random.shuffle(randomize)
    list(map(lambda i: frame_batch.append(optical_flow_frames[randomize[i]]), range(len(optical_flow_frames))))

    return frame_batch

def create_data_for_model_speeds(optical_flow_speeds, batch_size=32):
    print("Shuffling speed data")
    label_batch = []

    randomize = np.arange(len(optical_flow_speeds))
    np.random.shuffle(randomize)
    list(map(lambda i: label_batch.append(optical_flow_speeds[randomize[i]]), range(len(optical_flow_speeds))))

    return label_batch

def mehar_model():
    input_frame= (NEW_FRAME_HEIGHT,NEW_FRAME_WIDTH,NEW_FRAME_CHANNELS)
    model = Sequential()

    model.add(Convolution2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=input_frame))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (8, 8), padding='same', strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (4, 4), padding='same', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (2, 2), padding='same', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=opt,loss='mse')
    return model


def train_model(model, x, y):
    # for i in range(len(x)):
    # x = list(map(lambda f: f.reshape((1,)+f.shape), x))
    #x = x.reshape((20399, )+x.shape)
    # np.reshape(x[i], (1, NEW_FRAME_HEIGHT, NEW_FRAME_WIDTH, NEW_FRAME_CHANNELS))
    model.fit(x, y, epochs=150, batch_size=32)
    # x[i] = x[i].reshape((1,) + x[i].shape)
    scores = model.evaluate(x, y, verbose=1)
    print("score: ",scores)
    model.save('modelv1.h5')  # creates a HDF5 file 'my_model.h5'

def test_model(h5_file_path, X_test):
    print("starting testing...")
    model = load_model(h5_file_path)
    prediction = model.predict([X_test], verbose=1)

    np.savetxt("testv2.txt", prediction,newline='\n')
    #
    # print("writing to test.txt")
    # f = open("test.txt", "w+")
    #
    # for i in range(len(prediction)):
    #     f.write(prediction + '\n')
    # f.close()


def main():

    cleaned_frames = preprocess('data/test.mp4')
    cleaned_frames = optical_flow_cleaned_frames_frames(cleaned_frames)
    print("finished with cleaning test data step.")
    print("generating model...")
    X = create_data_for_model_frames(cleaned_frames)
    #X is frame, Y is speed

    print("loading testing frames pickle")
    with open("testing_frames.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle))


    print("creating results.txt file...")
    test_model("modelv2.h5", X)







if __name__ == '__main__':
    main()

