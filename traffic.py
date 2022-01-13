import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

NUM_CAT_TEST = 3


def main():
    
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
     
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir): 
    images = list ()
    labels = list ()

    # the images and lables are all placed in their respective lists
    for i in range ( NUM_CATEGORIES) :
        joiner = os.path.join ( data_dir, str(i) ) 
        for filename in os.listdir("{}".format(joiner) ):
            imgjoiner= os.path.join ( joiner, filename )
            img = cv2.imread(imgjoiner)
            imgresize = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT) )
            # the function is completely normalized to be between 0 and 1
            imgresize = imgresize/ 255
            images.append( imgresize )
            labels.append( i )

    return ( images, labels) 
        
    
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    raise NotImplementedError


def get_model():
    inputdim= (IMG_WIDTH,IMG_HEIGHT,3)
    pool = ( 2,2)    
    # this is the convolution sequence 
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3) ,activation = 'relu' ,
    input_shape= inputdim ) ,
    tf.keras.layers.MaxPooling2D(pool_size=pool )  ,
    tf.keras.layers.Conv2D( 64, (3,3), activation = 'relu' ) , 
    tf.keras.layers.MaxPooling2D(pool_size=pool) 
       
    ])
    # first flatten the layers and then add hidden layer of 256 nodes 
    model.add(tf.keras.layers.Flatten () )
    model.add(tf.keras.layers.Dense(256,activation = 'relu' ))
    model.add(tf.keras.layers.Dropout(0.5)  )

    # the final softmax activation function is returned and the model is printed
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax' ))
    model.summary()
    model.compile ( optimizer = 'adam',
    loss = 'categorical_crossentropy' ,
    metrics = ['accuracy']
    )

    return model 
    
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
