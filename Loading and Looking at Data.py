import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist # imports the fashion dataset

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_labels[0])  # outputs a number between 0-9 which tells us the classification

"""
Label       Class               Label       Class
0           T-shirt/top         5           Sandal   
1           Trouser             6           Shirt
2           Pullover            7           Sneaker
3           Dress               8           Bag
4           Coat                9           Ankle Boot
"""

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]       # attaches names to the classes

print(class_names[train_labels[0]]) # The clothing at index 0 is an Ankle Boot

# print(train_images[0])  # Shows us each pixel value in the image from 0-255

"""
Images are arrays of 28x28 pixels with each index having a value from 0 (being the color white) to 255 (being black).
This is fine but it's better to shrink our data so it's within a smaller range. In order to do that here, all we need
to do is divide the data by 255
"""

train_images = train_images/255.0   # shrinks the data so it's between 0-1
test_images = test_images/255.0

# print(train_images[0])    # The numbers here are now much smaller than before but the pictures are still the same

# plt.imshow(train_images[0]) # cmap=plt.cm.binary will make the picture look a little better
# plt.show()  # shows us the image at index 0 which is in fact an Ankle Boot

"""
Now we're going to create a model with 784 input neurons, 128 hidden neurons (arbitrary number), and 10 output neurons 
(classes from 0-9)
Sequential is just a sequence of layers (defining each layer in order)
Whenever you have a 2D or 3D array, you need to flatten (aka convert matrix to a single array) the data so it can pass 
to an individual neuron as opposed to sending a whole list to one neuron
A dense layer is a fully connected layer (each neuron is connected to every other neuron)
relu = rectified linear unit function and is the most widely used activation function
softmax is used as an activation function for the output layer and calculates relative probabilities so all values will
add up to 1
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# We're going to be looking at the accuracy when testing the model or how low we can get the loss function
model.fit(train_images, train_labels, epochs=5) # epochs is how many times the model will see the information

test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("\nTest Accuracy: ", test_acc)

prediction = model.predict(test_images) # outputs list of predictions for classes (0-9)

print(prediction[0])
print(np.argmax(prediction[0]), class_names[np.argmax(prediction[0])])
# argmax gets the largest number aka the prediction that is most likely correct (ankle boot)

plt.figure(figsize=(5, 5))  # This is all just setting up the graph
for i in range(5):  # number of images cycled through
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual: {class_names[test_labels[i]]}")
    plt.title(f"Prediction: {class_names[np.argmax(prediction[i])]}")
    plt.show()



