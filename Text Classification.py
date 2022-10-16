import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
In this we can classify movie reviews to determine whether they are negative or positive. We will use a keras dataset
of IMDB movie reviews to make our model then use test.txt to classify a review that our program has never seen before.

Classification is output as a number between 0 and 1. If the output is closer to 0 then it's classified as a bad review
but if it's closer to 1 then it's classified as a positive review.
"""

imdb = keras.datasets.imdb  # imdb = data we're using

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=880000)  # takes 10000 most frequent words (updated to 880000)
# Note: number of words was changed to 88000 for the test.txt stuff

"""The following print statement will give out a list of numbers between 1 and 10000 where each number correlates with
each word in the review"""
# print(train_data[0])

word_index = imdb.get_word_index()  # dictionary where words are mapped to numbers ("Hello": 4206)

word_index = {k: (v+3) for k, v in word_index.items()} # for key in dict: remap key to value + 3

"""If we get values that aren't valid then we add them to the following. This is why we move all the values up 3
spaces in the line of code above."""

word_index["<PAD>"] = 0 # used to make all reviews the same amount of words long by adding "padding" words
word_index["<START>"] = 1   # will automatically be added at the beginning of each review
word_index["<UNK>"] = 2 # Unknown
word_index["<UNUSED>"] = 3

# Right now, we have words as keys and numbers as values so we want to reverse that
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Since not all reviews are the same length, we need to add or subtract words from them to make it all the same
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
    """This function returns what words in our word_index are used in the review (text)"""
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode_review(train_data[0]))

# Model down here
"""
Embedding determines the meaning of a word in a sentence and maps it to a position vector so words like "awesome",
"good", and "great" will be placed close to each other
GlobalAveragePooling1D just scales down our data's dimensions to make computation easier
Dense layers are fully connected and the relu one find patterns between different words in the review, while the sigmoid
outputs whether the review is positive or negative (output will be between 0 and 1)"""

"""
model = keras.Sequential()
model.add(keras.layers.Embedding(880000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary() # prints summary of model

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]  # we'll be using validation data so the model doesn't memorize our data's answers
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Number of words was changed to 88000 from 10000 but the number wasn't changed above

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5") # h5 is an extension for saved models in Keras and Tensorflow"""
# After saving the model we can comment out the above code
model = keras.models.load_model("model.h5")


def review_encode(s):
    """This function encodes any custom reviews that weren't in our original dataset"""
    encoded = [1] # start at 1 to stay consistent with our data. See word_index["<START>"] = 1 up above

    for word in s:
        if word.lower() in word_index: # if word already exists in our word index
            encoded.append(word_index[word.lower()])    # append the words corresponding number to encoded
        else:
            encoded.append(2)   # Otherwise make word UNK aka unknown

    return encoded


with open("test.txt", encoding="utf-8") as f:   # Opens a new review that wasn't in our dataset
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

"""
# Testing our model
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print(f"Prediction: {predict[0]}")
print(f"Actual: {test_labels[0]}")
print(results)"""



