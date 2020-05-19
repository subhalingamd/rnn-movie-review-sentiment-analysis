import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--review", required=True, help="Your movie review (within double quotes)")
ap.add_argument("-o", "--output",
	help="Flag to print the predicted value",action="store_true")
args = ap.parse_args()



import json
import re
import numpy as np


# For keras to stop echoing...
import os
import sys
stderr = sys.stderr

sys.stderr = open(os.devnull, 'w')

from keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

sys.stderr = stderr




model = load_model("model.h5")


with open('tokenizer.json') as f:
	data = json.load(f)
	tokenizer = tokenizer_from_json(data)


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
	return TAG_RE.sub('', text)

def preprocess_text(sen):
	# Removing html tags
	sentence = remove_tags(sen)

	# Remove punctuations and numbers
	sentence = re.sub('[^a-zA-Z]', ' ', sentence)

	# Single character removal
	sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

	# Removing multiple spaces
	sentence = re.sub(r'\s+', ' ', sentence)

	return sentence



# DO NOT CHANGE THIS!!!
vocab_size = len(tokenizer.word_index) + 1
maxlen = 256

X =[]

# x = input("Enter your review:\n\t")
x = args.review
print("\n\nYour Review:\n"+str(x))

X.append(preprocess_text(x))
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, padding='post', maxlen=maxlen)


y = model.predict(X)


Y = np.round(y)[0]


if Y == 0:
	print("\nOops... Was that a negative review?")
else :
	print("\nThat looked like a positive review... Hope you enjoyed the movie :) \n")


if args.output:
	print("** Predicted:  %f (Note: Displaying this line because you had used --output)\n\n" % y[0])