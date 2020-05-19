# Movie Review Sentiment Classifier
***- By [Subhalingam D](https://subhalingamd.github.io)***


The model has been trained using the [IMDB dataset obtained from Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download). [Pretrained word embeddings from GloVe](http://nlp.stanford.edu/data/glove.6B.zip) were used with 2 LSTM layers to predict `0` for negative review and `1` for a positive one. 

The whole model was trained in Python (using Notebook in Colab). Keras was used primarily with Tensorflow backend.

### Prediction

It's straightforward to use `predict.py` to predict with your own review, unless you catch an error message!

#### Requirments
The basic dependencies (after installing python) are `numpy`, `keras` with `tensorflow` backend. The same can be installed using the following Terminal command:

```
pip install -r requirements.txt
```

**Make sure you have `model.h5` and `tokenizer.json` in the same directory as `predict.py`.** You can get all these files from my repo by [clicking here](https://github.com/subhalingamd/rnn-movie-review-sentiment-analysis)

### Running the python code
`predict.py` requires one mandatory and accepts one optional command-line argument. You can run the file by the following *template* command:

```
python predict.py -r "<REVIEW>" [-o]

```
* `-r` should be followed by the review **within double quotes**
* `-o` if present, will display the prediction value (between 0 and 1) at the end

For example,

```
python predict.py -r "excellent story and what a finishing! so many twists and should i even say about the direction?"
```

```
python predict.py -r "it's a good movie from the story standpoint but the screenplay was pathetic" -o
```

##### How to interpret the predicted value when `-o` flag is used?

This is quite straightforward. The predicted value will be between 0 and 1 which will be rounded to the nearest integer. After that, `0` would mean a negative review whereas `1` would mean a positive review. 

So if the predicted value is 0.009686, it is very close to 0 and signifies that the predictor strongly believes that it is a negative review. However, if the predicted value is 0.574038, it would mean positive but that's an intermediate value. 

In other words, this value would be a measure of confidence of the prediction-the extreme ends would mean a more strong prediction.

---

***Note:** This was an individual project done for learning purpose and several blogs were referred in the process. I would like to mention [this blog](https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/) specifically, which I used to learn how to preprocess data (removing punctuations and tokenising)*

**For any queries, contact Subhalingam at subhalingam.d@gmail.com**