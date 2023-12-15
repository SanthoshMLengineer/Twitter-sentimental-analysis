import re
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import pickle


def lower_casing(sentence):
    """
    This function is used to convert string into 
    lower case
    Parameters
    ----------
      sentence : str
        input string
    Returns
    --------
      lower_cased_sentence : str
        Lower cased sentence
    """
    lower_cased_sentence = " ".join([word.lower() \
                                     for word in sentence.split()])
    print("lower case completed",sentence)
    return lower_cased_sentence


def remove_unwanted_char(sentence):
    """
    This function is to remove unwanted characters
    Parameters
    ----------
    sentence : str
      input string
    Returns
    --------
    processed_sentences : str
      processed string s
    """
    processed_sentences = " ".join([re.sub('[0-9]','digit', str(word)) \
                                    for word in sentence.split()])
    
    processed_sentences = " ".join([re.sub('[^a-zA-Z]','', str(word)) \
                                    for word in processed_sentences.split()])
    
    processed_sentences = " ".join([re.sub('[^a-zA-Z]','', str(word)) \
                                    for word in processed_sentences.split()])
    
    processed_sentences = " ".join([re.sub('user','', str(word)) \
                                    for word in processed_sentences.split()])
    
    processed_sentences = " ".join([re.sub('amp','', str(word)) \
                                    for word in processed_sentences.split()])
    
    return processed_sentences


def remove_stop_words(sentences):
    """
    This function is to remove stop words
    Parameters
    ----------
    sentences : str
      input string
    Returns
    --------
    sentence_processed : str
      processed string s
    """
    stop = stopwords.words('english')
    sentence_processed =  " ".join(i for i in sentences.split() if i not in stop)
    return sentence_processed


def get_root_words(sentences):
    """
    This function is to get root words
    Parameters
    ----------
    sentences : str
      input string
    Returns
    --------
    root_worded_sentence : str
      processed string 
    """
    lemmatizer = WordNetLemmatizer()
    root_worded_sentence = " ".join([lemmatizer.lemmatize(word) for word in sentences.split()])
    return root_worded_sentence


def text_preprocess(sentence):
  """
  This function helps to preprocess text
  Parameter
  ---------
    text : list()
      list of sentences
  Returns
  --------
    sentences : list()
      list of preprocessed sentences
  """
  try:
    lemmatizer = WordNetLemmatizer()
    print(sentence)
    sentence = lower_casing(sentence)
    sentence = remove_unwanted_char(sentence)
    sentence = remove_stop_words(sentence)
    sentence = get_root_words(sentence)
    return sentence
  except Exception as e:
    print(e)
    return sentence


def text_encoding(text):
	"""
	This function is used to convert
	text to encoder
	Parameter
	----------
		text : str
			Input text
	Returns
	--------
		text_vector : numpy.array()
			vector of text
	"""
	vectoriser_path = ".//..//Data Files//vectorizer.pkl"
	with open(vectoriser_path, "rb") as file_pointer:
		vectorizer = pickle.load(file_pointer)

	text_vector = vectorizer.transform([text]).toarray()

	return text_vector


output_dict = {
	0 : "Positive",
	1 : "Negative"
}


def prediction_from_model(text):
	"""
	This function is to get output from
	the model.
	Parameter
	----------
		text : str
			input text
	Return
	-------
		display_message : str
			Message to display on screen 
	"""
	model_path = ".//..//Data Files//Random_forest.pkl"
	with open(model_path, "rb") as file_pointer:
		random_forest_model = pickle.load(file_pointer)

	text = lower_casing(text)
	text = remove_unwanted_char(text)
	print("removed unwanted char", text)
	text = remove_stop_words(text)
	print("remove stop words", text)
	text = get_root_words(text)

	#text = text_preprocess(text)

	text_vector = text_encoding(text)

	output_model = random_forest_model.predict(text_vector)

	output_string = output_dict[output_model[0]]

	return output_string

