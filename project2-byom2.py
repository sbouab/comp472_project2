# Project2-BYOM2 code for the second project of COMP472 - AI class at Concordia University.
# Written by: Soufiane Bouabdallah - ID: 40029995
# Description: Second custom language identification model.

# import necessary libraries
import sys, string, copy, itertools
from collections import Counter
from math import log10
from decimal import Decimal

# hyperparameters that provided the best results
vocabulary_type = '2'
ngram_size = '3'
smoothing_value = 0.01
# new data set used for training
training_file_en = open('en.txt', 'r', encoding="latin-1")
training_data_en = [i.strip('\n') for i in training_file_en]
training_file_en.close()
training_file_es = open('es.txt', 'r', encoding="latin-1")
training_data_es = [i.strip('\n') for i in training_file_es]
training_file_es.close()
training_file_eu = open('eu.txt', 'r', encoding="latin-1")
training_data_eu = [i.strip('\n') for i in training_file_eu]
training_file_eu.close()
training_file_ca = open('ca.txt', 'r', encoding="latin-1")
training_data_ca = [i.strip('\n') for i in training_file_ca]
training_file_ca.close()
training_file_gl = open('gl.txt', 'r', encoding="latin-1")
training_data_gl = [i.strip('\n') for i in training_file_gl]
training_file_gl.close()
training_file_pt = open('pt.txt', 'r', encoding="latin-1")
training_data_pt = [i.strip('\n') for i in training_file_pt]
training_file_pt.close()
testing_file = open(sys.argv[1], 'r', encoding="latin-1")
# used for building the vocabulary in the case of isalpha() (following Moodle's FAQ)
testing_data = [i.strip('\n').split('\t') for i in testing_file]
testing_file.close()

# build vocabulary
vocabulary_dirty = set()
for element in training_data_en:
    for char in element:
        if char.isalpha():
            vocabulary_dirty.add(char)
for element in training_data_es:
    for char in element:
        if char.isalpha():
            vocabulary_dirty.add(char)
for element in training_data_eu:
    for char in element:
        if char.isalpha():
            vocabulary_dirty.add(char)
for element in training_data_ca:
    for char in element:
        if char.isalpha():
            vocabulary_dirty.add(char)
for element in training_data_gl:
    for char in element:
        if char.isalpha():
            vocabulary_dirty.add(char)
for element in training_data_pt:
    for char in element:
        if char.isalpha():
            vocabulary_dirty.add(char)
# in order to not have any suprises when running the code on the test file
for tweet in testing_data:
    for char in tweet[3]:
        if char.isalpha():
            vocabulary_dirty.add(char)

# clean vocabulary (remove any non-latin characters, keep lowercase only and add space character)
vocabulary = set()
for element in vocabulary_dirty:
    try:
        element.encode('latin-1')
        vocabulary.add(element.lower())
    except:
        pass
vocabulary.remove('å'), vocabulary.remove('ð'), vocabulary.remove('ª'), vocabulary.remove('þ'), vocabulary.remove('µ'), vocabulary.remove('º')
vocabulary.add(' ')

# build ngrams
counter_es, counter_pt, counter_en, counter_eu, counter_gl, counter_ca = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
combinations = list(itertools.product(vocabulary, repeat=3))
empty_ngram_model = dict.fromkeys(combinations, 0)
counter_es.update(empty_ngram_model), counter_pt.update(empty_ngram_model), counter_en.update(empty_ngram_model), counter_eu.update(empty_ngram_model), counter_gl.update(empty_ngram_model), counter_ca.update(empty_ngram_model)

# [ TRAIN ]

# function that returns the filtered ngrams that are used to populate the ngram counter
def populate_ngram_dictionary(data):
    dirty = [data[i:i+3] for i in range(len(data)-3+1)]
    filtered = copy.deepcopy(dirty)
    for trigram in dirty:
        if trigram[0] not in vocabulary or trigram[1] not in vocabulary or trigram[2] not in vocabulary:
            filtered.remove(trigram)
    filtered[:] = [tuple(trigram) for trigram in filtered]
    return filtered

for element in training_data_en:
    filtered = populate_ngram_dictionary(element)
    counter_en.update(filtered)
for element in training_data_es:
    filtered = populate_ngram_dictionary(element)
    counter_es.update(filtered)
for element in training_data_eu:
    filtered = populate_ngram_dictionary(element)
    counter_eu.update(filtered)
for element in training_data_ca:
    filtered = populate_ngram_dictionary(element)
    counter_ca.update(filtered)
for element in training_data_gl:
    filtered = populate_ngram_dictionary(element)
    counter_gl.update(filtered)
for element in training_data_pt:
    filtered = populate_ngram_dictionary(element)
    counter_pt.update(filtered)

# total number of ngrams in each language is calculated first (with smoothing value)
# get the posteriors for each ngram in each language
def calculate_posterior(counter):
    total_ngrams = sum(counter.values()) + smoothing_value*len(counter)
    for k,v in counter.items():
        counter[k] = log10((v+smoothing_value)/total_ngrams)

# calculates the posteriors for each ngram in a language (in log10)
calculate_posterior(counter_es), calculate_posterior(counter_pt), calculate_posterior(counter_en), calculate_posterior(counter_eu), calculate_posterior(counter_gl), calculate_posterior(counter_ca)

# [ TEST ]

# calculates probability of the tweet for the language provided
# added language-specific characteristics which increase the probability for some languages
def calculate_probability(tweet, counter):
    probability = 0
    if ('¿' in tweet or '¡' in tweet) and counter == counter_es:
        probability += 100
    if ('á' in tweet) and (counter == counter_es or counter == counter_gl or counter == counter_pt):
        probability += 100
    if ('é' in tweet or 'í' in tweet or 'ó' in tweet or 'ú' in tweet or 'ü' in tweet) and (counter == counter_es or counter == counter_gl or counter == counter_pt or counter == counter_ca):
        probability += 100
    if ('tz' in tweet or 'tx' in tweet) and (counter == counter_ca or counter == counter_eu):
        if counter == counter_ca:
            probability += 50
        elif counter == counter_eu:
            probability += 100
    if ('ã' in tweet or 'õ' in tweet or 'â' in tweet or 'ê' in tweet or 'ô' in tweet) and (counter == counter_pt):
        probability += 100
    dirty = [tweet[i:i+3] for i in range(len(tweet)-3+1)]
    filtered = copy.deepcopy(dirty)
    for trigram in dirty:
        if trigram[0] not in vocabulary or trigram[1] not in vocabulary or trigram[2] not in vocabulary:
            filtered.remove(trigram)
    filtered[:] = [tuple(trigram) for trigram in filtered]
    for trigram in filtered:
        probability += counter[trigram]
    return probability 

# accuracy variables
total_test_tweets = len(testing_data)
correct_answers = 0

# precision and recall variables
total_es, total_classified_as_es, correctly_classified_as_es = 0, 0, 0
total_pt, total_classified_as_pt, correctly_classified_as_pt = 0, 0, 0
total_en, total_classified_as_en, correctly_classified_as_en = 0, 0, 0
total_eu, total_classified_as_eu, correctly_classified_as_eu = 0, 0, 0
total_gl, total_classified_as_gl, correctly_classified_as_gl = 0, 0, 0
total_ca, total_classified_as_ca, correctly_classified_as_ca = 0, 0, 0

# keep track of precision and recall variables
def update_precision_and_recall_variables(classified_language, answer, correct_classification):
    if correct_classification == 'es':
        global total_es
        total_es += 1
    elif correct_classification == 'pt':
        global total_pt
        total_pt += 1
    elif correct_classification == 'en':
        global total_en
        total_en += 1
    elif correct_classification == 'eu':
        global total_eu
        total_eu += 1
    elif correct_classification == 'gl':
        global total_gl
        total_gl += 1
    elif correct_classification == 'ca':
        global total_ca
        total_ca += 1
    if classified_language == 'es':
        global total_classified_as_es
        total_classified_as_es += 1
        if answer == 'correct':
            global correctly_classified_as_es
            correctly_classified_as_es += 1
    elif classified_language == 'pt':
        global total_classified_as_pt
        total_classified_as_pt += 1
        if answer == 'correct':
            global correctly_classified_as_pt
            correctly_classified_as_pt += 1
    elif classified_language == 'en':
        global total_classified_as_en
        total_classified_as_en += 1
        if answer == 'correct':
            global correctly_classified_as_en
            correctly_classified_as_en += 1
    elif classified_language == 'eu':
        global total_classified_as_eu
        total_classified_as_eu += 1
        if answer == 'correct':
            global correctly_classified_as_eu
            correctly_classified_as_eu += 1
    elif classified_language == 'gl':
        global total_classified_as_gl
        total_classified_as_gl += 1
        if answer == 'correct':
            global correctly_classified_as_gl
            correctly_classified_as_gl += 1
    elif classified_language == 'ca':
        global total_classified_as_ca
        total_classified_as_ca += 1
        if answer == 'correct':
            global correctly_classified_as_ca
            correctly_classified_as_ca += 1

# noisy characters to remove from tweet
filter_characters = string.punctuation.replace('\'', '') + '0123456789'

# function that cleans the tweet from noise in order to improve the classifier
def clean_tweet(tweet):
    temp_dirty = tweet.split(' ')
    temp_clean = copy.deepcopy(temp_dirty)
    for element in temp_dirty:
        if 'http' in element or 'www' in element or '@' in element or '#' in element:
            temp_clean.remove(element)
            continue
    temp_str = ""
    tweet = (temp_str.join(temp_clean)).lower()
    global filter_characters
    tweet = tweet.translate(str.maketrans('', '', filter_characters))
    return tweet

# trace file
trace_file = open("trace_myModel2_"+vocabulary_type+"_"+ngram_size+"_"+str(smoothing_value)+".txt", "w+", encoding="latin-1")

# get the probabilities of each language for each tweet
for tweet in testing_data:
    tweet_id = tweet[0]
    tweet[3] = clean_tweet(tweet[3])
    prob_es, prob_pt, prob_en, prob_eu, prob_gl, prob_ca = calculate_probability(tweet[3], counter_es), calculate_probability(tweet[3], counter_pt), calculate_probability(tweet[3], counter_en), calculate_probability(tweet[3], counter_eu), calculate_probability(tweet[3], counter_gl), calculate_probability(tweet[3], counter_ca)
    language_probabilities = {'es': prob_es, 'pt': prob_pt, 'en': prob_en, 'eu': prob_eu, 'gl': prob_gl, 'ca': prob_ca}
    classified_language = max(language_probabilities, key=language_probabilities.get)
    classified_value = '%.2E' % Decimal(language_probabilities[classified_language])
    correct_classification = tweet[2]
    if correct_classification == classified_language:
        answer = "correct"
        correct_answers += 1
    else:
        answer = "wrong"
    update_precision_and_recall_variables(classified_language, answer, correct_classification)
    trace_file.write(tweet_id + "  " + classified_language + "  " + str(classified_value) + "  " + correct_classification + "  " + answer + "\n")

trace_file.close()

# avoid division by 0
def new_division(div1, div2):
    return div1 / div2 if div2 else 0.0

# calculate accuracy, per-class precision, per-class recall, per-class F1-measure, macro-F1 and weighted-average-F1
accuracy = correct_answers / total_test_tweets
precision_es, precision_pt, precision_en, precision_eu, precision_gl, precision_ca = new_division(correctly_classified_as_es, total_classified_as_es), new_division(correctly_classified_as_pt, total_classified_as_pt), new_division(correctly_classified_as_en, total_classified_as_en), new_division(correctly_classified_as_eu, total_classified_as_eu), new_division(correctly_classified_as_gl, total_classified_as_gl), new_division(correctly_classified_as_ca, total_classified_as_ca)
recall_es, recall_pt, recall_en, recall_eu, recall_gl, recall_ca = new_division(correctly_classified_as_es, total_es), new_division(correctly_classified_as_pt, total_pt), new_division(correctly_classified_as_en, total_en), new_division(correctly_classified_as_eu, total_eu), new_division(correctly_classified_as_gl, total_gl), new_division(correctly_classified_as_ca, total_ca)

# make sure we avoid division by 0 when dividing
def calculate_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

f1_es, f1_pt, f1_en, f1_eu, f1_gl, f1_ca = calculate_f1(precision_es, recall_es), calculate_f1(precision_pt, recall_pt), calculate_f1(precision_en, recall_en), calculate_f1(precision_eu, recall_eu), calculate_f1(precision_gl, recall_gl), calculate_f1(precision_ca, recall_ca)
macro_f1 = (f1_es + f1_pt + f1_en + f1_eu + f1_gl + f1_ca) / 6
weighed_average_f1 = (total_es * f1_es + total_pt * f1_pt + total_en * f1_en + total_eu * f1_eu + total_gl * f1_gl + total_ca * f1_ca) / total_test_tweets

# evaluation file
evaluation_file = open("eval_myModel2_"+vocabulary_type+"_"+ngram_size+"_"+str(smoothing_value)+".txt", "w+", encoding="latin-1")

evaluation_file.write(str("%.4f" % accuracy) + "\n" + str("%.4f" % precision_eu) + "  " + str("%.4f" % precision_ca) + "  " + str("%.4f" % precision_gl) + "  " + str("%.4f" % precision_es) + "  " + str("%.4f" % precision_en) + "  " + str("%.4f" % precision_pt) + "\n" + str("%.4f" % recall_eu) + "  " + str("%.4f" % recall_ca) + "  " + str("%.4f" % recall_gl) + "  " + str("%.4f" % recall_es) + "  " + str("%.4f" % recall_en) + "  " + str("%.4f" % recall_pt) + "\n" + str("%.4f" % f1_eu) + "  " + str("%.4f" % f1_ca) + "  " + str("%.4f" % f1_gl) + "  " + str("%.4f" % f1_es) + "  " + str("%.4f" % f1_en) + "  " + str("%.4f" % f1_pt) + "\n" + str("%.4f" % macro_f1) + "  " + str("%.4f" % weighed_average_f1) + "\n")

evaluation_file.close()