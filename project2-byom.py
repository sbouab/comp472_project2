# Project2-BYOM code for the second project of COMP472 - AI class at Concordia University.
# Written by: Soufiane Bouabdallah - ID: 40029995
# Description: First custom language identification model.

# import necessary libraries
import sys, string, copy, itertools
from collections import Counter
from math import log10
from decimal import Decimal

# get the hyperparameters and the training & testing files from the arguments (for the BYOM I used V=2, N=3, D=0.01)
vocabulary_type = sys.argv[1]
ngram_size = sys.argv[2]
smoothing_value = float(sys.argv[3])
training_file = open(sys.argv[4], 'r', encoding="utf-8")
testing_file = open(sys.argv[5], 'r', encoding="utf-8")

training_data = [line.strip('\n').split('\t') for line in training_file]
# used for building the vocabulary in the case of isalpha() (following Moodle's FAQ)
testing_data = [line.strip('\n').split('\t') for line in testing_file]

# build vocabulary
if vocabulary_type == '0':
    vocabulary = set(string.ascii_lowercase)
elif vocabulary_type == '1':
    vocabulary = set(string.ascii_letters)
elif vocabulary_type == '2':
    vocabulary = set()
    # only consider lowercase characters to improve results (BYOM)
    for tweet in training_data:
        for char in tweet[3]:
            if char.isalpha():
                vocabulary.add(char.lower())
    # in order to not have any suprises when running the code on the test file
    for tweet in testing_data:
        for char in tweet[3]:
            if char.isalpha():
                vocabulary.add(char.lower())

# add a space to the vocabulary improve results (BYOM)
vocabulary.add(' ')

# build ngrams
counter_es, counter_pt, counter_en, counter_eu, counter_gl, counter_ca = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
if ngram_size == '1':
    empty_ngram_model = dict.fromkeys(vocabulary, 0)
elif ngram_size == '2':
    combinations = list(itertools.product(vocabulary, repeat=2))
    empty_ngram_model = dict.fromkeys(combinations, 0)
elif ngram_size == '3':
    combinations = list(itertools.product(vocabulary, repeat=3))
    empty_ngram_model = dict.fromkeys(combinations, 0)
counter_es.update(empty_ngram_model), counter_pt.update(empty_ngram_model), counter_en.update(empty_ngram_model), counter_eu.update(empty_ngram_model), counter_gl.update(empty_ngram_model), counter_ca.update(empty_ngram_model)

# [ TRAIN ]

total_tweets = len(training_data)
# number of tweets in each language in the training data
tweets_es, tweets_pt, tweets_en, tweets_eu, tweets_gl, tweets_ca = 0, 0, 0, 0, 0, 0

# function that returns the filtered tweet depending on the vocabulary and the ngram_size
def populate_ngram_dictionary(tweet):
    if vocabulary_type == '0':
        if ngram_size == '1':
            dirty = tweet.lower()
            filtered = copy.deepcopy(dirty)
            for unigram in dirty:
                if unigram not in vocabulary:
                    filtered = filtered.replace(unigram, '')
        elif ngram_size == '2':
            dirty = [tweet.lower()[i:i+2] for i in range(len(tweet.lower())-2+1)]
            filtered = copy.deepcopy(dirty)
            for bigram in dirty:
                if bigram[0] not in vocabulary or bigram[1] not in vocabulary:
                    filtered.remove(bigram)
            filtered[:] = [tuple(bigram) for bigram in filtered]
        elif ngram_size == '3':
            dirty = [tweet.lower()[i:i+3] for i in range(len(tweet.lower())-3+1)]
            filtered = copy.deepcopy(dirty)
            for trigram in dirty:
                if trigram[0] not in vocabulary or trigram[1] not in vocabulary or trigram[2] not in vocabulary:
                    filtered.remove(trigram)
            filtered[:] = [tuple(trigram) for trigram in filtered]
    elif vocabulary_type == '1' or vocabulary_type == '2':
        if ngram_size == '1':
            dirty = tweet
            filtered = copy.deepcopy(dirty)
            for unigram in dirty:
                if unigram not in vocabulary:
                    filtered = filtered.replace(unigram, '')
        elif ngram_size == '2':
            dirty = [tweet[i:i+2] for i in range(len(tweet)-2+1)]
            filtered = copy.deepcopy(dirty)
            for bigram in dirty:
                if bigram[0] not in vocabulary or bigram[1] not in vocabulary:
                    filtered.remove(bigram)
            filtered[:] = [tuple(bigram) for bigram in filtered]
        elif ngram_size == '3':
            dirty = [tweet[i:i+3] for i in range(len(tweet)-3+1)]
            filtered = copy.deepcopy(dirty)
            for trigram in dirty:
                if trigram[0] not in vocabulary or trigram[1] not in vocabulary or trigram[2] not in vocabulary:
                    filtered.remove(trigram)
            filtered[:] = [tuple(trigram) for trigram in filtered]
    return filtered

# update the number of tweets in each language
# update the quantity of ngrams in the vocabulary for each language
for tweet in training_data:
    if tweet[2] == 'es':
        tweets_es += 1
        filtered = populate_ngram_dictionary(tweet[3])
        counter_es.update(filtered)
    elif tweet[2] == 'pt':
        tweets_pt += 1
        filtered = populate_ngram_dictionary(tweet[3])
        counter_pt.update(filtered)
    elif tweet[2] == 'en':
        tweets_en += 1
        filtered = populate_ngram_dictionary(tweet[3])
        counter_en.update(filtered)
    elif tweet[2] == 'eu':
        tweets_eu += 1
        filtered = populate_ngram_dictionary(tweet[3])
        counter_eu.update(filtered)
    elif tweet[2] == 'gl':
        tweets_gl += 1
        filtered = populate_ngram_dictionary(tweet[3])
        counter_gl.update(filtered)
    elif tweet[2] == 'ca':
        tweets_ca += 1
        filtered = populate_ngram_dictionary(tweet[3])
        counter_ca.update(filtered)

# calculate the priors (number of tweets in language X / number of total tweets) in log10
prior_es, prior_pt, prior_en, prior_eu, prior_gl, prior_ca = log10(tweets_es/total_tweets), log10(tweets_pt/total_tweets), log10(tweets_en/total_tweets), log10(tweets_eu/total_tweets), log10(tweets_gl/total_tweets), log10(tweets_ca/total_tweets)

# total number of ngrams in each language is calculated first (with smoothing value)
# get the posteriors for each ngram in each language
def calculate_posterior(counter):
    if smoothing_value > 0:
        total_ngrams = sum(counter.values()) + smoothing_value*len(counter)
        for k,v in counter.items():
            counter[k] = log10((v+smoothing_value)/total_ngrams)
    elif smoothing_value == 0:
        total_ngrams = sum(counter.values())
        for k,v in counter.items():
            if v == 0:
                counter[k] = float('-inf')
            else:
                counter[k] = log10(v/total_ngrams)

# calculates the posteriors for each ngram in a language (in log10)
calculate_posterior(counter_es), calculate_posterior(counter_pt), calculate_posterior(counter_en), calculate_posterior(counter_eu), calculate_posterior(counter_gl), calculate_posterior(counter_ca)

# [ TEST ]

# calculates probability of the tweet for the language provided
def calculate_probability(tweet, prior, counter):
    probability = prior
    if vocabulary_type == '0':
        if ngram_size == '1':
            dirty = tweet.lower()
            filtered = copy.deepcopy(dirty)
            for unigram in dirty:
                if unigram not in vocabulary:
                    filtered = filtered.replace(unigram, '')
            for unigram in filtered:
                probability += counter[unigram]
        elif ngram_size == '2':
            dirty = [tweet.lower()[i:i+2] for i in range(len(tweet.lower())-2+1)]
            filtered = copy.deepcopy(dirty)
            for bigram in dirty:
                if bigram[0] not in vocabulary or bigram[1] not in vocabulary:
                    filtered.remove(bigram)
            filtered[:] = [tuple(bigram) for bigram in filtered]
            for bigram in filtered:
                probability += counter[bigram]
        elif ngram_size == '3':
            dirty = [tweet.lower()[i:i+3] for i in range(len(tweet.lower())-3+1)]
            filtered = copy.deepcopy(dirty)
            for trigram in dirty:
                if trigram[0] not in vocabulary or trigram[1] not in vocabulary or trigram[2] not in vocabulary:
                    filtered.remove(trigram)
            filtered[:] = [tuple(trigram) for trigram in filtered]
            for trigram in filtered:
                probability += counter[trigram]
    elif vocabulary_type == '1' or vocabulary_type == '2':
        if ngram_size == '1':
            dirty = tweet
            filtered = copy.deepcopy(dirty)
            for unigram in dirty:
                if unigram not in vocabulary:
                    filtered = filtered.replace(unigram, '')
            for unigram in filtered:
                probability += counter[unigram]
        elif ngram_size == '2':
            dirty = [tweet[i:i+2] for i in range(len(tweet)-2+1)]
            filtered = copy.deepcopy(dirty)
            for bigram in dirty:
                if bigram[0] not in vocabulary or bigram[1] not in vocabulary:
                    filtered.remove(bigram)
            filtered[:] = [tuple(bigram) for bigram in filtered]
            for bigram in filtered:
                probability += counter[bigram]
        elif ngram_size == '3':
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

# trace file
trace_file = open("trace_myModel_"+vocabulary_type+"_"+ngram_size+"_"+str(smoothing_value)+".txt", "w+", encoding="utf-8")

# get the probabilities of each language for each tweet
for tweet in testing_data:
    tweet_id = tweet[0]
    # lowercase tweet in order to flatten everything and improve results (BYOM)
    tweet[3] = tweet[3].lower()
    # calculate the probability of each tweet for each language 
    prob_es, prob_pt, prob_en, prob_eu, prob_gl, prob_ca = calculate_probability(tweet[3], prior_es, counter_es), calculate_probability(tweet[3], prior_pt, counter_pt), calculate_probability(tweet[3], prior_en, counter_en), calculate_probability(tweet[3], prior_eu, counter_eu), calculate_probability(tweet[3], prior_gl, counter_gl), calculate_probability(tweet[3], prior_ca, counter_ca)
    language_probabilities = {'es': prob_es, 'pt': prob_pt, 'en': prob_en, 'eu': prob_eu, 'gl': prob_gl, 'ca': prob_ca}
    # get the most probable language for the tweet
    # check if correct or wrong
    # print to trace file
    classified_language = max(language_probabilities, key=language_probabilities.get)
    classified_value = '%.2E' % Decimal(language_probabilities[classified_language])
    correct_classification = tweet[2]
    if correct_classification == classified_language:
        answer = 'correct'
        correct_answers += 1
    else:
        answer = 'wrong'
    update_precision_and_recall_variables(classified_language, answer, correct_classification)
    trace_file.write(tweet_id + "  " + classified_language + "  " + str(classified_value) + "  " + correct_classification + "  " + answer + "\n")

trace_file.close()

# avoid division by 0
def new_division(div1, div2):
    return div1 / div2 if div2 else 0.0

# calculate accuracy, per-class precision, per-class recall, per-class F1-measure, macro-F1 and weighted-average-F1
accuracy = correct_answers / total_test_tweets
precision_es, precision_pt, precision_en, precision_eu, precision_gl, precision_ca = new_division(correctly_classified_as_es, total_classified_as_es), new_division(correctly_classified_as_pt, total_classified_as_pt), new_division(correctly_classified_as_en, total_classified_as_en), new_division(correctly_classified_as_eu, total_classified_as_eu), new_division(correctly_classified_as_gl, total_classified_as_gl), new_division(correctly_classified_as_ca, total_classified_as_ca)
recall_es, recall_pt, recall_en, recall_eu, recall_gl, recall_ca = correctly_classified_as_es / total_es, correctly_classified_as_pt / total_pt, correctly_classified_as_en / total_en, correctly_classified_as_eu / total_eu, correctly_classified_as_gl / total_gl, correctly_classified_as_ca / total_ca

# make sure we avoid division by 0 when dividing
def calculate_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

f1_es, f1_pt, f1_en, f1_eu, f1_gl, f1_ca = calculate_f1(precision_es, recall_es), calculate_f1(precision_pt, recall_pt), calculate_f1(precision_en, recall_en), calculate_f1(precision_eu, recall_eu), calculate_f1(precision_gl, recall_gl), calculate_f1(precision_ca, recall_ca)
macro_f1 = (f1_es + f1_pt + f1_en + f1_eu + f1_gl + f1_ca) / 6
weighed_average_f1 = (total_es * f1_es + total_pt * f1_pt + total_en * f1_en + total_eu * f1_eu + total_gl * f1_gl + total_ca * f1_ca) / total_test_tweets

# evaluation file
evaluation_file = open("eval_myModel_"+vocabulary_type+"_"+ngram_size+"_"+str(smoothing_value)+".txt", "w+", encoding="utf-8")

evaluation_file.write(str("%.4f" % accuracy) + "\n" + str("%.4f" % precision_eu) + "  " + str("%.4f" % precision_ca) + "  " + str("%.4f" % precision_gl) + "  " + str("%.4f" % precision_es) + "  " + str("%.4f" % precision_en) + "  " + str("%.4f" % precision_pt) + "\n" + str("%.4f" % recall_eu) + "  " + str("%.4f" % recall_ca) + "  " + str("%.4f" % recall_gl) + "  " + str("%.4f" % recall_es) + "  " + str("%.4f" % recall_en) + "  " + str("%.4f" % recall_pt) + "\n" + str("%.4f" % f1_eu) + "  " + str("%.4f" % f1_ca) + "  " + str("%.4f" % f1_gl) + "  " + str("%.4f" % f1_es) + "  " + str("%.4f" % f1_en) + "  " + str("%.4f" % f1_pt) + "\n" + str("%.4f" % macro_f1) + "  " + str("%.4f" % weighed_average_f1) + "\n")

evaluation_file.close()
