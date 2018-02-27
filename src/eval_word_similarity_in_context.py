import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from Conceptualizer import Conceptualizer
from LDA import LDA

"""
This is an implementation of the Experiment 2: Context-Dependent Word Similarity
of the paper Context-Dependent Conceptualization.
"""


def evaluate(word1, word2, sentence1, sentence2):
    """
    Evaluates the cosine_similarity between the similarity of the given word-sentence pairs and the true score.
    :param word1: the first word in context of sentence1, which should be compared to word2 in sentence2
    :param word2: the second word in context of sentence2, which should be compared to word1 in sentence1
    :param sentence1: the first sentence as the context of word1
    :param sentence2: the second sentence as the context of word2
    :return: cosine similarity. Calculated with sklearn cosine similarity and same result is returned
    """
    concept_distribution1 = conceptualizer.conceptualize(sentence1, word1, eval=True)
    concept_distribution2 = conceptualizer.conceptualize(sentence2, word2, eval=True)
    if concept_distribution1 is None or concept_distribution2 is None:
        return None

    vector1 = np.zeros(np.max([len(concept_distribution1), len(concept_distribution2)]))
    vector2 = np.zeros(np.max([len(concept_distribution1), len(concept_distribution2)]))
    if len(concept_distribution1) == np.max([len(concept_distribution1), len(concept_distribution2)]):
        counter = 0
        for concept1 in concept_distribution1:
            vector1[counter] = concept1[1]
            for concept2 in concept_distribution2:
                if concept1[0] == concept2[0]:
                    vector2[counter] = concept2[1]
                    break
            counter += 1
    elif len(concept_distribution2) == np.max([len(concept_distribution1), len(concept_distribution2)]):
        counter = 0
        for concept2 in concept_distribution2:
            vector2[counter] = concept2[1]
            for concept1 in concept_distribution1:
                if concept2[0] == concept1[0]:
                    vector1[counter] = concept1[1]
                    break
            counter += 1
    try:
        estimated_similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
    except ValueError as e:
        raise e

    return estimated_similarity


def get_eval_set(path_of_eval_set):
    """
    Reads the evaluation set of the given txt file.
    :param path_of_eval_set: the path of the evaluation data file
    :return: the evaluation set
    """
    eval_set = []
    with(open(path_of_eval_set, 'r')) as file:
        for line in file.readlines():
            splitted_line = line.split('\t')
            id = splitted_line[0]
            word1 = splitted_line[1]
            pos_word1 = splitted_line[2]
            word2 = splitted_line[3]
            pos_word2 = splitted_line[4]
            word1_in_context = splitted_line[5]
            word2_in_context = splitted_line[6]
            average_human_rating = splitted_line[7]
            ten_individual_human_ratings = [rating.replace('\n', '') for rating in splitted_line[7:len(splitted_line)]]
            word1_in_context = word1_in_context.replace('<b> ', '').replace(' </b>', '')
            word2_in_context = word2_in_context.replace('<b> ', '').replace(' </b>', '')
            eval_set.append((id, word1, word2, word1_in_context, word2_in_context, average_human_rating))
    return eval_set


if __main__ == '__main__':
    """
    Runs the evaluation for each LDA model in model_names.
    Based on the EVALUATION_DATASET.
    """

    start_time = time.time()
    MODEL_BASE_DIR = 'F:/Not_Uploaded/conceptualization_eval/models/'
    EVALUATION_DATASET = '../ratings.txt'
    MODEL_FILE_EXTENSION = '.gensim'
    model_names = [
        'ldamodel_simple_mallet_20_10_keep_300000_gensimstop_topics100',
        'ldamodel_topics100_trainiter20_en_noStopWords',
        'ldamodel_topics100_trainiter20_full_en',
        'ldamodel_topics100_trainiter20_train_en',
    ]
    model_stat = []
    eval_set = get_eval_set(EVALUATION_DATASET)
    for model_name in model_names:
        print("\nTest", model_name, ':')
        lda = LDA()
        lda.load(MODEL_BASE_DIR + model_name + MODEL_FILE_EXTENSION)

        conceptualizer = Conceptualizer(lda)

        none_counter = 0
        estimated_similarities = []
        real_similarities = []

        for element in eval_set:
            estimated_similarity = evaluate(element[1], element[2], element[3], element[4])
            if estimated_similarity is None:
                none_counter += 1
                continue
            estimated_similarities.append(float(estimated_similarity[0][0]))
            real_similarities.append(float(element[5]))
            if (int(element[0]) - 1) % 100 == 0 and int(element[0]) > 10:
                print('now at entry with id', element[0], 'tooks', (time.time() - start_time), 'seconds')
                estimated_similarities_np = np.array(estimated_similarities)
                real_similarities_np = np.array(real_similarities)
                pearson_score = pearsonr(estimated_similarities_np, real_similarities_np)
                print(pearson_score)
                np.save('real_similarites' + model_name, real_similarities_np)
                np.save('estimated_similarities' + model_name, estimated_similarities_np)

        estimated_similarities_np = np.array(estimated_similarities)
        real_similarities_np = np.array(real_similarities)
        pearson_score = pearsonr(estimated_similarities_np, real_similarities_np)
        print("pearson_score", pearson_score)
        np.save('real_similarites' + model_name, real_similarities_np)
        np.save('estimated_similarities' + model_name, estimated_similarities_np)
        model_stat.append((model_name, pearson_score, none_counter))
        print('finished took', (time.time() - start_time), 'seconds for this model')
    print(model_stat)
    print('finished all models, took', (time.time() - start_time), 'seconds')
