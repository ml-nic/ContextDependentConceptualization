from LDA import tokenize
from gensim.utils import simple_preprocess
from utilities import get_concepts_of_instance_by_probase
from utilities import split_text_in_words


from utilities import get_concepts_of_instance_by_probase
from utilities import split_text_in_words
import collections
import numpy as np

class Conceptualizer():
    def __init__(self, lda):
        self.lda = lda
        self.ldamodel = lda.ldamodel

    def conceptualize(self, sentence, instance, eval=False):
        """
        Conceptualize the given instance in the given context (sentence)
        :param sentence: a sentence as context
        :param instance: the instance, which should be conceptualized in the given context
        :return: the most likely concept for the intance in the given context
        """
        concepts = get_concepts_of_instance_by_probase(instance, eval)
        if len(concepts) == 0:  # TODO
            return None

        #try:
            # check context
            #words = split_text_in_words(sentence.lower())
            #instance_words = split_text_in_words(instance.lower())
            #i = words.index(instance_words[0])
            #largerProbaseConcepts = get_concepts_of_instance_by_probase(
            #    " ".join(words[max(i - 1, 0):i + len(instance_words)]), eval)
            #if len(largerProbaseConcepts) > 0:
            #    pass
            #    #return None
            #largerProbaseConcepts = get_concepts_of_instance_by_probase(" ".join(words[i:i + len(instance_words) +
            #    # 1]), eval)
            #if len(largerProbaseConcepts) > 0:
            #    pass
                #return None
        #except Exception as e:
        #    print('Error getting larger concepts for {} in {}: {}'.format(instance.encode('utf-8'),
        #                                                                  sentence.encode('utf-8'), e))

        probabilities_of_concepts = self.__calculate_probs_of_concepts(concepts, sentence)
        #print(probabilities_of_concepts)
        if probabilities_of_concepts is None or len(probabilities_of_concepts) == 0:
            return None
        if eval:
            return probabilities_of_concepts
        most_likely_concept = max(probabilities_of_concepts, key=lambda item: item[1])[0]
        return most_likely_concept

    def __calculate_probs_of_concepts(self, concepts, sentence):
        """
        Calculates for each concept the probability of the concept for the given sentence
        :param concepts: the concepts and their probability
        :param sentence: the given sentence
        :return: the concepts and ther probabilities
        """
        probabilities_of_concepts = []
        for concept in concepts:
            if concept not in self.ldamodel.id2word.token2id.keys():
                continue
            prob_c_given_w = concepts[concept]

            #topics_of_concept = self.ldamodel.get_term_topics(concept, minimum_probability=0.0) # phi
            #probs_of_topics_for_given_concept = [0] * self.ldamodel.num_topics
            #summm = 0
            #for topic_id, prob_of_topic in topics_of_concept:
            #    probs_of_topics_for_given_concept[topic_id] = prob_of_topic
            #    summm += prob_of_topic
            #print(summm)
            topic_terms_ = self.ldamodel.state.get_lambda()
            topics_terms_proba_ = np.apply_along_axis(lambda x: x/x.sum(), 1, topic_terms_)
            probs_of_topics_for_given_concept = topics_terms_proba_[:,self.ldamodel.id2word.token2id[concept]]

            #for topic_id in range(100):
            #    print(np.sum(topics_terms_proba_[topic_id,:]))

            bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(sentence))
            # topic_distribution_for_given_bow
            topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
            sum = 0
            for topic_id, prob_of_topic in topics_of_text:
                sum += probs_of_topics_for_given_concept[topic_id] * prob_of_topic
            prob_c_given_w_z = prob_c_given_w * sum

            probabilities_of_concepts.append((concept, prob_c_given_w_z))
        return probabilities_of_concepts
