import time

from Conceptualizer import Conceptualizer
from LDA import LDA

dump_file = '../data/enwiki-20170720-pages-articles.xml.bz2'
bow_path = '../data/full_wiki_bow.mm'
dict_path = '../data/full_wiki.dictionary'
model_file = '../models/ldamodel_topics100_trainiter20_full_en.gensim'
num_topics = 100


def generate_input_files():
    print("Start generating input files")
    start_generating_input_files = time.time()
    lda.generate_bow_of_dump_file(dump_file, bow_path, dict_path)
    print('Generating input files of dump took', (time.time() - start_generating_input_files))


def train_lda(generate_input_files_flag=True, training_iteratios=20, max_docs=None):
    """
    Trains the lda model.
    The model could be trained based on existing bag of words and dictionary or with existing bow and dictionary.
    Training from dump file, like wikipedia dump, or documentsfolder possible
    :param generate_input_files_flag: if True the bag of words and dictionary will be generated
    """
    if generate_input_files_flag:
        generate_input_files()
    print("Start training:")
    start_training = time.time()
    lda.train_on_dump_file(num_topics, bow_path, dict_path, model_file, training_iterations, max_docs)
    print('Training LDA with', num_topics, 'topics took', (time.time() - start_training))


lda = LDA()

# Train LDA based on wiki dump file:
#train_lda(generate_input_files_flag=True)
# Train LDA based on document folder
# lda.train_on_document_folder(num_topics, document_folder, model_file, training_iterations=20)

#Update LDA based on document folder
# lda.update_on_document_folder(document_folder, model_file)

# Conceptualize with trained model:
lda.load(model_file)

conceptualizer = Conceptualizer(lda)
print(conceptualizer.conceptualize("When was Barack Obama born?", "Barack Obama"))
