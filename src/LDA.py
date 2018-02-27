import itertools
import os

import gensim
from gensim import corpora
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import smart_open, simple_preprocess
from stop_words import get_stop_words


def iter_over_dump_file(dump_file, min_length_of_article=50, ignore_namespaces=None):
    """
    Iterator over wiki_dump_file.
    Returns title and tokens for next article in dump file.
    Ignores short articles.
    Ignores meta articles, throug given namespaces.
    Default namespaces are 'Wikipedia', 'Category', 'File', 'Portal', 'Template', 'MediaWiki', 'User', 'Help', 'Book', 'Draft'
    :param dump_file: the dump file
    :param min_length_of_article: the min number of words in the next article. Default = 50
    :param ignore_namespaces: list of namespaces which should be ignored.
    :return: title, tokens
    """
    if ignore_namespaces is None:
        ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < min_length_of_article or any(
                title.startswith(namespace + ':') for namespace in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens


def tokenize(text):
    """
    Preprocess and then tokenize a given text
    :param text: the text which should be tokenized.
    :return: the token of the given text, after preprocess the text
    """
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]


class LDA():
    def __init__(self):
        self.stop_words = get_stop_words('en')

    def load(self, model_file):
        """
        Loads a LDA model from a given file
        :param model_file: the file which contains the model, which should be loaded
        """
        self.ldamodel = gensim.models.wrappers.LdaMallet.load(model_file)
        from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
        self.ldamodel = malletmodel2ldamodel(self.ldamodel)
        print(self.ldamodel.__dict__)

    def train_on_document_folder(self, num_topics, document_folder, model_outputfile, training_iterations=20):
        """
        Trains a new lda model, based on a folder with different document.
        Each document in a different file.
        :param num_topics: the number of topics, which should be generated
        :param document_folder: the folder, which contains the documents
        :param model_outputfile: the file in which the trained model should be saved
        :param training_iterations: the number of LDA training iterations
        """
        corpus, dictionary = self.__create_lda_corpus_based_on_document_folder(document_folder)

        self.ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary,
                                                                passes=training_iterations, minimum_probability=0)
        self.ldamodel.save(model_outputfile)

    def generate_bow_of_dump_file(self, dump_file, bow_output_file, dict_output_file):
        doc_stream = (tokens for _, tokens in iter_over_dump_file(dump_file))
        id2word_dict = gensim.corpora.Dictionary(doc_stream)
        print(id2word_dict)
        id2word_dict.filter_extremes(no_below=20, no_above=0.1, keep_n=250000)
        print(id2word_dict)
        dump_corpus = DumpCorpus(dump_file, id2word_dict)
        print("save bow...")
        gensim.corpora.MmCorpus.serialize(bow_output_file, dump_corpus)
        print("save dict")
        id2word_dict.save(dict_output_file)

    def train_on_dump_file(self, num_topics, bow_path, dict_path, model_outputfile, training_iterations=20,
                           max_docs=None):
        """
        Trains a new LDA model based on a wikipedia dump or any other dump in the same format.
        The dump could be zipped.
        :param num_topics: the number of topics, which should be generated
        :param bow_path: the path inclusive filename, where the bag of words should be saved
        :param dict_path: the path incl. filename, where the dictionary should be saved
        :param model_outputfile: the file in which the trained model should be stored
        :param training_iterations: the number of LDA training iterations
        :param max_docs: the number of how many docs should be used for training, if None all docs are used
        """
        print("load bow...")
        mm_corpus = gensim.corpora.MmCorpus(bow_path)
        print("load dict...")
        id2word_dict = gensim.corpora.Dictionary.load(dict_path)
        clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, max_docs)
        print("start trainig")
        self.ldamodel = gensim.models.ldamulticore.LdaMulticore(clipped_corpus, num_topics=num_topics,
                                                                id2word=id2word_dict, passes=training_iterations,
                                                                minimum_probability=0)
        print("save model")
        self.ldamodel.save(model_outputfile)

    def __create_lda_corpus_based_on_document_folder(self, document_folder):
        """
        Creates a corpus and the corresponding dictionary, for a given document folder
        :param document_folder: the folder, which contains the documents
        :return: the corpus and the dictionary
        """
        files = os.listdir(document_folder)
        doc_set = []
        for file in files:
            with open(document_folder + '/' + file, "r", encoding="utf8") as f:
                for line in f:
                    l = line.strip()
                    if len(l) > 0:
                        doc_set.append(l)

        print("Finished reading {} documents".format(len(doc_set)))

        # list for tokenized documents in loop
        texts = self.preprocess_documents(doc_set)

        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)

        # convert tokenized documents into a document-term matrix
        return ([dictionary.doc2bow(text) for text in texts], dictionary)

    def preprocess_documents_original(self, docs):
        """
        Tokenize and remove stop words of given documents
        :param docs: collection which contains the documents
        :return: the preprocessed texts
        """
        texts = []
        for doc in docs:
            raw = doc.lower()
            tokens = tokenize(raw)

            stopped_tokens = [i for i in tokens if not i in self.stop_words]
            texts.append(stopped_tokens)
        return texts

    def preprocess_documents(self, documents):
        """
        Preprocess given documents.

        removes meta-articles,
        ignores short documents, to avoid unwanted word 2 word connections,
        removes stop words
        :param documents: collection of to be processed documents
        """
        namespaces = ['Wikipedia', 'Category', 'File', 'Portal', 'Template', 'MediaWiki', 'User', 'Help', 'Book',
                      'Draft']
        namespaces = [namespace.lower() for namespace in namespaces]

        # remove_short_documents
        # remove_articles_where_title_specific_namespaces
        texts = []
        for document in documents:
            if len(document) >= 200:
                if document.title not in namespaces:
                    tokens = tokenize(document.lower())
                    # Stoplist cleaning:
                    cleaned_tokens = [token for token in tokens if token not in self.stop_words]
                    texts.append(tokens)
        # Remove most frequent and less frequent words
        # for word in all documents:
        #    if word appears in more than 10 % of the articles:
        #        remove(word) from whole corpora
        #    if word apperas in less than 20 articles:
        #        remove(word) from corpora

        # Additional possible steps
        # filter by length
        # lemmatization
        # stemming
        # parts of speech

        # then keep top n words # recommended 50.000 - 100.000
        return texts

    def update_on_document_folder(self, document_folder, model_output_file):
        """
        Online learning.
        Updates the current LDA model, trained on the given documents
        :param document_folder: the folder, which contains the new documents
        :param model_output_file: the outputfile, where the updated model should be stored
        """
        corpus, dictionary = self.__create_lda_corpus_based_on_document_folder(document_folder)
        print('start updating')
        self.ldamodel.update(corpus)
        print('finished updating. Now save model')
        self.ldamodel.save(model_output_file)


class DumpCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        """
        Iterator over wiki corpus
        :return: bag-of-words format = list of `(token_id, token_count)` 2-tuples
        """
        self.titles = []
        for title, tokens in itertools.islice(iter_over_dump_file(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs
