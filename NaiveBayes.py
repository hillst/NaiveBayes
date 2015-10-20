# coding=utf-8
__updated__ = '2015-10-20'

from optparse import OptionParser
import os
import sys
"""
The purpose of this program is to build a fast lookup for fastq file. Every 128 bytes represents one read with the following format:
read_id,position[PADDING]\n
"""


def main():
    program_name = os.path.basename(sys.argv[0])
    program_version = "v0.1"
    program_build_date = "%s" % __updated__

    program_version_string = '%%prog %s (%s)' % (program_version, program_build_date)
    program_license = "No license."
    program_longdesc = "The purpose of this program is to build a fast lookup for fastq file. Every 128 bytes represents\
                        one read with the following format:\n read_id,position[PADDING]\\n"

    argv = sys.argv[1:]
        # setup option parser
    parser = OptionParser(version=program_version_string, epilog=program_longdesc, description=program_license)
    """
    vocab, news_groups, train, test, train_labels, test_labels
    """
    parser.add_option("-v", "--vocab", dest="vocab", help="vocab file", metavar="FILE")
    parser.add_option("-n", "--news-groups", dest="news_groups", metavar="FILE")
    parser.add_option("-T", "--train-data", dest="train_data", help="Training set", metavar="FILE")
    parser.add_option("-t", "--test-data", dest="test_data", help="Testing data", metavar="FILE")
    parser.add_option("-L", "--train-label", dest="train_label", help="Training labels", metavar="FILE")
    parser.add_option("-l", "--test-label", dest="test_label", help="Testing labels", metavar="FILE")

    # process options
    (opts, args) = parser.parse_args(argv)
    for label in (opts.vocab, opts.news_groups, opts.train_label, opts.test_data, opts.test_label, opts.train_data):
        if label is None:
            parser.print_help()
            sys.exit(-1)

    vocab, news_groups, train_data, test_data, train_labels, test_labels = read_data(opts.vocab, opts.news_groups, opts.train_data, opts.train_label, opts.test_data, opts.test_label)
    classifier = NaiveBayes(train_data, test_data, train_labels, test_labels)
    classifier.predict(classifier.multinomial_dist())
    classifier.predict(classifier.bernoulli_dist())


class NaiveBayes:

    def __init__(self, train_data, test_data, train_label, test_label):
        self.model = {}
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
        self.probability_of_word()

        """
        Compute p_x
        """
    def predict(self, distribution):
        pass

    def bernoulli_dist(self):
        """
        x_i = number of times word i appears in document X

        So we need to estimate p(word_i | y )
        p(x|y) =
        for all words in our vocab
            p(Word_i | y)^X_i * (1 - p(word_i |y)^(1 - X_i))
        print "hello"
                """

        pass


    def probability_of_word(self):
        """
        Calculates P(x_i | y) = n_i/ n
        That is, the number of times word i appears in documents with the label y / number of docs the word appears in
        :return:
        """
        word_class_counts = {}
        for doc in self.train_data:
            print doc
            for word in self.train_data[doc]:
                label = self.train_label[doc]
                if word not in word_class_counts:
                    word_class_counts[word] = {}
                if label not in word_class_counts[word]:
                    word_class_counts[word][label] = 0
                word_class_counts[word][label] += 1

        word_class_probs = {}
        for word in word_class_counts:
            word_class_probs[word] = {}
            total = sum([word_class_counts[word][label] for label in word_class_counts[word]])
            for label in word_class_counts[word]:
                word_class_probs[word][label] = word_class_counts[word][label] / float(total)
                print word, label, word_class_counts[word][label] / float(total)
        self.word_class_probs = word_class_probs


    def multinomial_dist(self):
        print "goodbye"
        pass

    def train(self, distribution):
        """
        x = document
        y = class label
        return p(x|y)

        product for all words in vocab:


        In this assignment you will implement the Naive Bayes classifier for document classification with both the
        Bernoulli model and the Multinomial model. For Bernoulli model, a document is described by a set of binary
        variables, and each variable corresponds to a word in the vocabulary V and represents its presence/absence.
        The probability of observing a document x given its class label y is then defined as:
        p(x|y) = Y
        |V |
        i=1
        p
        xi
        i|y
        (1 − pi|y)
        (1−xi)
        where pi|y denotes the probability that the word i will be present for a document of class y. If xi = 1,
        the contribution of this word to the product will be pi|y, otherwise it will be 1 − pi|y.
            p(x|y) =
        """
        pass


def read_data(vocabulary, newsgroups, train_data, train_label, test_data, test_label):
    """
    • vocabulary.txt is a list of the words that may appear in documents. The line number is word’s id in
    other files. That is, the first word (‘archive’) has wordId 1, the second (‘name’) has wordId 2, etc.
    1
    • newsgrouplabels.txt is a list of newsgroups from which a document may have come. Again, the line
    number corresponds to the label’s id, which is used in the .label files. The first line (class ‘alt.atheism’)
    has id 1, etc.
    • train.label contains one line for each training document specifying its label. The document’s id (docId)
    is the line number.
    • test.label specifies the labels for the testing documents.
    - train.data word count per word perdoc
                docId, wordId, count
    • train.data describes the counts for each of the words used in each of the documents. It uses a sparse
    format that contains a collection of tuples “docId wordId count”. The first number in the tuple species
    the document ID, and the second number is the word ID, and the final number species the number of
    times the word with id wordId in the training document with id docId. For example “5 3 10” species
    that the 3rd word in the vocabulary appeared 10 times in document 5.
    • test.data is exactly the same as train.data, but contains the counts for the test documents.
    Note that you don’t have to use the vocabulary.txt file in your learning and testing. It is only provided
    to help possibly interpret the models. For example, if you find that removing the first feature can help
    the performance,
    """
    line_index = 1
    vocab = {}
    news_groups = {}
    train = {}
    test = {}
        #doc_id => {word_id => count}
    train_labels = {}
    test_labels = {}

    for line in open(vocabulary, 'r'):
        vocab[line.strip()] = line_index
        line_index += 1

    line_index = 1
    for line in open(newsgroups, 'r'):
        news_groups[line.strip()] = line_index

    for line in open(train_data, 'r'):
        doc_id, word_id, count = line.strip().split()
        doc_id = int(doc_id)
        if doc_id not in train:
            train[doc_id] = {}
        train[doc_id][word_id] = int(count)

    for line in open(test_data, 'r'):
        doc_id, word_id, count = line.strip().split()
        doc_id = int(doc_id)

        if doc_id not in test:
            test[doc_id] = {}
        test[doc_id][word_id] = int(count)

    line_index = 1
    for line in open(train_label, 'r'):
        train_labels[line_index] = int(line.strip())
        line_index += 1

    line_index = 1
    for line in open(test_label, 'r'):
        test_labels[line_index] = int(line.strip())
        line_index += 1

    return vocab, news_groups, train, test, train_labels, test_labels





if __name__ == "__main__":
    main()