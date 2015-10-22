# coding=utf-8
__updated__ = '2015-10-20'

from optparse import OptionParser
import os
import sys
from math import log10
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
    classifier = NaiveBayes(train_data, test_data, train_labels, test_labels, vocab)
    #classifier.predict(classifier.multinomial_dist())

    print "label", train_labels[2]

    correct = 0
    incorrect = 0
    for example in train_data:
        res = classifier.predict(classifier.bernoulli_dist, train_data[example])
        if res == train_labels[example]:
            correct += 1
        else:
            incorrect += 1
    print "correct", "incorrect"
    print correct, incorrect


class NaiveBayes:
    """
    For probability datastructures use the names:
    p(x|y) = p_x_y
    """

    def __init__(self, train_data, test_data, train_label, test_label, vocab):
        self.model = {}
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
        self.p_label = {}
        self.p_word_class = {}

        self.vocab = vocab
        self.prob_label()
        self.probability_of_word()


        """
        Compute p_x
        """
    def predict(self, distribution, x):
        """
        It may be possible to just use p(y = k) * (product) p(x_i | y = k)

        log(p(y=k)) = log(p(y=k)) + sum(log(p(x_i) | y = k)
        we have p(x_i|y=k) estimated earlier, this is our training.
        Allegedly we should be using log probabilities here which would make it a sum of logs.


        :param distribution: bernoulli or multinomial
        :return: probability of class k
        """
        maximum = (-1, -1)
        for i in range(1, 21):
            val = distribution(x, i)
            if val > maximum[0]:
                maximum = (val, i)
        return maximum[1]

    def bernoulli_dist(self, x, k):
        """
        See predict docstring for now
        log(p(y=k)) = log(p(y=k)) + sum(log(p(x_i) | y = k)

        I assume y = k is a constant that we can estimate from the data.

        word_id is a string right now.
        """
        p_y_k = self.p_label[k]
        total = 0
        #sparse representation
        for word in x:
            if k in self.p_word_class[str(word)]:
                total += log10(self.p_word_class[str(word)][k])

        log_prob_k_x = log10(p_y_k) + total #sum([log10(self.p_word_class[str(word)][k]) for word in x])
        return log_prob_k_x

    def probability_of_word(self):
        """
        :return:
        """


        """

        p_word_class =
        num docs word i appeared in that had label y
        _______
        num docs that had label y
        """

        doc_label = {}
        word_doc_label = {}
        print len(self.vocab.keys())
        print >> sys.stderr, "calculating all denomenators."
        for doc in self.train_data:
            label = self.train_label[doc]
            if label not in doc_label:
                doc_label[label] = 0
            doc_label[label] += 1

            for key in self.vocab:
                word = self.vocab[key]
                word_doc_label[word] = {}
                if self.train_label[doc] == label and word in self.train_data[doc]:
                    if label not in word_doc_label[word][doc]:
                        word_doc_label[word][label] = 0
                    word_doc_label[word][label] += 1

        print >> sys.stderr, "calculating all numerators."
        for word in self.vocab.values():
            if word not in self.p_word_class:
                self.p_word_class[word] = {}
            for label in range(1, 21):
                if label not in self.p_word_class[word]:
                    self.p_word_class[word][label] = 0.0
                self.p_word_class[word][label] = word_doc_label[word][label] / float(doc_label[label])

    def prob_label(self):
        total = 0
        for doc in self.train_label:
            label = self.train_label[doc]
            if label not in self.p_label:
                self.p_label[label] = 0
            self.p_label[label] += 1
            total += 1

        for l in self.p_label:
            self.p_label[l] /= float(total)

    def multinomial_dist(self):
        print "goodbye"
        pass


    def train(self, distribution):
        """
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