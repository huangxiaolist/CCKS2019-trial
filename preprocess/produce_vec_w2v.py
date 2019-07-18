import logging
import multiprocessing
import os.path
import sys
import numpy as np


from gensim.models import word2vec


def train_w2v():
    """
    Get the word2vec's model
    :return:
    """
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    outputp1 = 'model/person.model'

    sentences = word2vec.Text8Corpus('data/text_segment.txt')
    model = word2vec.Word2Vec(sentences, size=50, window=10, min_count=2, workers=multiprocessing.cpu_count(), iter=20)
    model.init_sims(replace=True)
    model.save(outputp1)


def get_w2v():
    """
    Use the model to get vec.txt [word vec]
    :return:
    """
    model = word2vec.Word2Vec.load('model/person.model')
    with open('data/vec.txt', 'w') as f:
        for word in model.wv.vocab.keys():
            vec_string = np.array2string(model.wv[word]).replace('[ ', '').replace(']', '').replace('[', '').replace('\n', '')
            line = "{0} {1}\n".format(word, vec_string)
            f.write(line)


if __name__ == '__main__':

    # train_w2v()
    get_w2v()
