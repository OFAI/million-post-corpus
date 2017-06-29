import os
import pickle
import sys

from gensim.models.word2vec import Word2Vec
import numpy
from sklearn.cluster import KMeans

from customlogging import logger

import conf

if __name__ == '__main__':
    w2vmodelfile = os.path.join(conf.W2V_DIR, 'model')
    if not os.path.exists(w2vmodelfile):
        print('Word2vec model file "%s" not found.' % w2vmodelfile)
        print('Did you run train_word2vec.py?')
        sys.exit(1)

    logger.debug('Loading word embedding')
    emb = Word2Vec.load(w2vmodelfile)
    vocab = emb.index2word
    wordvecs = emb.wv.syn0

    # add UNK word at origin of embedding space
    vocab.append('UNK')
    wordvecs = numpy.vstack((wordvecs, numpy.zeros(wordvecs.shape[1])))

    clusterer = KMeans(n_clusters=conf.BOCID_NCLUSTERS, random_state=conf.SEED,
        max_iter=conf.BOCID_CLUSTITER, n_jobs=-1)
    logger.debug('Starting clustering')
    VC = clusterer.fit_predict(wordvecs)
    logger.debug('Matching words to cluster IDs')
    word2cid = { vocab[k]: VC[k] for k in range(len(vocab)) }
    pickle.dump(word2cid, open(conf.BOCID_CLUSTFILE, 'wb'))
    logger.debug('Wrote word-to-cluster-ID mapping to "%s"',
        conf.BOCID_CLUSTFILE)
