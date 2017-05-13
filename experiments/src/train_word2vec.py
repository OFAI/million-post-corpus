import logging
import multiprocessing
import os
import sqlite3

from gensim.models import word2vec

from preprocessing import normalize, micro_tokenize
from customlogging import logger
import conf

def preprocess(row):
    if row[0] and row[1]:
        txt = row[0] + ' ' + row[1]
    elif row[0]:
        txt = row[0]
    elif row[1]:
        txt = row[1]
    else:
        txt = ''
    return micro_tokenize(normalize(txt))

def prepare_data():
    logger.debug('Fetching unlabeled posts from database')
    con = sqlite3.connect(conf.CORPUSDB)
    sql = '''
        SELECT Headline, Body
        FROM Posts
        WHERE ID_Post NOT IN (
            SELECT DISTINCT ID_Post
            FROM Annotations
        )
    '''
    r = con.execute(sql)
    pool = multiprocessing.Pool()
    posts = pool.map(preprocess, r)
    return posts

if __name__ == '__main__':
    if not os.path.exists(conf.W2V_DIR):
        os.mkdir(conf.W2V_DIR)
    sentences = prepare_data()

    logger.debug('word2vec training...')
    logging.basicConfig(format='%(asctime)s [word2vec]: %(message)s',
        level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=conf.W2V_DIMS, window=5,
        min_count=5, seed=conf.SEED, workers=1, iter=conf.W2V_EPOCHS)
    model.delete_temporary_training_data(
        replace_word_vectors_with_normalized=True)
    outfile = os.path.join(conf.W2V_DIR, 'model')
    logger.debug('Storing word2vec object to "%s"' % outfile)
    model.save(fname_or_handle=outfile, separately=None, pickle_protocol=3)
    logger.debug('Finished.')
