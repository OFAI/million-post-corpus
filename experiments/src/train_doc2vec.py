import logging
import multiprocessing
import os
import sqlite3

from gensim.models.doc2vec import TaggedDocument, Doc2Vec

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

def get_post_documents():
    logger.debug('Fetching unlabeled posts from database')
    con = sqlite3.connect(conf.CORPUSDB)
    sql = '''
        SELECT ID_Post, COALESCE(Headline, '') || ' ' || COALESCE(Body, '')
        FROM Posts
        WHERE ID_Post NOT IN (
            SELECT DISTINCT ID_Post
            FROM Annotations
        )
    '''
    r = con.execute(sql)
    pool = multiprocessing.Pool()
    while True:
        rows = r.fetchmany(100000)
        if len(rows) == 0:
            break
        logger.debug('Normalizing and tokenizing')
        wordlists = pool.map(micro_tokenize,
            pool.map(normalize, [ r[1] for r in rows ]))
        for i, words in enumerate(wordlists):
            yield TaggedDocument(words, [ rows[i][0] ])
    pool.close()
    pool.join()
    logger.debug('End of generator')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [doc2vec] : %(message)s',
        level=logging.INFO)

    d2v = Doc2Vec(dm=1, size=conf.D2V_DIMS, negative=5, iter=1,
        alpha=conf.D2V_ALPHA, seed=conf.SEED, workers=1)

    logger.debug('Building doc2vec vocabulary...')
    d2v.build_vocab(get_post_documents())

    logger.debug('doc2vec training...')
    alpha = conf.D2V_ALPHA
    alpha_delta = (conf.D2V_ALPHA - conf.D2V_MINALPHA) / conf.D2V_EPOCHS
    for i in range(conf.D2V_EPOCHS):
        logger.debug('Epoch %d of %d (alpha = %f)', i+1, conf.D2V_EPOCHS, alpha)
        d2v.alpha = alpha
        d2v.train(get_post_documents(), report_delay=10.0)
        alpha -= alpha_delta

    if not os.path.exists(conf.D2V_DIR):
        os.mkdir(conf.D2V_DIR)
    outfile = os.path.join(conf.D2V_DIR, 'model')
    logger.debug('Storing doc2vec object to "%s"' % outfile)
    del d2v.docvecs.doctag_syn0
    del d2v.docvecs.doctag_syn0_lockf
    d2v.save(outfile, pickle_protocol=3)
    logger.debug('Finished.')
