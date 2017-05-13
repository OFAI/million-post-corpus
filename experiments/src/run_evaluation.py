from collections import OrderedDict
import datetime
import itertools
import multiprocessing
import os
import sqlite3

import numpy

from customlogging import logger
import conf
import evaluate_bow
import evaluate_mnb
import evaluate_nbsvm
import evaluate_bocid
import evaluate_d2v
import evaluate_lstm

con = None
methodmodules = OrderedDict([
    # [ 'BOW', evaluate_bow ],
    # [ 'MNB', evaluate_mnb ],
    # [ 'NBSVM', evaluate_nbsvm ],
    # [ 'BOCID', evaluate_bocid ],
    # [ 'D2V', evaluate_d2v ],
    [ 'LSTM', evaluate_lstm ],
])

def get_categories():
    con = sqlite3.connect(conf.CORPUSDB)
    r = con.execute('SELECT DISTINCT Category FROM Annotations_consolidated')
    cats = [ row[0] for row in r ]

    # remove SentimentNeutral
    cats = [ c for c in cats if c != 'SentimentNeutral' ]

    cats.sort()
    return cats

def get_folds():
    con = sqlite3.connect(conf.CORPUSDB)
    r = con.execute('SELECT DISTINCT Fold FROM CrossValSplit')
    folds = [ row[0] for row in r ]
    folds.sort()
    return folds

def get_data(cat, fold, verbose=False):
    sql = '''
        SELECT Headline, Body, Value, Fold
        FROM Annotations_consolidated a
            JOIN Posts p USING(ID_Post)
            JOIN CrossValSplit c USING(ID_Post, Category)
        WHERE a.Category = ?
        ORDER BY p.ID_Post ASC
    '''
    con = sqlite3.connect(conf.CORPUSDB)
    r = con.execute(sql, (cat,))
    txt_train = []
    txt_test = []
    y_train = []
    y_test = []
    for row in r:
        if row[0] and row[1]:
            txt = row[0] + ' ' + row[1]
        elif row[0]:
            txt = row[0]
        elif row[1]:
            txt = row[1]
        else:
            txt = ''
        if row[3] == fold:
            txt_test.append(txt)
            y_test.append(row[2])
        else:
            txt_train.append(txt)
            y_train.append(row[2])
    txt_train = numpy.array(txt_train)
    txt_test = numpy.array(txt_test)
    y_train = numpy.array(y_train)
    y_test = numpy.array(y_test)

    numpy.random.seed(conf.SEED)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y_train)))
    txt_train = txt_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y_test)))
    txt_test = txt_test[shuffle_indices]
    y_test = y_test[shuffle_indices]

    return (txt_train, txt_test, y_train, y_test)

def evaluate(method, cat, fold):
    resultrows = []
    txt_train, txt_test, y_train, y_test = get_data(cat, fold)

    y_pred = methodmodules[method].evaluate(
        cat, fold, txt_train, txt_test, y_train, y_test
    )
    now = datetime.datetime.now()
    for i in range(len(y_test)):
        resultrows.append(
            [ now, cat, method, fold, int(y_test[i]), int(y_pred[i]) ]
        )
    return resultrows

if __name__ == '__main__':
    if not os.path.exists(conf.RESULTDB):
        con_results = sqlite3.connect(conf.RESULTDB)
        con_results.execute(conf.RESULTDB_SETUP)
    else:
        con_results = sqlite3.connect(conf.RESULTDB)

    cats = get_categories()
    folds = get_folds()
    sql = 'INSERT INTO Results VALUES(?, ?, ?, ?, ?, ?)'
    for method in methodmodules.keys():
        logger.debug('-' * 40)
        logger.debug('Method %s', method)
        logger.debug('Computing results for %d categories and %d folds...' %
            (len(cats), len(folds)))

        jobs = []
        for c in cats:
            for fold in folds:
                jobs.append([method, c, fold])

        # LSTM runs on GPU, where all memory is needed for a single job. Hence,
        # we need to run each job sequentially.
        if method == 'LSTM':
            results = list(itertools.starmap(evaluate, jobs))
        # For all other methods, we can spawn parallel processes.
        else:
            pool = multiprocessing.Pool()
            results = pool.starmap(evaluate, jobs)
            pool.close()
            pool.join()
        logger.debug('%d jobs completed.' % len(jobs))

        logger.debug('Storing results to database "%s"...' % conf.RESULTDB)
        n = 0
        for r in results:
            con_results.executemany(sql, r)
            n += len(r)
        con_results.commit()
        logger.debug('%d rows written.' % n)
