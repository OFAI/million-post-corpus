import os
import sys

from gensim.models.doc2vec import Doc2Vec
import numpy
from sklearn.svm import SVC

from preprocessing import normalize, micro_tokenize
import conf

class D2V_feature_extractor(object):
    def __init__(self, d2vfile):
        self.d2v = Doc2Vec.load(d2vfile)

    def fit(self, *args, **kwargs):
        pass

    def transform(self, txts):
        res = []
        for txt in txts:
            # see https://github.com/RaRe-Technologies/gensim/issues/447
            self.d2v.random.seed(conf.SEED)
            v = self.d2v.infer_vector(micro_tokenize(normalize(txt)))
            res.append(v)
        return numpy.vstack(res)

def evaluate(cat, folds, txt_train, txt_test, y_train, y_test):
    d2vmodelfile = os.path.join(conf.D2V_DIR, 'model')
    if not os.path.exists(d2vmodelfile):
        print('Doc2vec model file "%s" not found' % d2vmodelfile)
        print('Did you run train_doc2vec.py?')
        sys.exit(1)

    fe = D2V_feature_extractor(d2vmodelfile)
    predictor = SVC(
        kernel=conf.SVM_KERNEL,
        class_weight=conf.SVM_CLWEIGHT,
        C=conf.SVM_C,
        random_state=conf.SEED,
    )
    X = fe.transform(txt_train)
    predictor.fit(X, y_train)
    X_test = fe.transform(txt_test)
    y_pred = predictor.predict(X_test)

    return y_pred
