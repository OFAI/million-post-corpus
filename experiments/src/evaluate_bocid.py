import os
import pickle
import sys

import scipy.sparse
from sklearn.svm import SVC

from preprocessing import normalize, micro_tokenize
import conf

class BoCID_feature_extractor(object):
    def __init__(self, w2cidfile):
        self.word2cid = pickle.load(open(w2cidfile, 'rb'))
        self.n_clust = max(self.word2cid.values()) + 1

    def fit(self, *args, **kwargs):
        pass

    def transform(self, txts):
        data = {}
        unk = self.word2cid['UNK']
        for i, txt in enumerate(txts):
            words = micro_tokenize(normalize(txt))
            cids = [ self.word2cid.get(w, unk) for w in words ]
            for c in cids:
                if (i, c) in data:
                    data[(i, c)] += 1
                else:
                    data[(i, c)] = 1
        keys = sorted(data.keys())
        values = [ data[k] for k in keys ]
        row_ind = [ k[0] for k in keys ]
        col_ind = [ k[1] for k in keys ]
        X = scipy.sparse.csr_matrix((values, (row_ind, col_ind)),
            shape=(len(txts), self.n_clust))
        return X

def evaluate(cat, colds, txt_train, txt_test, y_train, y_test):
    if not os.path.exists(conf.BOCID_CLUSTFILE):
        print('Word-do-cluster-ID mapping file "%s" not found.' %
            conf.BOCID_CLUSTFILE)
        print('Did you run train_word2vec.py and bocid_clustering.py?')
        sys.exit(1)

    fe = BoCID_feature_extractor(conf.BOCID_CLUSTFILE)
    predictor = SVC(
        kernel=conf.SVM_KERNEL,
        class_weight=conf.SVM_CLWEIGHT,
        C=conf.SVM_C,
        random_state=conf.SEED,
    )
    fe.fit(txt_train)
    X = fe.transform(txt_train)
    predictor.fit(X, y_train)
    X_test = fe.transform(txt_test)
    y_pred = predictor.predict(X_test)

    return y_pred
