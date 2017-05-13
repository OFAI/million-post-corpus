import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import scipy.sparse

from preprocessing import normalize, micro_tokenize
import conf

class NBSVM_predictor(object):
    def __init__(self, kernel, class_weight, C):
        self.svm = SVC(
            kernel=kernel,
            class_weight=class_weight,
            C=C,
            random_state=conf.SEED,
        )
        self.r = None

    def fit(self, X, y):
        # NBSVM as described in "Baselines and bigrams: simple, good
        # sentiment and topic classification"
        # http://dl.acm.org/citation.cfm?id=2390665.2390688

        # compute log-count ratio r
        alpha = 1.0
        p = alpha + X[y == 1].sum(axis=0)
        q = alpha + X[y != 1].sum(axis=0)
        self.r = numpy.log((p / p.sum()) / (q / q.sum()))

        # X is of type scipy.sparse.csr.csr_matrix
        # and of size N x vocabsize
        # r is of type numpy.matrixlib.defmatrix.matrix
        # and of size 1 x vocabsize
        #
        # We want to multiply each row of X by r element-wise.
        # Broadcasting does not work for sparse matrices, but we can
        # instead make r a diagonal matrix and use regular matrix
        # multiplication to achieve the same result.
        self.r = scipy.sparse.diags(numpy.array(self.r)[0], 0)
        X *= self.r
        self.svm.fit(X, y)

    def predict(self, X):
        # see above
        X *= self.r
        return self.svm.predict(X)

def evaluate(cat, fold, txt_train, txt_test, y_train, y_test):
    fe = CountVectorizer(
        preprocessor=normalize,
        tokenizer=micro_tokenize,
        binary=True,
    )
    predictor = NBSVM_predictor(
        kernel=conf.SVM_KERNEL,
        class_weight=conf.SVM_CLWEIGHT,
        C=conf.SVM_C,
    )
    fe.fit(txt_train)
    X = fe.transform(txt_train)
    predictor.fit(X, y_train)
    X_test = fe.transform(txt_test)
    y_pred = predictor.predict(X_test)

    return y_pred
