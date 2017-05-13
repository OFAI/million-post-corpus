from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from preprocessing import normalize, micro_tokenize
import conf

def evaluate(cat, fold, txt_train, txt_test, y_train, y_test):
    fe = CountVectorizer(
        preprocessor=normalize,
        tokenizer=micro_tokenize,
        binary=True,
    )
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
