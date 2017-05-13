from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocessing import normalize, micro_tokenize
import conf

def evaluate(cat, fold, txt_train, txt_test, y_train, y_test):
    fe = CountVectorizer(
        preprocessor=normalize,
        tokenizer=micro_tokenize,
        binary=True,
    )
    predictor = MultinomialNB()
    fe.fit(txt_train)
    X = fe.transform(txt_train)
    predictor.fit(X, y_train)
    X_test = fe.transform(txt_test)
    y_pred = predictor.predict(X_test)

    return y_pred
