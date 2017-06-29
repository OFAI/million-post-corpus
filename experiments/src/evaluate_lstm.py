import math
import multiprocessing
import os
import warnings

from gensim.models.word2vec import Word2Vec
import numpy
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from customlogging import logger
from preprocessing import normalize, micro_tokenize
import conf

class LSTMModel(object):
    def __init__(self, emb, num_classes):
        self.data = tf.placeholder(tf.int32,
            [conf.LSTM_BATCHSIZE, conf.LSTM_MAXPOSTLEN])
        self.target = tf.placeholder(tf.float32,
            [conf.LSTM_BATCHSIZE, num_classes])
        self.lengths = tf.placeholder(tf.int32, [conf.LSTM_BATCHSIZE])
        self.dropout_lstm = tf.placeholder(tf.float32)
        self.dropout_fully = tf.placeholder(tf.float32)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_emb = tf.Variable(emb.wv.syn0, name="W")
            self.embedded = tf.nn.embedding_lookup(self.W_emb, self.data)

        self.cell = tf.contrib.rnn.LSTMCell(conf.LSTM_HIDDEN,
            state_is_tuple=True)

        # dropout for LSTM cell
        self.cell = tf.contrib.rnn.DropoutWrapper(cell=self.cell,
            output_keep_prob=self.dropout_lstm)

        # add sequence_length to dynamic_rnn
        self.val, self.state = tf.nn.dynamic_rnn(self.cell, self.embedded,
            dtype=tf.float32, sequence_length=self.lengths)

        out_size = int(self.val.get_shape()[2])
        index = tf.range(0, conf.LSTM_BATCHSIZE)
        index = index * conf.LSTM_MAXPOSTLEN + self.lengths - 1
        flat = tf.reshape(self.val, [-1, out_size])
        self.last = tf.gather(flat, index)

        # dropout for fully-connected layer
        self.last_drop = tf.nn.dropout(self.last, self.dropout_fully)

        self.weight = tf.Variable(tf.truncated_normal(
            [conf.LSTM_HIDDEN, int(self.target.get_shape()[1])]))
        self.bias = tf.Variable(
            tf.constant(0.1, shape=[self.target.get_shape()[1]]))
        self.prediction = tf.nn.softmax(
            tf.matmul(self.last_drop, self.weight) + self.bias)
        self.cross_entropy = -tf.reduce_sum(
            self.target * tf.log(tf.clip_by_value(self.prediction, 1e-10, 1.0)))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=conf.LSTM_LEARNINGRATE)
        self.minimize = self.optimizer.minimize(self.cross_entropy)
        self.mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))
        self.init_op = tf.global_variables_initializer()

def plot_losses_f1s(losses, f1s_train,
    precisions_vali, recalls_vali, f1s_vali,
    precisions_test, recalls_test, f1s_test,
    plotfile):
    f, axes = plt.subplots(1, 4, figsize=(15,5), dpi=100)
    axes[0].plot(losses)
    axes[1].plot(f1s_train)
    axes[2].plot(precisions_vali, color='red', label='Precision')
    axes[2].plot(recalls_vali, color='green', label='Recall')
    axes[2].plot(f1s_vali, color='blue', label='F1')
    axes[3].plot(precisions_test, color='red', label='Precision')
    axes[3].plot(recalls_test, color='green', label='Recall')
    axes[3].plot(f1s_test, color='blue', label='F1')

    # indicate epoch with highest validation F1
    best_epoch = numpy.argmax(f1s_vali)
    axes[0].axvline(best_epoch, color='#ffe100')
    axes[1].axvline(best_epoch, color='#ffe100')
    axes[2].axvline(best_epoch, color='#ffe100')
    axes[3].axvline(best_epoch, color='#ffe100')

    axes[0].set_title('Loss on Training Data')
    axes[1].set_title('F1 on Training Data')
    axes[2].set_title('Evaluation on Validation Data')
    axes[3].set_title('Evaluation on Test Data')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[2].set_xlabel('Epoch')
    axes[3].set_xlabel('Epoch')

    axes[2].legend(loc='best', fontsize='small')
    axes[3].legend(loc='best', fontsize='small')

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()
    axes[3].grid()

    f.tight_layout()
    f.savefig(plotfile, dpi=100)
    plt.close()

def preprocess(txt):
    words = micro_tokenize(normalize(txt))
    # sequences of length 0 can make the training crash (tf.gather)
    if len(words) == 0:
        words = [ 'asdfasdf' ]
    return words

def stratified_batch_generator(X_orig, y_orig, lengths_orig, batchsize):
    X = numpy.copy(X_orig)
    y = numpy.copy(y_orig)
    lengths = numpy.copy(lengths_orig)

    shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    lengths = lengths[shuffle_indices]

    cl0_indices = numpy.where(y[:,0] == 1)[0]
    cl1_indices = numpy.where(y[:,1] == 1)[0]

    ratio = y[:,0].sum() / len(y)
    cl0perbatch = int(round(ratio * batchsize))
    cl1perbatch = batchsize - cl0perbatch

    cl0i = 0
    cl1i = 0
    while cl0i < len(cl0_indices) and cl1i < len(cl1_indices):
        cl0end = min(cl0i + cl0perbatch, len(cl0_indices))
        cl1end = min(cl1i + cl1perbatch, len(cl1_indices))

        if (cl0end - cl0i) + (cl1end - cl1i) < batchsize:
            cl0end = len(cl0_indices)
            cl1end = len(cl1_indices)

        batchindices = numpy.concatenate((
            cl0_indices[cl0i:cl0end],
            cl1_indices[cl1i:cl1end],
        ))
        batchindices.sort()
        batchX = X[batchindices]
        batchy = y[batchindices]
        batchlengths = lengths[batchindices]
        yield (batchX, batchy, batchlengths)

        cl0i = cl0end
        cl1i = cl1end

def evaluate(cat, fold, txt_train, txt_test, y_train, y_test):
    pool = multiprocessing.Pool()
    wordlists_train = pool.map(preprocess, txt_train)
    wordlists_test = pool.map(preprocess, txt_test)
    pool.close()
    pool.join()

    emb = Word2Vec.load(os.path.join(conf.W2V_DIR, 'model'))
    # add point at orign for unknown words
    emb.wv.syn0 = numpy.vstack((emb.wv.syn0,
        numpy.zeros(emb.wv.syn0.shape[1], dtype=numpy.float32)))

    # train data: replace words with embedding IDs, zero-padding and truncation
    X = numpy.zeros((len(y_train), conf.LSTM_MAXPOSTLEN), dtype=numpy.int32)
    X_lengths = numpy.zeros((len(y_train)))
    for i, words in enumerate(wordlists_train):
        X_lengths[i] = len(words)
        for j, w in enumerate(words):
            if j >= conf.LSTM_MAXPOSTLEN:
                break
            if w in emb:
                X[i,j] = emb.vocab[w].index
            else:
                X[i,j] = len(emb.vocab)

    # test data: replace words with embedding IDs, zero-padding and truncation
    test_X = numpy.zeros((len(y_test), conf.LSTM_MAXPOSTLEN), dtype=numpy.int32)
    test_lengths = numpy.zeros((len(y_test)))
    for i, words in enumerate(wordlists_test):
        test_lengths[i] = len(words)
        for j, w in enumerate(words):
            if j >= conf.LSTM_MAXPOSTLEN:
                break
            if w in emb:
                test_X[i,j] = emb.vocab[w].index
            else:
                test_X[i,j] = len(emb.vocab)

    # one-hot encode y
    enc = OneHotEncoder()
    y = enc.fit_transform(y_train.reshape(-1,1)).todense()
    test_y = enc.transform(y_test.reshape(-1,1)).todense()

    # split training data 80/20 into training and validation data for early
    # stopping
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
        random_state=conf.SEED)
    train_i, vali_i = next(splitter.split(X, y_train))
    X_vali = X[vali_i,:]
    y_vali = y[vali_i,:]
    vali_lengths = X_lengths[vali_i]
    X = X[train_i,:]
    y = y[train_i,:]
    X_lengths = X_lengths[train_i]

    numpy.random.seed(conf.SEED)
    tf.set_random_seed(conf.SEED)
    model = LSTMModel(emb, y.shape[1])

    # The following, in combination with
    #   export CUDA_VISIBLE_DEVICES=""
    # in the shell disables all parallelism, which leads to reproducible results
    # but takes a very long time to complete
    # sess = tf.Session(config=tf.ConfigProto(
        # inter_op_parallelism_threads=1
        # intra_op_parallelism_threads=1))

    sess = tf.Session()

    sess.run(model.init_op)
    no_of_batches = math.ceil(len(X) / conf.LSTM_BATCHSIZE)
    losses = []
    f1s_train = []
    precisions_vali = []
    recalls_vali = []
    f1s_vali = []
    precisions_test = []
    recalls_test = []
    f1s_test = []
    best_vali_f1 = -1.0
    best_y_pred = []
    for i in range(conf.LSTM_EPOCHS):
        ptr = 0
        totalloss = 0.0
        predictions = []
        true = []
        batch_gen = stratified_batch_generator(X, y, X_lengths,
            conf.LSTM_BATCHSIZE)
        for inp, out, leng in batch_gen:
            extra = conf.LSTM_BATCHSIZE - len(inp)
            if extra > 0:
                inp = numpy.vstack((inp, numpy.zeros((extra, inp.shape[1]))))
                out = numpy.vstack((out, numpy.zeros((extra, out.shape[1]))))
                leng = numpy.concatenate((leng, numpy.zeros(extra)))
            _, loss, pred = sess.run(
                [
                    model.minimize,
                    model.cross_entropy,
                    model.prediction
                ],
                {
                    model.data: inp,
                    model.target: out,
                    model.lengths: leng,
                    model.dropout_lstm: conf.LSTM_DROPOUT_LSTM,
                    model.dropout_fully: conf.LSTM_DROPOUT_FULLY,
                }
            )
            pred = list(numpy.argmax(pred, axis=1))
            true.extend(out)
            if extra > 0:
                pred = pred[:-extra]
                true = true[:-extra]
            predictions.extend(pred)
            totalloss += loss
        losses.append(totalloss)
        true = numpy.argmax(true, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            f1s_train.append(f1_score(predictions, true))


        # validation set F1
        predictions = []
        ptr2 = 0
        for j in range(math.ceil(len(X_vali) / conf.LSTM_BATCHSIZE)):
            inp2 = X_vali[ptr2:ptr2+conf.LSTM_BATCHSIZE]
            leng = vali_lengths[ptr2:ptr2+conf.LSTM_BATCHSIZE]
            extra = conf.LSTM_BATCHSIZE - len(inp2)
            if extra > 0:
                inp2 = numpy.vstack((inp2, numpy.zeros((extra, inp2.shape[1]))))
                leng = numpy.concatenate((leng, numpy.zeros(extra)))

            ptr2 += conf.LSTM_BATCHSIZE
            pred = sess.run(model.prediction,
                {
                    model.data: inp2,
                    model.lengths: leng,
                    model.dropout_lstm: 1.0,
                    model.dropout_fully: 1.0,
                }
            )
            pred = list(numpy.argmax(pred, axis=1))
            if extra > 0:
                pred = pred[:-extra]
            predictions.extend(pred)
        true = numpy.argmax(y_vali, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            precisions_vali.append(precision_score(predictions, true))
            recalls_vali.append(recall_score(predictions, true))
            f1s_vali.append(f1_score(predictions, true))


        # test set F1
        predictions = []
        ptr2 = 0
        for j in range(math.ceil(len(test_X) / conf.LSTM_BATCHSIZE)):
            inp2 = test_X[ptr2:ptr2+conf.LSTM_BATCHSIZE]
            leng = test_lengths[ptr2:ptr2+conf.LSTM_BATCHSIZE]
            extra = conf.LSTM_BATCHSIZE - len(inp2)
            if extra > 0:
                inp2 = numpy.vstack((inp2, numpy.zeros((extra, inp2.shape[1]))))
                leng = numpy.concatenate((leng, numpy.zeros(extra)))

            ptr2 += conf.LSTM_BATCHSIZE
            pred = sess.run(model.prediction,
                {
                    model.data: inp2,
                    model.lengths: leng,
                    model.dropout_lstm: 1.0,
                    model.dropout_fully: 1.0,
                }
            )
            pred = list(numpy.argmax(pred, axis=1))
            if extra > 0:
                pred = pred[:-extra]
            predictions.extend(pred)
        true = numpy.argmax(test_y, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            precisions_test.append(precision_score(predictions, true))
            recalls_test.append(recall_score(predictions, true))
            f1s_test.append(f1_score(predictions, true))

        # "early stopping" (not really stopping)
        if f1s_vali[-1] > best_vali_f1:
            best_y_pred = predictions
            best_vali_f1 = f1s_vali[-1]
            logger.debug('New best Validation F1: %f', best_vali_f1)

        logger.debug('Epoch %3d of %3d, total loss = %.4f, ' +
            'F1_train = %.4f, F1_test = %.4f',
            i + 1, conf.LSTM_EPOCHS, totalloss, f1s_train[-1], f1s_test[-1])
        if not os.path.exists(conf.LSTM_PLOTDIR):
            os.mkdir(conf.LSTM_PLOTDIR)
        plotfile = os.path.join(conf.LSTM_PLOTDIR,
            'plot_%s_%d.png' % (cat, fold))
        plot_losses_f1s(
            losses, f1s_train,
            precisions_vali, recalls_vali, f1s_vali,
            precisions_test, recalls_test, f1s_test,
            plotfile
        )

    sess.close()
    del model
    tf.reset_default_graph()

    return best_y_pred
