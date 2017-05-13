CORPUSDB = 'data/million_post_corpus/corpus.sqlite3'
RESULTDB = 'data/results.sqlite3'
SEED = 123456
RESULTDB_SETUP = '''
    CREATE TABLE Results (
        ts TIMESTAMP,
        Category TEXT NOT NULL,
        Method TEXT NOT NULL,
        Fold INTEGER NOT NULL,
        TrueLabel INTEGER NOT NULL,
        PredictedLabel INTEGER NOT NULL
    )
'''
PLOTDIR = 'plots'
TABLEDIR = 'tables'

# word2vec
W2V_DIR = 'models/word2vec'
W2V_DIMS = 300
W2V_EPOCHS = 10 # CHANGE ME ! to: 10 I guess (was 100)

# D2V
D2V_DIR = 'models/doc2vec'
D2V_ALPHA = 0.025
D2V_MINALPHA = 0.001
D2V_EPOCHS = 10 # CHANGE ME ! to: 10 I guess (was 1)
D2V_DIMS = 300

# SVM
SVM_KERNEL = 'linear'
SVM_CLWEIGHT = 'balanced'
SVM_C = 0.375

# BOCID
BOCID_CLUSTFILE = 'models/word2cid.pkl'
BOCID_NCLUSTERS = 1000
BOCID_CLUSTITER = 500

# LSTM
LSTM_MAXPOSTLEN = 200
LSTM_BATCHSIZE = 2000
LSTM_HIDDEN = 128
LSTM_EPOCHS = 30 # CHANGE ME ! to: 30
LSTM_LEARNINGRATE = 1e-3
LSTM_DROPOUT_LSTM = 0.5
LSTM_DROPOUT_FULLY = 0.5
LSTM_PLOTDIR = PLOTDIR + '/LSTM_plots'
