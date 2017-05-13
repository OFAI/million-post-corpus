import os
import sqlite3
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score, recall_score, f1_score

import conf

name_map = {
    'ArgumentsUsed': 'Argum',
    'Discriminating': 'Discrim',
    'Inappropriate': 'Inappr',
    'OffTopic': 'OffTopic',
    'PersonalStories': 'Personal',
    'PossiblyFeedback': 'Feedb',
    'SentimentNegative': 'Negative',
    'SentimentPositive': 'Positive',
}

if __name__ == '__main__':
    con = sqlite3.connect(conf.RESULTDB)
    con.row_factory = sqlite3.Row
    sql = 'SELECT * FROM Results'
    r = con.execute(sql)
    results = {}
    methods_seen = set()
    categories_seen = set()
    for row in r:
        m = row['Method']
        c = row['Category']
        methods_seen.add(m)
        categories_seen.add(c)
        if (m, c) in results:
            results[(m, c)][0].append(row['TrueLabel'])
            results[(m, c)][1].append(row['PredictedLabel'])
        else:
            results[(m, c)] = [ [ row['TrueLabel'] ], [ row['PredictedLabel'] ] ]

    con2 = sqlite3.connect(conf.CORPUSDB)
    r = con2.execute('SELECT Name FROM Categories ORDER BY Ord ASC')
    categories = [ row[0] for row in r ]
    methods = [ 'BOW', 'MNB', 'NBSVM', 'BOCID', 'D2V', 'LSTM' ]

    # keep only categories and methods that appear in the result data
    categories = [ c for c in categories if c in categories_seen ]
    methods = [ m for m in methods if m in methods_seen ]

    precisions = {}
    recalls = {}
    f1s = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)

        for c in categories:
            for m in methods:
                y_true = results[(m, c)][0]
                y_pred = results[(m, c)][1]
                precisions[(m, c)] = precision_score(y_true, y_pred)
                recalls[(m, c)] = recall_score(y_true, y_pred)
                f1s[(m, c)] = f1_score(y_true, y_pred)

    print('\\begin{tabular}{@{}l@{\hspace{2mm}}l@{\hspace{2mm}}' +
        '@{\\hspace{3mm}}'.join('l' * len(methods)) +
        '@{}}')
    print('\\toprule')
    print('Category & Meas. & ' + ' & '.join(methods) + '\\\\')
    print('\\midrule')
    for c in categories:
        print('\\multirow{3}{*}{' + name_map[c] + '} & Prec. ', end='')
        values = [ precisions[(m, c)] for m in methods ]
        for v in values:
            if v == max(values):
                print(' & \\textbf{%.4f}' % v, end='')
            else:
                print(' & %.4f' % v, end='')
        print('\\\\')
        print(' & Rec. ', end='')
        values = [ recalls[(m, c)] for m in methods ]
        for v in values:
            if v == max(values):
                print(' & \\textbf{%.4f}' % v, end='')
            else:
                print(' & %.4f' % v, end='')
        print('\\\\')
        print(' & $F_1$. ', end='')
        values = [ f1s[(m, c)] for m in methods ]
        for v in values:
            if v == max(values):
                print(' & \\textbf{%.4f}' % v, end='')
            else:
                print(' & %.4f' % v, end='')
        if c == categories[-1]:
            print('\\\\')
        else:
            print('\\\\[0.5em]')
    print('\\bottomrule')
    print('\\end{tabular}')


    width = 1 / (len(methods) + 1)
    f, axes = plt.subplots(3, 1, figsize=(10,8))
    for i, m in enumerate(methods):
        xs = numpy.arange(len(categories)) + i * width
        vals = [ precisions[(m, c)] for c in categories ]
        axes[0].bar(xs, vals, width=width, label=m)
        vals = [ recalls[(m, c)] for c in categories ]
        axes[1].bar(xs, vals, width=width, label=m)
        vals = [ f1s[(m, c)] for c in categories ]
        axes[2].bar(xs, vals, width=width, label=m)

    axes[0].set_xticks(numpy.arange(len(categories)) + 2.5 * width)
    axes[1].set_xticks(numpy.arange(len(categories)) + 2.5 * width)
    axes[2].set_xticks(numpy.arange(len(categories)) + 2.5 * width)
    axes[0].set_xticklabels([ name_map[c] for c in categories ])
    axes[1].set_xticklabels([ name_map[c] for c in categories ])
    axes[2].set_xticklabels([ name_map[c] for c in categories ])
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_ylabel('Precision')
    axes[1].set_ylabel('Recall')
    axes[2].set_ylabel('F1-score')
    axes[0].set_xlim((-2 * width, xs[-1] + 10 * width))
    axes[1].set_xlim((-2 * width, xs[-1] + 10 * width))
    axes[2].set_xlim((-2 * width, xs[-1] + 10 * width))

    f.tight_layout()
    f.savefig(os.path.join(conf.PLOTDIR, 'results.png'))
