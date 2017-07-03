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

def make_table(categories, methods, precisions, recalls, f1s, kind, filename):
    if kind not in [ 'html', 'latex']:
        print('Unknown value for "kind" operator, skipping table creation.')
        return

    if kind == 'latex':
        rowbegin = ''
        betweencells = ' & '
        rowend = '\\\\\n'
        inexistent_cell = ' & '
        bold = lambda x: '\\textbf{%s}' % x
    elif kind == 'html':
        rowbegin = '<tr><td>'
        betweencells = '</td><td>'
        rowend = '</td></tr>\n'
        inexistent_cell = ''
        bold = lambda x: '<b>%s</b>' % x

    wins = {
        'precision': {},
        'recall': {},
        'f1': {},
    }

    src = ''
    if kind == 'latex':
        src += ('\\begin{tabular}{@{}l@{\hspace{2mm}}l@{\hspace{2mm}}' +
            '@{\\hspace{3mm}}'.join('c' * len(methods)) +
            '@{}}\n')
        src += '\\toprule\n'
    elif kind == 'html':
        src += '<table>\n'
    heads = [ 'Category', 'Meas.' ] + methods
    src += rowbegin + betweencells.join(heads) + rowend
    if kind == 'latex':
        src += '\\midrule\n'
    for c in categories:
        if kind == 'latex':
            src += '\\multirow{3}{*}{' + name_map[c] + '}'
        elif kind == 'html':
            src += '<tr><td rowspan="3">' + name_map[c]
        src += betweencells + 'Prec.'
        values = [ precisions[(m, c)] for m in methods ]
        for i, v in enumerate(values):
            val = '%.4f' % v
            if v == max(values):
                val = bold(val)
                if methods[i] in wins['precision']:
                    wins['precision'][methods[i]] += 1
                else:
                    wins['precision'][methods[i]] = 1
            src += betweencells + val
        src += rowend

        src += rowbegin + inexistent_cell + 'Rec. '
        values = [ recalls[(m, c)] for m in methods ]
        for i, v in enumerate(values):
            val = '%.4f' % v
            if v == max(values):
                val = bold(val)
                if methods[i] in wins['recall']:
                    wins['recall'][methods[i]] += 1
                else:
                    wins['recall'][methods[i]] = 1
            src += betweencells + val
        src += rowend

        src += rowbegin + inexistent_cell
        if kind == 'latex':
            src += '$F_1$.'
        elif kind == 'html':
            src += 'F1'
        values = [ f1s[(m, c)] for m in methods ]
        for i, v in enumerate(values):
            val = '%.4f' % v
            if v == max(values):
                val = bold(val)
                if methods[i] in wins['f1']:
                    wins['f1'][methods[i]] += 1
                else:
                    wins['f1'][methods[i]] = 1
            src += betweencells + val
        if kind == 'latex' and c != categories[-1]:
            src += '\\\\[0.5em]\n'
        else:
            src += rowend

    if kind == 'latex':
        src += '\\midrule\n\\multirow{3}{*}{Wins}'
    elif kind == 'html':
        src += '<tr><td rowspan="3">Wins'
    src += betweencells + 'Prec.'
    for m in methods:
        src += betweencells + str(wins['precision'].get(m, 0))
    src += rowend + rowbegin + inexistent_cell + 'Rec.'
    for m in methods:
        src += betweencells + str(wins['recall'].get(m, 0))
    src += rowend + rowbegin + inexistent_cell
    if kind == 'latex':
        src += '$F_1$'
    elif kind == 'html':
        src += 'F1'
    for m in methods:
        src += betweencells + str(wins['f1'].get(m, 0))
    src += rowend

    if kind == 'latex':
        src += '\\bottomrule\n'
        src += '\\end{tabular}\n'
    elif kind == 'html':
        src += '</table>\n'
    with open(filename, 'w') as f:
        f.write(src)
    print('Result %s table written to %s' % (kind, filename))

def make_plot(categories, methods, precisions, recalls, f1s, filename):
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
    f.savefig(filename)
    print('Result plot written to %s' % filename)

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

    htmlfile = os.path.join(conf.TABLEDIR, 'results.html')
    make_table(categories, methods, precisions, recalls, f1s, 'html', htmlfile)
    latexfile = os.path.join(conf.TABLEDIR, 'results.tex')
    make_table(categories, methods, precisions, recalls, f1s, 'latex', latexfile)
    plotfile = os.path.join(conf.PLOTDIR, 'results.png')
    make_plot(categories, methods, precisions, recalls, f1s, plotfile)
