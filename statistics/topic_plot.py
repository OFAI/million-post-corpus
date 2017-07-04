import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

db = '../experiments/data/million_post_corpus/corpus.sqlite3'

con = sqlite3.connect(db)
sql = '''
    SELECT COUNT(ID_Post) AS cnt, Path
    FROM Posts JOIN Articles USING(ID_Article)
    GROUP BY Path
    ORDER BY cnt DESC
'''
r = con.execute(sql)
rows = r.fetchall()
vals = [ row[0] for row in rows ]
# vals = vals[:1000]

f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(range(len(vals)), vals)
ax[0].set_xlabel('All 1229 topics, sorted by number of posts')
ax[0].set_ylabel('Number of posts')
ax[0].grid()

ax[1].plot(range(len(vals)), vals)
ax[1].set_yscale('log')
ax[1].set_xlabel('All 1229 topics, sorted by number of posts')
ax[1].set_ylabel('Number of posts (log scale)')
ax[1].grid()

f.tight_layout()
f.savefig('topicplot.png')
