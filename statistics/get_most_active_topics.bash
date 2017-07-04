#!/bin/bash

db=../experiments/data/million_post_corpus/corpus.sqlite3
N=100

sqlite3 "$db" "
SELECT COUNT(ID_Post) AS cnt, path
FROM Articles JOIN Posts USING(ID_Article)
GROUP BY path
ORDER BY cnt DESC
LIMIT $N
" > most_active_topics.txt

i=1
rm -f most_active_topics.md
cat most_active_topics.txt | while read s
do
    echo -n "<tr><td>$i</td><td>" >> most_active_topics.md
    echo "$s" | sed 's#[|]#</td><td>#' | sed 's#$#</td></tr>#' >> most_active_topics.md
    i=$(expr $i + 1)
done
