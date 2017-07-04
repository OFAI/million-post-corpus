db='../experiments/data/million_post_corpus/corpus.sqlite3'
outfile=category_per_topic.md

rm -f "$outfile"

echo "# Category Prevalence per Topic Path

The following tables list for the respective category the topic paths which have the most prevalence of positively labeled instances. For example, for SentimentNegative, the first row shows a topic path with "82% of 387". This means that we have 387 posts that were submitted to articles which belong to the topic path \"Newsroom/Panorama/Flucht/Fluchtgeschichten/Serie_Aufderflucht\" and for which we have an annotator judgement with regard to SentimentNegative. 82 percent of these (i.e., 317 posts) were labeled to actually have the SentimentNegative quality, and the remaining 18 percent (70 posts) were labeled not to belong to the SentimentNegative category.

Topic paths which have less than 20 labeled posts are not listed.
" >> "$outfile"

cats=$(sqlite3 "$db" "SELECT Name FROM Categories ORDER BY Ord")
for cat in $cats
do
    echo -e "## $cat\n\n<table>" >> "$outfile"
    sqlite3 "$db" "
        SELECT CAST(ROUND(100.0 * SUM(Value) / COUNT(Value)) AS INTEGER) AS perc,
             COUNT(Value), Path
        FROM Articles
            JOIN Posts USING(ID_Article)
            JOIN Annotations_consolidated USING(ID_Post)
        WHERE Category = '$cat'
        GROUP BY Path
        HAVING COUNT(Value) >= 20
        ORDER BY perc DESC
        LIMIT 10
    " | sed -r 's#^([^|]+)[|]([^|]+)[|](.*)$#<tr><td>\1 % of \2</td><td>\3</td></tr>#' >> "$outfile"
    # sed 's#^#<tr><td>#' | sed 's#[|]# %</td><td>#' | sed 's#$#</td></tr>#' >> "$outfile"
    echo -e "</table>\n" >> "$outfile"
done
