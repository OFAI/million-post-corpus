# One Million Posts Corpus

### A Data Set of German Online Discussions

This is the home of the "One Million Posts" corpus, an annotated data set consisting of user comments posted to a German-language newspaper website.

This data set will be presented as a short paper at the [40th International ACM SIGIR Conference on Research and Development in Information Retrieval](http://sigir.org/sigir2017/) (SIGIR 2017):

> Dietmar Schabus, Marcin Skowron, Martin Trapp  
**One Million Posts: A Data Set of German Online Discussions**  
Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)  
Tokyo, Japan, August 2016

Please cite this paper if you use the data set (BibTex below). You can download a [preprint version of the paper here]().

```
@InProceedings{Schabus2017,
  Author    = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
  Title     = {One Million Posts: A Data Set of German Online Discussions},
  Booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  Year      = {2017},
  Address   = {Tokyo, Japan},
  Month     = aug
}
```

## Data Set Description

[Der Standard](http://derstandard.at) is an Austrian daily broadsheet newspaper which had a circulation of more than 390,000 in the year 2015 (according to [http://www.media-analyse.at/table/2612](http://www.media-analyse.at/table/2612) , 2017-04-13). On the newspaper's website, there is a discussion section below each news article where readers engage in online discussions. Since this feature was introduced in 1999, a total of 54 million user posts have accumulated, and in 2015 alone, 7.6 million posts were authored by more than 52,000 distinct users.

The data set contains a selection of user posts from the 12 month time span from 2015-06-01 to 2016-05-31. There are 11,773 labeled and 1,000,000 unlabeled posts in the data set. Detailed descriptions of the selection and annotation procedures are given in the paper.

### Annotated Categories

Potentially undesirable content:

* **Sentiment** (negative/neutral/positive)  
An important goal is to detect changes in the prevalent sentiment in a discussion, e.g., the location within the fora and the point in time where a turn from positive/neutral sentiment to negative sentiment takes place.
* **Off-Topic** (yes/no)  
Posts which digress too far from the topic of the corresponding article.
* **Inappropriate** (yes/no)  
Swearwords, suggestive and obscene language, insults, threats etc.
* **Discriminating** (yes/no)  
Racist, sexist, misogynistic, homophobic, antisemitic and other misanthropic content.

Neutral content that requires a reaction:

* **Feedback** (yes/no)  
Sometimes users ask questions or give feedback to the author of the article or the newspaper in general, which may require a reply/reaction.

Potentially desirable content:

* **Personal Stories** (yes/no)  
In certain fora, users are encouraged to share their personal stories, experiences, anecdotes etc. regarding the respective topic.
* **Arguments Used** (yes/no)  
It is desirable for users to back their statements with rational argumentation, reasoning and sources.

Several of these categories are based on respective rules in the [community guidelines of the website](http://derstandard.at/2934632/Forenregeln-Community-Richtlinien).

| Category | Does Apply | Does Not Apply | Total | Percentage |
| --- | ---: | ---: | ---: | ---: |
| SentimentNegative | 1691 | 1908 | 3599 | 47 % |
| SentimentNeutral | 1865 | 1734 | 3599 | 52 % |
| SentimentPositive | 43 | 3556 | 3599 | 1 % |
| OffTopic | 580 | 3019 | 3599 | 16 % |
| Inappropriate | 303 | 3296 | 3599 | 8 % |
| Discriminating | 282 | 3317 | 3599 | 8 % |
| PossiblyFeedback | 1250 | 4639 | 5889 | 21 % |
| PersonalStories | 1625 | 7711 | 9336 | 17 % |
| ArgumentsUsed | 1022 | 2577 | 3599 | 28 % |

## License

## Experiments

## Acknowledgments

This research was partially funded by the [Google Digital News Initiative](https://www.digitalnewsinitiative.com). We thank Der Standard and their moderators for the interesting collaboration and the annotation of the presented corpus. The GPU used for this research was donated by [NVIDIA](https://developer.nvidia.com/academic_gpu_seeding).
