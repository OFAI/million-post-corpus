# Experiments

Please see the [corpus website](https://ofai.github.io/million-post-corpus) for an introduction to this data set.

## Requirements

The version numbers in parantheses indicate the version we used, other versions may or may not work.

We recommend a [miniconda](https://conda.io/miniconda.html) environment.

* [python](https://www.python.org/) (3.5.2)
* [numpy](http://www.numpy.org/) (1.12.0)
* [scipy](https://www.scipy.org/) (0.18.1)
* [gensim](https://radimrehurek.com/gensim/) (0.13.4.1)
* [scikit-learn](http://scikit-learn.org/) (0.18.1)
* [tensorflow](https://www.tensorflow.org/) (1.0.0)
* [matplotlib](http://matplotlib.org/) (2.0.0)

To obtain reproducible results, parallel execution is disabled at several points in the code. This means things could run quite a bit faster, but would not result in the exact same results, which is now the case, with the exception of the LSTM part.

Total duration: about 4.5 hours on a machine with the following specs:

* Intel Core i7-6900K, 8x 3.20GHz
* 64 GB RAM (4x16GB DDR4-2133)
* 1TB SSD
* NVIDIA Titan X (Pascal)

## Running the Experiments

To run everything, simply execute

```
./run.sh
```

The script will ask you if you want to download the corpus (requiring `wget` or `curl` and `bzip2`).

If you are interested only in certain parts, uncomment what you don't need in `run.sh` and in `src/run_evaluation.py` (in particular, you can comment out entries from `methodmodules` if you want to run only certain methods).

The code produces some output in the directories `logs`, `plots` and `tables`. We have included our results here, so you can see what to expect.

The code also creates a directory `models`, which will be about 500 MB in size.