## Examples of EM Workflows Created with MadLib

MadLib functions can be combined (typically in Python script) to create various EM workflows. These workflows can be Pandas workflows on a single machine, Spark workflows on a single machine, or Spark workflows on a cluster. 

The [examples directory](https://github.com/MadMatcher/MadLib/tree/main/examples) provides examples of Python scripts implementing a variety of these workflows. In what follows we briefly describe these workflows. 

### Workflows in Directory pandas-examples
All workflows in this directory run Pandas on a single machine. 
* madlib_pandas.py: This workflow implements the entire matching step for matching DBLP and ACM tables. It loads the data, creates features, featurizes the candidate set, takes a sample, creates seeds, does active learning on the sample in batch mode to label more examples, trains a matcher, and applies a matcher. Here we use the gold labeler.
* madlib_pandas_passive_learning.py: **CHECK THIS EXAMPLE**
* madlib_pandas_labeling_only.py: This workflow loads the data, then asks the user to label the tuple pairs in the candidate set, using the CLI labeler (or the Web labeler). Note that it is straightforward to modify this script to take a sample from the candidate set using down_sample(), then asks the user to label only the pairs in the sample. In general, this script shows how you can use MadLib functions, especially labeled_pairs() to label a set of tuple pairs.

### Workflows in Directory spark-local-examples
All workflows in this directory run Spark on a single machine. The three Python scripts in this directory mirror the three Python scripts in Directory pandas-example. 

### Workflows in Directory spark-cluster-examples
All workflows in this directory run Spark on a cluster of machines. The three Python scripts in this directory mirror the three Python scripts in Directory pandas-example. 

