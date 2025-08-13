## Examples of EM Workflows Created with MadLib

MadLib functions can be combined (typically in Python script) to create various EM workflows. These workflows can be Pandas workflows on a single machine, Spark workflows on a single machine, or Spark workflows on a cluster. 

The [examples directory](https://github.com/MadMatcher/MadLib/tree/main/examples) provides examples of Python scripts implementing a variety of these workflows. In what follows we briefly describe these workflows. 

### Workflows in Directory pandas-examples
All workflows in this directory run Pandas on a single machine. 
* madlib_pandas.py: This workflow implements the entire matching step for matching DBLP and ACM tables. It loads the data, creates features, featurizes the candidate set, takes a sample, creates seeds, does active learning on the sample in batch mode to label more examples, trains a matcher, and applies a matcher. Here we use the gold labeler.
* madlib_pandas_passive_learning.py: **CHECK THIS EXAMPLE**
* madlib_pandas_labeling_only.py: 
