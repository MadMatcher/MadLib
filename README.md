## MadLib: A Library of EM Functions

When performing entity matching (EM), *users often want to experiment with a variety of EM workflows and run these workflows in a variety of runtime environments.* MadLib addresses these needs. It is an open-source library of EM functions that can be combined to create a variety of EM workflows, for a variety of runtime environments. MadLib focuses on the workflows in the matching step (but will also support workflows in the blocking step in the future). 

### Example Workflows

Examples that MadLib can help create for the matching step: 
* A *featurizing workflow* that creates features then uses them to convert all tuple pairs in the candidate set (which is the output of the blocking step) to a set of feature vectors.
* A *labeling workflow* to label a set of tuple pairs as match/non-match.
* A *training-data-creation workflow* that examines a very large set of tuple pairs to select a small set of tuple pairs that are "informative", then helps the user label this set. This workflow uses a well-known machine learning technique called active learning.
* A *sampling workflow* that outputs a sample of tuple pairs from a large set of tuple pairs. The sample is likely to contain both matches and non-matches.  
* A *matching-using-passive-learning workflow* that reads two tables A and B, a candidate set C of tuple pairs (obtained from running a blocking solution for A and B), a set of labeled tuple pairs P, featurizes C and P, trains a matcher M on P, then applies M to predict each pair in C as match/non-match. This workflow is well suited for the case where the user already has a set of labeled tuple pairs P to serve as training data for the matcher. 
* A *matching-using-active-learning* workflow that reads two tables A and B, a candidate set C of tuple pairs (obtained from running a blocking solution for A and B), takes a sample S of C, performs active learning on S to label a set of tuple pairs, uses this set to train a matcher M, and applies M to predict match/non-match for each tuple pair in C. This workflow is well suited for the case where the user does not yet have any training data for the matcher. 
* And many more possible workflows.

For example, MadLib functions can be combined to create a matching-using-active-learning workflow that is equivalent to the workflow used by [ActiveMatcher](https://github.com/anhaidgroup/active_matcher). See the Python script for this workflow [here](https://github.com/MadMatcher/MadLib/blob/main/examples/spark-cluster-examples/madlib_spark_cluster.py). 

### Runtime Environments

The above workflows are created by stitching together MadLib functions and Python code in Python scripts. You can run these scripts in three runtime environments: 
* *Pandas on a single machine:* Use this if you have a relatively small amount of data, or just want to experiment with MadLib.
* *Spark on a single machine:* Use this if you have a relatively small amount of data, or just want to experiment with MadLib, or if you want to test your Spark scripts before running them on a cluster.
*  *Spark on a cluster of machines:* Use this if you have a large amount of data, such as 5M+ tuples or more in Tables A and B. 

### Solving Matching Challenges

Today the matching step of EM often uses machine learning: train a matcher M on a set of labeled tuple pairs, then apply M to new pairs to predict match/non-match. In the EM context this raises a set of challenges, as described below. MadLib functions address these challenges, and distinguish MadLib from other existing EM packages: 
* *How to create the features?* MadLib analyzes the schema and data of the two tables A and B to be matched, to create a comprehensive set of features that involve similarity functions and tokenizers.
* *How to create the training data?* If a set of labeled tuple pairs for training is not available (a very common scenario), MadLib can help the user create such a set, using active learning.
* *How to scale?* When the tables A and B to be matched are large (e.g., 5M+ tuples), scaling is difficult. MadLib provides Spark-based solutions to these problems.
* *How to label examples?* MadLib provides a variety of labelers that the user can use (to label data in the command-line interface, using a Web browser, etc.), and ways to extend or customize these labelers.

### Additional Details

* The core functions of MadLib include create_features, featurize, down_sample, create_seeds, label_data, label_pairs, train_matcher, apply_matcher.
* Features use well-known similarity measures such as TF/IDF, Jaccard, Overlap, Cosine, etc., and well-known tokenizers such as word-level tokenization and qgrams.
* Labelers are objects that allow users to label using command-line interface and Web browser. If all matches are known, you can use them to simulate labeling (using the Gold Labeler). This facilitates development, debugging, and computing matching accuracy.
* Matchers can use classification models from Scikit Learn and PySpark MLlib.
* Features, labelers, and matchers can be customized and new types can be easily added. 

### Case Studies and Performance Statistics

We have used the functions in MadLib or earlier variants in many real-world applications in domain science and industry. We will report more details here in the near future. 

### Installation

See instructions to install MadLib on a [single machine](https://github.com/MadMatcher/MadLib/blob/main/docs/installation-guides/install-single-machine.md) or a [cloud-based cluster](https://github.com/MadMatcher/MadLib/blob/main/docs/installation-guides/install-cloud-based-cluster.md). 

### How to Use

See the [MadLib technical guide](https://github.com/MadMatcher/MadLib/blob/main/docs/madlib-technical-guide.md) for an in-depth look into the format, working, and usage of MadLib functions. See this page for the [Python scripts of examples of EM workflows](https://github.com/MadMatcher/MadLib/blob/main/docs/workflow-examples.md) created with MadLib. We recommend reading the technical guide before examining the Python scripts. You may also want to take a quick look at [this brief note about Spark](https://github.com/MadMatcher/MadLib/blob/main/docs/spark-note.md).

### Further Pointers

See [API documentation](https://madmatcher.github.io/MadLib). For questions / comments, contact [our research group](mailto:entitymatchinginfo@gmail.com).
