### label_pairs()

What it does: Allows a user to label pairs of data using their choice of MadLib labeling interfaces.

```python
def label_pairs(

    labeler: Labeler,                # Labeler object
    pairs: Union[pd.DataFrame, SparkDataFrame] # DataFrame with pairs of id's
) -> Union[pd.DataFrame, SparkDataFrame]
```

**Parameter Explanations:**

`labeler`: Method for generating labels

- What it is: A Labeler object that defines how to label examples
- Purpose: Provides the mechanism to convert unlabeled examples into labeled training data

_See [Built-in Labeler Classes](#built-in-labeler-classes) for available options and usage. Currently, label_pairs supports all of the built-in labeler types, except for the gold labeler._

* `pairs`: A Pandas or Spark dataframe that specifies a set of pairs of record IDs. This DataFrame must have at least two columns:
  - `column 1`: Record IDs from table A
  - `column 2`: Record IDs from table B
  - 
The columns may be named anything, but the first column must have the records IDs from table A and the second column must have the record IDs from table B.
Table A and Table B are specified as `a_df` and `b_df`, respectively, when you create your labeler object.
If there are more than two columns, only the first two columns will be considered. The rest will be ignored and are not for labeling data.
