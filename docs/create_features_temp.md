### create_features()
```python
def create_features(
    A: Union[pd.DataFrame, SparkDataFrame],         # Your first dataset
    B: Union[pd.DataFrame, SparkDataFrame],         # Your second dataset
    a_cols: List[str],                              # Columns from A to compare
    b_cols: List[str],                              # Columns from B to compare
    sim_functions: Optional[List[Callable]] = None, # List of similarity functions
    tokenizers: Optional[List[Tokenizer]] = None,   # List of tokenizers
    null_threshold: float = 0.5                     # Max null percentage allowed
) -> List[Callable]
```

This function uses heuristics that analyzes the columns of Tables A and B to create a set of features. The features uses a combination of similarity functions and tokenizers. See [here](./sim-functions-tokenizers.md) for a brief discussion of similarity functions and tokenizers for MadLib.

* Parameters A and B are Pandas or Spark dataframes (depending on whether your runtime environment is Pandas on a single machine, Spark on a single machine, or Spark on a cluster of machines). They are the two tables to match. Both dataframes must contain an `_id` column.
* a_cols and b_cols are lists of column names from A and B. We use these columns to create features. As of now, a_cols and b_cols must be the same list. But in the future we will modify the function create_features to handle the case where these two lists can be different.
* sim_functions is a list of sim_functions (this will override the default sim functions used by MadLib, see below).
* tokenizers is the list of extra tokenizers (this will override the default tokenizers used by MadLib, see below).
* null_threshold is a number in [0,1] specifying the maximal fraction of missing values allowed in a column. MadLib automatically excludes columns with too much missing data from feature generation (see below).

The above function returns a list of features. Each feature is an executable object (as indicated by the `Callable` keyword). 

**Example:** Suppose A has columns `['_id', 'name', 'address', 'phone']` and B has columns `['_id', 'name', 'address', 'phone', 'comment']`. Then both a_cols and b_cols can be `['name', 'address', 'phone']`, meaning we only want to create features involving these columns. Suppose null_threshold = 0.6, and suppose that column 'phone' has more than 70% of its values missing. Then we drop 'phone' and will create features involving just 'name' and 'address'. 

**Creating Features for the Default Case:** If the parameter sim_function has value None, then we use a set of default similarity functions. As of now this set is the following five functions:
 - `TFIDFFeature`: Term frequency-inverse document frequency similarity
 - `JaccardFeature`: Set-based similarity using intersection over union
 - `SIFFeature`: Smooth inverse frequency similarity
 - `OverlapCoeffFeature`: Set overlap coefficient
 - `CosineFeature`: Vector space cosine similarity

If the parameter tokenizers has value None, then we use a set of default tokenizers. As of now this set is the following three functions: 
- `StrippedWhiteSpaceTokenizer()`: Splits on whitespace and normalizes
- `NumericTokenizer()`: Extracts numeric values from text
- `QGramTokenizer(3)`: Creates 3-character sequences (3-grams)

In this case, we create the features as follows: 

* First, we detect and drop all columns with too many missing values (above null_threshold). Then we analyze the remaining columns to detect their types (e.g., numeric vs text). We also compute the average token count for each tokenizer-column combination.
* Then we create the following features:
    + *Exact Match Features*: Created for all columns.
     ```python
     # Every column gets an exact match feature
     ExactMatchFeature(column_name, column_name)
     ```
   + *Numeric Features*: Created for columns that are detected as numeric.
     ```python
     # Numeric columns get relative difference features
     RelDiffFeature(column_name, column_name)
     ```
   + *Token-based Features*: Created based on tokenizer analysis
   ```python
   # For each tokenizer-column combination with avg_count >= 3:
   # Creates the five features using the default similarity functions (TF-IDF, Jaccard, SIF, Overlap, Cosine)
   TFIDFFeature(column_name, column_name, tokenizer=tokenizer)
   JaccardFeature(column_name, column_name, tokenizer=tokenizer)
   SIFFeature(column_name, column_name, tokenizer=tokenizer)
   OverlapCoeffFeature(column_name, column_name, tokenizer=tokenizer)
   CosineFeature(column_name, column_name, tokenizer=tokenizer)
   ```

**Creating Features for Other Cases:** You can add more similarity functions using the parameter sim_functions and more tokenizers using the parameter tokenizers.

In particular, we provide the following extra tokenizers that you can use: 

   - `AlphaNumericTokenizer()`: Extracts alphanumeric sequences
   - `QGramTokenizer(5)`: Creates 5-character sequences (5-grams)
   - `StrippedQGramTokenizer(3)`: Creates 3-grams with whitespace stripped
   - `StrippedQGramTokenizer(5)`: Creates 5-grams with whitespace stripped

In the case where the user specifies similarity functions and/or tokenizers, we create the features as follows: 
* We still create the exact match features and the numeric features, as described in the default case.
* We will create features for each similarity function and tokenizer combination. Each feature will be applied to columns where the average number of tokens produced by the particular tokenizer is at least 3. 
* Finally, we create the following special features if you have included the AlphaNumeric tokenizer. 
   ```python
   # Only for AlphaNumericTokenizer with avg_count <= 10:
   MongeElkanFeature(column_name, column_name, tokenizer=tokenizer)
   EditDistanceFeature(column_name, column_name)
   SmithWatermanFeature(column_name, column_name)
   ```
**Example:** The following is a simple example of creating features:

   ```python
   features = create_features(
       customers_df,
       prospects_df,
       ['name', 'address', 'revenue'],
       ['name', 'address', 'revenue']
   )

   # What happens behind the scenes:
   # 1. All columns get ExactMatchFeature
   # 2. If 'revenue' is numeric, it gets RelDiffFeature
   # 3. For each tokenizer (Defaults: StrippedWhiteSpace, Numeric, QGram):
   #    - If average token count >= 3 over the column, creates features for all 5 default similarity functions


   # Typical output might include:
   # - ExactMatchFeature for name, address, revenue
   # - RelDiffFeature for revenue (if numeric)
   # - TFIDFFeature, JaccardFeature, SIFFeature, OverlapCoeffFeature, CosineFeature
   #   for each tokenizer-column combination with sufficient tokens
   ```

If you want to use some set of our default similarity functions, default tokenizers, or extra tokenizers, we provide the following helper methods:
```
get_base_sim_functions() # returns the default similarity function classes used in MadLib: TFIDF, Jaccard, SIF, OverlapCoeffecient, Cosine
```
```
get_base_tokenizers() # returns the default tokenizer classes used in MadLib: StrippedWhiteSpace, Numeric, 3gram
```
```
get_extra_tokenizers() # returns the extra tokenizer classes that are implemented, but not used in MadLib: AlphaNumeric, 5gram, Stripped3gram, Stripped5gram
```

If you wanted to use all of the default and extra tokenizers, your script would include the following:
```python3
from MadLib import get_base_tokenizers, get_extra_tokenizers

default_tokenizers = get_base_tokenizers()
extra_tokenizers = get_extra_tokenizers()

all_tokenizers = default_tokenizers + extra_tokenizers

features = create_features(
   customers_df,
   prospects_df,
   ['name', 'address', 'revenue'],
   ['name', 'address', 'revenue'],
   tokenizers=all_tokenizers
)

```
