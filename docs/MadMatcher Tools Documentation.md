<!-- MadMatcher Documentation -->

# MadLib Technical Documentation

This document provides an in-depth technical overview of MadLib' functionality, internal workings, and advanced use cases. For quick start guide and basic usage, refer to the README.md file [insert link here]. For complete API reference, see the API documentation. [insert link here]

## Understanding Entity Matching

### What is Entity Matching?

Entity matching (also known as record linkage or data matching) is the process of identifying and linking records that refer to the same real-world entity across different datasets. For example:

- Matching customer records across different databases
- Identifying duplicate product listings
- Linking patient records across healthcare systems

Consider these examples:

```
Dataset A:
"John Smith, 123 Main St, New York"
"J. Smith, 123 Main Street, NY"

Dataset B:
"Smith, John, 123 Main Street, New York, NY"
"John Smith Jr., 123 Main St., New York"
```

These records might refer to the same person, but they're written differently. Entity matching helps determine which records represent the same entity.

### Why is Entity Matching Challenging?

1. **Data Inconsistency**

   - Different formats ("John Smith" vs "Smith, John")
   - Typos and errors ("Main Street" vs "Main St." vs "Main Streat")
   - Missing information
   - Different conventions across systems

2. **Scale**
   - Comparing every record with every other record is computationally expensive
   - Large datasets make manual matching impractical
   - Need for automated, accurate solutions

## Core Concepts

### Record Matching Pipeline

The matching process follows several steps, each building on the previous:

1. **Feature Generation**  
   What it is: Converting raw record pairs into comparable characteristics  
   Example: Converting "John Smith" and "Smith, John" into features that capture their similarity
2. **Model Training**  
   What it is: Teaching a model to recognize what features indicate matching records  
   How it works: Using known matches and non-matches to learn patterns

3. **Prediction**  
   What it is: Using learned patterns to determine if two previously unseen records represent the same entity

4. **Active Learning**  
   What it is: Creating labeled data with human (or simulated human) labeling  
   How it works: The model picks the examples that are most ambiguous and has the user label them

### Understanding Features

A feature is a way to compare two values and determine how similar they are. Think of it as answering questions like:

- How similar are these two names?
- How different are these two ages?

Features are created through two main components:

1. **Tokenizers**  
   What they are: Tools that break text into pieces  
   Example:

   ```python
   Input: "John Smith"
   Tokenizer output: ["john", "smith"]

   Input: "123 Main Street"
   Tokenizer output: ["123", "main", "street"]
   ```

   Breaking text into tokens helps handle:

   - Different word orders
   - Extra/missing words
   - Punctuation
   - Typos
   - Variations in formatting
   - etc.

2. **Similarity Functions**  
   What they are: Methods to compute how similar two sets of tokens are  
   Available functions in MadMatcher:
   - TF-IDF: Term frequency-inverse document frequency similarity
   - Jaccard: Set-based similarity using intersection over union
   - SIF: Smooth inverse frequency similarity
   - Overlap Coefficient: Set overlap measure
   - Cosine: Vector space similarity between token vectors

### Understanding Your Similiarity Functions

When using MadMatcher, it's crucial to understand how your chosen similarity functions work:

**Why This Matters:**

- Different similarity functions interpret "high" and "low" scores differently
- Some functions return higher scores for more similar items
- Some functions return higher scores for less similar items
- Some functions have different score ranges (0-1, 0-100, etc.)

**Common Patterns:**

- **Set-based functions** (Jaccard, Overlap): Higher scores = more similar
- **Distance functions** (Edit Distance): Lower scores = more similar
- **Vector functions** (Cosine, TF-IDF): Higher scores = more similar
- **Custom functions**: You need to test and understand them yourself

**Best Practice:** Always test your similarity functions with examples to see how the score relates to the probability of a match

## Core Functions In-Depth

### create_features()

This function analyzes your data and creates combinations of tokenizers and similarity functions. Let's break down how it works:

```python
def create_features(
    A: pd.DataFrame,                               # Your first dataset
    B: pd.DataFrame,                               # Your second dataset
    a_cols: List[str],                             # Columns from A to compare
    b_cols: List[str],                             # Columns from B to compare
    sim_functions: Optional[List[Callable]] = None, # Custom similarity functions
    tokenizers: Optional[List[Tokenizer]] = None,  # Custom tokenizers
    null_threshold: float = 0.5                    # Max null percentage allowed
) -> List[Callable]
```

**Parameter Explanations:**

`A`: Your first dataset (pandas DataFrame)

- What it is: Any pandas DataFrame containing records you want to match against dataset B
- Schema requirement: Must have an `_id` column with unique identifiers for each record
- Example: A customer database with columns like `['_id', 'name', 'address', 'phone']`
- Purpose: These are your "candidate" records - potential matches for records in B

`B`: Your second dataset (pandas DataFrame)

- What it is: The pandas DataFrame containing records you want to to find matches for
- Schema requirement: Must have an `_id` column with unique identifiers for each record
- Example: A prospect database with columns like `['_id', 'name', 'address', 'phone']`
- Purpose: These are your "reference" records - the ones you want to find duplicates or matches for

`a_cols`: Column names from dataset A to use for comparison

- What it is: List of strings specifying which columns from A contain the data you want to compare
- Purpose: Tells the system which fields in A contain meaningful information for matching
- Example: `['name', 'address', 'phone']` - these are the columns that will help identify if two records represent the same entity
- Requirement: These columns must be the same as the columns for b_cols

`b_cols`: Column names from dataset B to use for comparison

- What it is: List of strings specifying which columns from B contain the data you want to compare
- Purpose: Tells the system which fields in B contain meaningful information for matching
- Example: `['name', 'address', 'phone']` - these are the columns that will help identify if two records represent the same entity
- Requirement: These columns must be the same as the columns for a_cols

`sim_functions`: Custom similarity functions (optional)

- What it is: List of function objects that define how to compute similarity between values
- Default behavior: If None, uses TF-IDF, Jaccard, SIF, Overlap Coefficient, and Cosine similarity functions
- Purpose: Allows you to customize how the system determines if two values are similar
- When to customize: When you have domain-specific knowledge about what makes records similar

`tokenizers`: Custom text processing functions (optional)

- What it is: List of tokenizer objects that define how to break text into comparable pieces
- Default behavior: If None, uses Stripped Whitespace, Numeric, and 3-gram tokenizers
- Purpose: Allows you to customize how text is preprocessed before similarity comparison
- When to customize: When your data has special formatting or you need specialized text processing

`null_threshold`: Maximum allowed missing data fraction

- What it is: A decimal number between 0.0 and 1.0 representing the maximum fraction of missing values allowed in a column
- Purpose: Automatically excludes columns with too much missing data from feature generation
- Example: 0.5 means if more than 50% of values in a column are missing, that column won't be used for feature generation
- Why it is important: Columns with mostly missing data provide little useful information for matching

1. **Column Analysis**
   What it does: Examines your data to understand its characteristics

   a) Data Type Detection

   - Identifies numeric columns (integer, float) vs text columns
   - All data is eventually converted to strings for processing
   - Columns with too many null values (above null_threshold) are excluded

   b) Content Analysis

   - Analyzes average token count for each tokenizer-column combination
   - Determines which features to create based on token counts

2. **Feature Creation Strategy**
   The function creates features systematically based on data characteristics:

   a) **Exact Match Features**: Created for all columns that pass the null threshold

   ```python
   # Every column gets an exact match feature
   ExactMatchFeature(column_name, column_name)
   ```

   b) **Numeric Features**: Created for columns that are detected as numeric

   ```python
   # Numeric columns get relative difference features
   RelDiffFeature(column_name, column_name)
   ```

   c) **Token-based Features**:  
   Created based on tokenizer analysis

   ```python
   # For each tokenizer-column combination with avg_count >= 3:
   # Creates all similarity functions (TF-IDF, Jaccard, SIF, Overlap, Cosine)
   TFIDFFeature(column_name, column_name, tokenizer=tokenizer)
   JaccardFeature(column_name, column_name, tokenizer=tokenizer)
   SIFFeature(column_name, column_name, tokenizer=tokenizer)
   OverlapCoeffFeature(column_name, column_name, tokenizer=tokenizer)
   CosineFeature(column_name, column_name, tokenizer=tokenizer)
   ```

   d) **Special Features for AlphaNumeric Tokenizer**:  
   Additional features for alphanumeric data

   ```python
   # Only for AlphaNumericTokenizer with avg_count <= 10:
   MongeElkanFeature(column_name, column_name, tokenizer=tokenizer)
   EditDistanceFeature(column_name, column_name)
   SmithWatermanFeature(column_name, column_name)
   ```

3. **Default Tokenizers and Similarity Functions**  
   When no custom tokenizers or similarity functions are provided:

   **Default Tokenizers:**

   - `StrippedWhiteSpaceTokenizer()`: Splits on whitespace and normalizes
   - `NumericTokenizer()`: Extracts numeric values from text
   - `QGramTokenizer(3)`: Creates 3-character sequences (3-grams)

   **Default Similarity Functions:**

   - `TFIDFFeature`: Term frequency-inverse document frequency similarity
   - `JaccardFeature`: Set-based similarity using intersection over union
   - `SIFFeature`: Smooth inverse frequency similarity
   - `OverlapCoeffFeature`: Set overlap coefficient
   - `CosineFeature`: Vector space cosine similarity

4. **Extra Tokenizers**  
   If a user wants more tokenizers, we provide the following extra implemented tokenizers:

   - `AlphaNumericTokenizer()`: Extracts alphanumeric sequences
   - `QGramTokenizer(5)`: Creates 5-character sequences (5-grams)
   - `StrippedQGramTokenizer(3)`: Creates 3-grams with whitespace stripped
   - `StrippedQGramTokenizer(5)`: Creates 5-grams with whitespace stripped

5. **Feature Creation Example**

   ```python
   # Real-world example - note that a_cols and b_cols must be the same
   features = create_features(
       customers_df, prospects_df,
       ['name', 'address', 'revenue'],
       ['name', 'address', 'revenue']
   )

   # What happens behind the scenes:
   # 1. All columns get ExactMatchFeature
   # 2. If 'revenue' is numeric, it gets RelDiffFeature
   # 3. For each tokenizer (Defaults: StrippedWhiteSpace, Numeric, QGram):
   #    - If average token count >= 3 over the column, creates all 5 similarity functions
   #    - If using AlphaNumericTokenizer and the average token count <= 10 over the column, it adds 3 extra features:
   #      * MongeElkanFeature: String similarity using Jaro-Winkler
   #      * EditDistanceFeature: Levenshtein edit distance
   #      * SmithWatermanFeature: Smith-Waterman sequence alignment

   # Typical output might include:
   # - ExactMatchFeature for name, address, revenue
   # - RelDiffFeature for revenue (if numeric)
   # - TFIDFFeature, JaccardFeature, SIFFeature, OverlapCoeffFeature, CosineFeature
   #   for each tokenizer-column combination with sufficient tokens
   ```

### featurize()

This function takes the features created earlier and applies them to your data. Here's how it works:

```python
def featurize(
    features: List[Callable],           # Feature objects
    A: Union[pd.DataFrame, SparkDataFrame],  # Your first dataset
    B: Union[pd.DataFrame, SparkDataFrame],  # Your second dataset
    candidates: Union[pd.DataFrame, SparkDataFrame],  # Which pairs to compare
    output_col: str = 'features',       # Name for feature vector column
    fill_na: float = 0.0                # Value for missing data
) -> pd.DataFrame
```

**Parameter Explanations:**

`features`: Feature objects for computing similarity

- What it is: List of callable feature objects (typically from `create_features()`, but can be from another source)
- Purpose: These objects define how to compare columns between datasets A and B
- What they contain: Each feature combines a tokenizer (text processing method) and similarity function (comparison method)
- Why you need them: Raw data can't be directly compared - features convert text/numbers into similarity scores

`A`: Your first dataset

- What it is: A pandas DataFrame or Spark DataFrame containing records from dataset A
- Schema requirement: Must have an `_id` column with unique identifiers for each record
- Why you need them: Contains the actual record data that will be tokenized and compared
- Purpose: These are your "candidate" records - potential matches for records in B

`B`: Your second dataset

- What it is: A pandas DataFrame or Spark DataFrame containing records from dataset B
- Schema requirement: Must have an `_id` column with unique identifiers for each record
- Why you need them: Contains the actual record data that will be tokenized and compared
- Purpose: These are your "reference" records - the ones you want to find duplicates or matches for

`candidates`: Specification of record pairs to analyze

- What it is: A pandas DataFrame or Spark DataFrame that specifies exactly which record pairs should be compared for matching
- Purpose: Defines the specific comparisons you want to perform between datasets A and B
- Required columns:
  - `id2`: Record identifiers from dataset B (these should match the `_id` column in B)
  - `id1_list`: Record identifiers from dataset A (these should match the `_id` column in A) to compare against each B record (as lists)
- Data source: These record pairs are typically generated by external blocking systems (like Sparkly(link))

`output_col`: Name for the resulting feature vector column

- What it is: String that will become the column name containing computed feature vectors
- Default value: 'features'
- Purpose: Allows you to customize the output column name to match your naming conventions
- What gets stored: Each row represents one record pair comparison and contains an array of similarity scores (one score per feature function)

`fill_na`: Default value for missing data

- What it is: The value to use when similarity computation fails due to missing data
- Default behavior: 0.0 (no similarity)
- Common alternatives: -1.0 (unknown), numpy.nan, or other float values
- Why important: Missing data is common, and the system needs a consistent way to handle it

1. **Understanding the Candidates DataFrame**  
   This specifies which record pairs to analyze for matching:

   ```python
   # Example candidates DataFrame (typically from external blocking system)
   candidates = pd.DataFrame({
       'id1_list': [[10, 11], [12, 13]],  # Record IDs from dataset A (as lists)
       'id2': [1, 2],                     # Record IDs from dataset B
       'source': ['blocking_system', 'manual_review'] # Possible Metadata, will not impact the featurization
   })

   # This means:
   # - Analyze if A record 10 matches B record 1
   # - Analyze if A record 11 matches B record 1
   # - Analyze if A record 12 matches B record 2
   # - Analyze if A record 13 matches B record 2
   ```

2. **Feature Vector Generation**
   What it is: Converting comparisons into numbers that machine learning models can understand

   ```python
   # Example: Comparing two company records
   record1 = {
       "name": "ACME Corp",
       "address": "123 Main St",
       "revenue": "1M"
   }

   record2 = {
       "name": "Acme Corporation",
       "address": "123 Main Street",
       "revenue": "1.1M"
   }

   # Feature vector might look like:
   # [name_similarity, address_similarity, revenue_similarity]
   # [0.9, 0.95, 0.98]
   ```

3. **Complete Example**

   ```python
   feature_vectors = featurize(
       features,              # Features from create_features() (or other program)
       customers_df,          # Your first dataset
       prospects_df,          # Your second dataset
       candidates_df,         # Pairs we want to compare
       output_col='feature_vectors' # Name of the column we want to save the vectors in
       # fill_na defaults to 0.0, no need to specify unless you want a different fill value
   )

   # Result DataFrame looks like:
   result = pd.DataFrame({
       'id2': [1, 1, 2, 2],
       'id1': [10, 11, 12, 13],                # id1_list gets expanded to individual id1 entries
       'feature_vectors': [
           [0.9, 0.95, 0.98, 0.85, 0.92],  # Feature vector for A record 10 vs B record 1
           [0.3, 0.2, 0.1, 0.15, 0.25],    # Feature vector for A record 11 vs B record 1
           [0.8, 0.7, 0.9, 0.75, 0.88],    # Feature vector for A record 12 vs B record 2
           [0.1, 0.05, 0.02, 0.08, 0.12]   # Feature vector for A record 13 vs B record 2
       ],
       'source': ['blocking_system', 'blocking_system', 'manual_review', 'manual_review']  # Preserved from candidates
   })
   ```

### down_sample()

What it does: When you have large datasets, this function can be helpful to sample data while preserving high-scoring and low-scoring combinations.

```python
def down_sample(
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Your feature vectors
    percent: float,                            # How much to keep (0.0 to 1.0)
    search_id_column: str,                     # Column with unique IDs
    score_column: str = 'score',               # Column with similarity scores
    bucket_size: int = 1000                    # Hash bucket size for representative sampling
) -> Union[pd.DataFrame, SparkDataFrame]
```

**Parameter Explanations:**

`fvs`: Your feature vectors dataset

- What it is: A DataFrame containing feature vectors - can be pandas or Spark DataFrame
- Required content: Must contain feature vectors and similarity scores for record pairs
- Schema expectation: Must have columns with record IDs and a column with computed similarity scores
- Purpose: This is the data you want to reduce in size while maintaining representativeness

`percent`: Fraction of data to keep

- What it is: A decimal number between 0.0 and 1.0 representing how much data to retain
- Example values: 0.1 = keep 10%, 0.5 = keep 50%, 0.8 = keep 80%
- Purpose: Controls the size reduction - smaller values create smaller, faster-to-process datasets
- Trade-off consideration: Smaller percentages run faster but may lose important patterns

`search_id_column`: Unique identifier for each comparison

- What it is: String name of the column that uniquely identifies each record pair comparison
- Purpose: Ensures each comparison can be uniquely tracked during the sampling process
- Requirements: Values in this column must be unique across all rows
- Why needed: Prevents accidentally sampling the same comparison multiple times

`score_column`: Similarity score column name

- What it is: String name of the column containing similarity scores for each record pair
- Data type expected: Numeric values where higher numbers indicate greater similarity
- Purpose: Used to create stratified samples across different similarity levels
- Why important: Ensures the sample maintains the same distribution of high/medium/low similarity pairs as the original data

`bucket_size`: Hash bucket size for representative sampling

- What it is: Integer specifying the target size of each hash bucket for data partitioning
- Default value: 1000 records per bucket
- Purpose: Controls how the data is split into buckets for representative sampling - larger buckets provide more granular sampling
- When to adjust: Use smaller values for more granular sampling, larger values for faster processing

**Why you may need downsampling:**

- Large datasets are slow to process
- You want to train models on manageable amounts of data
- You need to maintain the same patterns in a smaller dataset

**How it works:**

1. **Score Analysis**: Looks at your similarity scores to understand the distribution

   ```python
   # Example score distribution in your data
   # Note: Interpretation depends on your similarity functions
   # You need to understand what your scoring functions return:
   # - Some functions return higher scores for more similar items
   # - Some functions return higher scores for less similar items
   # - Some functions have different ranges (0-1, 0-100, etc.)
   high_scores = [0.95, 0.94, 0.93, 0.92, 0.91]      # Likely matches (depending on your scoring)
   medium_scores = [0.7, 0.6, 0.5, 0.4, 0.35]        # Uncertain cases
   low_scores = [0.3, 0.2, 0.1, 0.05, 0.02]          # Likely non-matches (depending on your scoring)
   ```

2. **Stratified Sampling**: Takes samples from each score range proportionally

   ```python
   sampled_data = down_sample(
       feature_vectors,
       percent=0.1,                    # Keep 10% of data
       search_id_column='pair_id',     # Your unique pair identifier
       score_column='similarity',      # Your similarity score column
       bucket_size=500                 # The number of records in each bucket
   )

   # If you had 100,000 pairs, you'll get 10,000 pairs
   ```

### create_seeds()

What it does: Creates initial training examples when you don't have any labeled data yet. Seeds are the starting point for teaching your model what matches look like.

```python
def create_seeds(
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Your feature vectors
    nseeds: int,                               # How many examples to create
    labeler: Union[Labeler, Dict],             # How to label the examples
    score_column: str = 'score'                # Column with similarity scores
) -> pd.DataFrame
```

**Parameter Explanations:**

`fvs`: Your feature vectors dataset

- What it is: A DataFrame containing feature vectors - can be pandas or Spark DataFrame
- Required content: Must have columns with record IDs and a column with computed similarity scores
- Data source: This is typically the same data you might pass to `down_sample()`, get back from `down_sample()` (if you want to work with a subset of your original data), or any dataset with feature vectors
- Purpose: Provides the pool of unlabeled record pairs from which to select examples for labeling

`nseeds`: Number of training examples to create

- What it is: Integer specifying how many labeled examples you want the function to generate
- Purpose: Controls the size of your initial training dataset
- Trade-offs: More seeds provide better training data but require more labeling effort
- Planning consideration: Consider how much time you have for manual labeling (if using human labeler), and if you are going to use Active Learning (with label_data) you will likely need fewer seeds

`labeler`: Method for determining if record pairs match

- What it is: Either a Labeler object or configuration dictionary that defines how to label record pairs
- Purpose: Provides the "ground truth" labels needed to train your machine learning model
- How it works: Takes two record IDs and returns a label indicating whether they represent the same entity

_See [Built-in Labeler Classes](#built-in-labeler-classes) for available options and usage._

`score_column`: Similarity score column identifier

- What it is: String name of the column in your fvs DataFrame containing similarity scores
- Data requirements: Column must contain numeric values representing similarity between record pairs
- Purpose: Used to select high-scoring and low-scoring pairs for labeling
- Strategy: THe function uses these scores to pick high-confidence matches and high-confidence non-matches

**Why you need seeds:** Machine learning models need examples of both matches and non-matches to learn patterns.

**Example Process:**

```python
from MadLib import GoldLabeler

# Suppose you have a DataFrame of known matches:
gold_df = pd.DataFrame({
    'id1': [10, 12, 14],
    'id2': [1, 2, 3]
})

# Create a GoldLabeler instance as your labeler spec
labeler = GoldLabeler(gold_df)

# Create 100 seeds
initial_labeled_data = create_seeds(feature_vectors, nseeds=100, labeler=labeler)

# Typical output DataFrame from create_seeds:
print(initial_labeled_data.head())

# Example output:
#    id1  id2                features   score  label
# 0   10    1  [0.9, 0.95, 0.98]       0.95   1.0
# 1   11    1  [0.3, 0.2, 0.1]         0.20   0.0
# 2   12    2  [0.8, 0.7, 0.9]         0.80   1.0
# 3   13    2  [0.1, 0.05, 0.02]       0.05   0.0
# 4   14    3  [0.85, 0.8, 0.9]        0.85   1.0
```

### train_matcher()

What it does: Takes your labeled examples and teaches a machine learning model to recognize patterns that indicate matches.

```python
def train_matcher(
    model_spec: Union[Dict, MLModel],     # What type of model to train
    labeled_data: Union[pd.DataFrame, SparkDataFrame],  # Your labeled examples
    feature_col: str = "features",        # Column name for your feature vectors
    label_col: str = "label"              # Column name for your labels
) -> MLModel
```

**Parameter Explanations:**

`model_spec`: Machine learning model configuration

- What it is: Either a dictionary specifying model type and parameters, or a pre-configured MLModel object
- Purpose: Defines what type of machine learning algorithm to use and how to configure it
- Common types: Random Forest, Logistic Regression, Gradient Boosting
- Why it is configurable: Different datasets may work better with different algorithms

`labeled_data`: Your training dataset

- What it is: A pandas DataFrame or Spark DataFrame containing record pairs with known match/non-match labels
- Data source: Typically the output from `create_seeds()` or manually labeled data
- Required columns: Must contain feature vectors and ground truth labels
- Schema requirements: Must have the columns specified in feature_col and label_col parameters

`feature_col`: Feature vector column identifier

- What it is: String name of the column in labeled_data that contains the computed feature vectors
- Expected content: Each row must contain an array/list of numeric similarity scores
- Data source: These are the feature vectors computed by the `featurize()` function or by an outside system
- Why it is needed: Tells the training algorithm which column contains the input features for learning
- Typical names: 'features', 'feature_vectors', 'fvs'

`label_col`: Ground truth labels column identifier

- What it is: String name of the column containing the correct match/non-match labels
- Expected values: Typically 1.0 for matches, 0.0 for non-matches
- Purpose: Provides the "answer key" that the model learns to predict
- Critical requirement: Labels must be accurate for the model to learn correct patterns

**Why you need it:** Once you have labeled examples, you need a model that can apply those patterns to new, unlabeled data.

**Understanding Model Specifications:**  
You can specify different types of models:

1. **Random Forest** (good default choice):

   ```python
   from sklearn.ensemble import RandomForestClassifier

   model_spec = {
       'model_type': 'sklearn',
       'model': RandomForestClassifier,
       'model_args': {
           'n_estimators': 100,        # Number of trees (more = better but slower)
           'max_depth': 10,            # How deep each tree can go (prevents overfitting)
           'random_state': 42          # For reproducible results
       },
       'nan_fill': 0.0                 # Important: Fill NaN values with 0.0
   }
   ```

2. **Logistic Regression** (simple and interpretable):

   ```python
   from sklearn.linear_model import LogisticRegression

   model_spec = {
       'model_type': 'sklearn',
       'model': LogisticRegression,
       'model_args': {
           'C': 1.0,                   # Regularization strength (higher = less regularization)
           'max_iter': 1000            # Maximum training iterations
       },
       'nan_fill': 0.0                 # Important: Fill NaN values with 0.0
   }
   ```

3. **Gradient Boosting** (often highest accuracy):

   ```python
   from sklearn.ensemble import GradientBoostingClassifier

   model_spec = {
       'model_type': 'sklearn',
       'model': GradientBoostingClassifier,
       'model_args': {
           'n_estimators': 100,        # Number of boosting stages
           'learning_rate': 0.1,       # How much each tree contributes
           'max_depth': 6              # Depth of individual trees
       },
       'nan_fill': 0.0                 # Important: Fill NaN values with 0.0
   }
   ```

4. **Spark ML Models** (for distributed training):

   ```python
   from pyspark.ml.classification import RandomForestClassifier
   model_spec = {
       'model_type': 'sparkml',
       'model': RandomForestClassifier,  # Spark ML RandomForestClassifier
       'model_args': {
           'numTrees': 100,
           'maxDepth': 10
       }
   }
   ```

5. **Additional Options** (for sklearn models):

   ```python
   from sklearn.ensemble import RandomForestClassifier

   model_spec = {
       'model_type': 'sklearn',
       'model': RandomForestClassifier,
       'model_args': {
           'n_estimators': 100,
           'random_state': 42
       },
       'nan_fill': 0.0,               # **REQUIRED**: Fill NaN values with 0.0 for sklearn models
       'execution': 'local',           # 'local' or 'spark' (default: 'local')
       'use_floats': True             # Use float32 for memory efficiency (default: True)
   }
   ```

**What your labeled_data should look like:**

```python
# Example labeled_data DataFrame
labeled_data = pd.DataFrame({
    'id1': [10, 11, 12, 13, 14, 15],
    'id2': [1, 1, 2, 2, 3, 3],
    'features': [
        [0.9, 0.95, 0.98],    # Feature Vector between id2 = 1 and id1 = 10
        [0.3, 0.2, 0.1],      # Feature Vector between id2 = 1 and id1 = 11
        [0.8, 0.7, 0.9],      # Feature Vector between id2 = 2 and id1 = 12
        [0.1, 0.05, 0.02],    # Feature Vector between id2 = 2 and id1 = 13
        [0.85, 0.8, 0.9],     # Feature Vector between id2 = 3 and id1 = 14
        [0.2, 0.15, 0.1]      # Feature Vector between id2 = 3 and id1 = 15
    ],
    'label': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # Labels: 1.0=match, 0.0=non-match
})
```

**Example Training Process:**

```python
# Train a model
from sklearn.ensemble import RandomForestClassifier

model = train_matcher(
    model_spec={
        'model_type': 'sklearn',
        'model': RandomForestClassifier,
        'model_args': {'n_estimators': 100},
        'nan_fill': 0.0                 # Fill NaN values with 0.0
    },
    labeled_data=labeled_examples,
    feature_col='features',
    label_col='label'
)

# The model can learn patterns like:
# "When name similarity > 0.8 AND address similarity > 0.7 → likely match"
# "When all similarities < 0.3 → likely non-match"
# "When name similarity > 0.9 but address similarity < 0.2 → uncertain"
```

### apply_matcher()

What it does: Uses your trained model to predict matches on new data that hasn't been labeled yet.

```python
def apply_matcher(
    model: Union[MLModel, SKLearnModel, SparkMLModel],  # Your trained model
    df: Union[pd.DataFrame, SparkDataFrame],            # Data to make predictions on
    feature_col: str,                                   # Column with feature vectors (the input to the model)
    output_col: str                                     # The column that will have the predictions
) -> pd.DataFrame
```

**Parameter Explanations:**

`model`: Your trained machine learning model

- What it is: A trained MLModel object returned from `train_matcher()`, a trained SKLearn model, or a SparkML Transformer
- Requirements: Must be a model that has already been trained on labeled data
- Purpose: Contains the learned patterns for identifying matches between records
- What it knows: How to convert feature vectors into match/non-match predictions

`df`: Your unlabeled data to predict on

- What it is: A pandas DataFrame or Spark DataFrame containing record pairs with computed feature vectors
- Data source: If you are using the MadLib, it will be your output from `featurize()`
- Required content: Must contain a column with feature vectors (specified by feature_col parameter)
- Schema requirement: Must have the same feature vector structure as the training data
- Purpose: These are the new record pairs you want to classify as matches or non-matches

`feature_col`: Feature vector column identifier

- What it is: String name of the column in df that contains the feature vectors
- Expected content: Each row must contain an array/list of numeric similarity scores
- Consistency requirement: Feature vectors must have same structure as those used during training
- Purpose: Tells the model which column contains the input data for making predictions
- Common names: 'features', 'feature_vectors', 'fvs'

`output_col`: Prediction results column name

- What it is: String that will become the name of the new column containing predictions
- Output format: Column will contain numeric values (1.0 = match, 0.0 = non-match)
- Purpose: Allows you to customize the output column name to fit your workflow

**What it returns:**

```python
# Your input data with predictions added
result = pd.DataFrame({
    'id1': [20, 21, 22, 23],
    'id2': [3, 3, 4, 4],
    'features': [
        [0.85, 0.9, 0.92],    # High scores for some similarity functions
        [0.2, 0.15, 0.1],     # Low scores for some similarity functions
        [0.75, 0.8, 0.85],    # Medium-high scores for some similarity functions
        [0.4, 0.45, 0.5]      # Medium scores for some similarity functions
    ],
    'predictions': [1.0, 0.0, 1.0, 0.0]       # Model predictions
})
```

**Example Usage:**

```python
# Apply your trained model to new data
predictions = apply_matcher(
    model,                              # Your trained model
    new_feature_vectors,                # New data to predict on
    feature_col='features',             # Column with features
    output_col='match_prediction'       # Where to put results
)

# Now you can see which pairs the model thinks are matches
matches = predictions[predictions['match_prediction'] == 1.0]
print(f"Found {len(matches)} potential matches")
```

### label_data()

What it does: Implements active learning by selecting which examples would be most helpful to label next, improving your model iteratively.

```python
def label_data(
    model_spec: Union[Dict, MLModel],     # Untrained model or model config
    mode: Literal["batch", "continuous"], # How to do the labeling
    labeler_spec: Union[Dict, Labeler],   # How to label examples
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Unlabeled data
    seeds: Optional[pd.DataFrame] = None  # Existing labeled data
) -> pd.DataFrame
```

**Parameter Explanations:**

`model_spec`: Model for active learning

- What it is: Either a dictionary specifying model type and parameters, or a pre-configured MLModel object
- Purpose: Defines what type of machine learning algorithm to use and how to configure it
- Common types: Random Forest, Logistic Regression, Gradient Boosting
- Why it is configurable: Different datasets may work better with different algorithms

`mode`: Active learning strategy

- What it is: String specifying how to conduct the labeling process
- "batch" mode: Label a group of examples, then retrain the model, then repeat
- "continuous" mode: Examples are labeled in one thread, the model gets trained in another thread and provides new examples continuously

`labeler_spec`: Method for generating labels

- What it is: Either a Labeler object or configuration dictionary that defines how to label examples
- Purpose: Provides the mechanism to convert unlabeled examples into labeled training data

_See [Built-in Labeler Classes](#built-in-labeler-classes) for available options and usage._

`fvs`: Pool of unlabeled feature vectors

- What it is: A pandas DataFrame or Spark DataFrame containing feature vectors for record pairs that need labels
- Data source: If you are using the MadLib, it will be your output from `featurize()`
- Content requirements: Must contain feature vectors and any metadata needed for labeling decisions
- Purpose: Provides the candidate pool from which active learning will select examples for labeling

`seeds`: Initial labeled data (optional)

- What it is: A pandas DataFrame containing already-labeled examples to start the active learning process
- Data source: Could be the output from `create_seeds()` or any previously labeled data
- Why it is optional: If None, active learning starts with no prior knowledge
- Why it is helpful: Provides initial model training data to start the active learning process
- Content requirements: Must contain feature vectors and labels in the same format as expected output

**Why you may want to use label_data():** Instead of randomly labeling data, this focuses on examples where labeling will most improve your model's accuracy.

**How Active Learning Works:**

**Uncertainty Sampling**: Finds examples the model is most unsure about

```python
# Model predictions with confidence
uncertain_examples = [
    {'prediction': 1.0, 'confidence': 0.51},  # Very uncertain! (close to 0.5)
    {'prediction': 0.0, 'confidence': 0.49},  # Very uncertain! (close to 0.5)
    {'prediction': 1.0, 'confidence': 0.95},  # Very certain (high confidence)
    {'prediction': 0.0, 'confidence': 0.95}   # Very certain (high confidence)
]
# Active learning would pick the first two for labeling
```

## Built-in Labeler Classes

MadMatcher provides several built-in labeler classes for different labeling workflows. You can use these directly or extend them for custom logic. All labelers inherit from the public `Labeler` abstract class:

- **GoldLabeler**: For automated labeling using a gold standard set of matches.

  - Usage: `GoldLabeler(gold)` where `gold` is a DataFrame with columns `id1` and `id2` for known matches.
  - Returns `1.0` for pairs in the gold set, `0.0` otherwise.

- **CLILabeler**: For interactive, command-line labeling by a human.

  - Usage: `CLILabeler(a_df, b_df, id_col='_id')`
  - Presents record pairs to the user in the terminal and asks for a label (`yes`, `no`, `unsure`, or `stop`).
  - Useful for manual review and active learning.

- **CustomLabeler**: For advanced users who want to implement their own labeling logic.

  - Usage: Subclass `CustomLabeler` and implement the `label_pair(row1, row2)` method.
  - Example:

    ```python
    from MadLib import CustomLabeler

    class MyCustomLabeler(CustomLabeler):
        def label_pair(self, row1, row2):
            # Custom logic here
            return 1.0 if row1['email'] == row2['email'] else 0.0
    ```

- **ThresholdLabeler** (example): For rule-based labeling using similarity scores.
  - See the example above for how to implement this by extending `Labeler`.

Refer to this section when choosing or instantiating a labeler for functions like `create_seeds`, `label_data`, or any workflow that requires labeling record pairs.

## Custom Feature Development

When the built-in features aren't enough, you can create your own:

```python
class PhoneNumberFeature(Featurizer):
    def __init__(self, a_column, b_column):
        self.a_column = a_column  # Column name in dataset A
        self.b_column = b_column  # Column name in dataset B

    def __str__(self):
        # Must return unique identifier for this feature
        return f"phone_similarity({self.a_column}, {self.b_column})"

    def __call__(self, record_b, records_a):
        # record_b: one record from dataset B
        # records_a: list of records from dataset A to compare against

        phone_b = self.normalize_phone(record_b[self.b_column])
        similarities = []

        for record_a in records_a:
            phone_a = self.normalize_phone(record_a[self.a_column])
            similarity = self.compare_phones(phone_a, phone_b)
            similarities.append(similarity)

        return similarities

    def normalize_phone(self, phone):
        # Remove formatting: "(555) 123-4567" → "5551234567"
        if phone is None or pd.isna(phone):
            return ""
        return ''.join(c for c in str(phone) if c.isdigit())

    def compare_phones(self, phone1, phone2):
        if phone1 == phone2:
            return 1.0  # Exact match
        elif len(phone1) >= 7 and len(phone2) >= 7 and phone1[-7:] == phone2[-7:]:
            return 0.8  # Same last 7 digits (same local number, different area code)
        elif len(phone1) >= 4 and len(phone2) >= 4 and phone1[-4:] == phone2[-4:]:
            return 0.3  # Same last 4 digits (might be related)
        else:
            return 0.0  # Different numbers
```

## Best Practices

### Starting a New Matching Project

1. **Understand Your Data**

   ```python
   # Always start by exploring your data
   print("Dataset A shape:", df_a.shape)
   print("Dataset B shape:", df_b.shape)
   print("Dataset A columns:", df_a.columns.tolist())
   print("Dataset B columns:", df_b.columns.tolist())

   # Check data quality
   print("Missing values in A:", df_a.isnull().sum())
   print("Missing values in B:", df_b.isnull().sum())

   # Ensure _id columns exist and are unique
   print("Dataset A _id unique:", df_a['_id'].nunique() == len(df_a))
   print("Dataset B _id unique:", df_b['_id'].nunique() == len(df_b))
   ```

2. **Start Small**

   ```python
   # Use a small sample first to test your pipeline
   sample_a = df_a.sample(1000)
   sample_b = df_b.sample(1000)

   # Develop and test with samples
   features = create_features(sample_a, sample_b, ...)
   ```

3. **Validate Each Step**

   ```python
   # Check your feature creation
   features = create_features(...)
   print(f"Created {len(features)} features")

   # Check your input candidates (from external blocking system)
   print(f"Received {len(candidates)} candidate pairs to analyze")
   print(f"Candidates cover {candidates['id2'].nunique()} records from dataset B")

   # Check your feature vectors
   feature_vectors = featurize(...)
   print(f"Feature vector shape: {feature_vectors['features'].iloc[0].shape}")
   ```
