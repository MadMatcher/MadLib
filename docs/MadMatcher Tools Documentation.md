<!-- MadMatcher Documentation -->

# MadMatcher Tools Technical Documentation

This document provides an in-depth technical overview of MadMatcher Tools' functionality, internal workings, and advanced use cases. For quick start guide and basic usage, refer to the README.md file. For complete API reference, see the API documentation.

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
   Example: Converting "John Smith" and "Smith, John" into features that capture name similarity
2. **Model Training**
   What it is: Teaching the system to recognize what makes records match
   How it works: Using known matches and non-matches to learn patterns

3. **Prediction**
   What it is: Using learned patterns to identify new matches
   Example: Determining if two previously unseen records represent the same entity

4. **Active Learning**
   What it is: Improving match quality by asking for human input on uncertain cases
   Why needed: Some matches are too ambiguous for automatic decision

### Understanding Features

A feature is a way to compare two values and determine how similar they are. Think of it as answering questions like:

- How similar are these two names?
- How likely are these two records to represent the same entity?

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

## Core Functions In-Depth

### create_features()

This function analyzes your data and creates the right combination of tokenizers and similarity functions. Let's break down how it works:

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

- What it is: Any pandas DataFrame containing records you want to find matches for
- Schema requirement: Must have the same column structure as dataset B (same types of information)
- Example: A customer database with columns like ['customer_id', 'name', 'address', 'phone']
- Purpose: These are your "reference" records - the ones you want to find duplicates or matches for

`B`: Your second dataset (pandas DataFrame)

- What it is: Any pandas DataFrame containing records you want to match against dataset A
- Schema requirement: Must have the same column structure as dataset A (same types of information)
- Example: A prospect database with columns like ['customer_id', 'name', 'address', 'phone']
- Purpose: These are your "candidate" records - potential matches for records in A
- Relationship to A: Even though column names may differ, the data types and meanings should correspond (name↔company_name, address↔location, phone↔contact_number)

`a_cols`: Column names from dataset A to use for comparison

- What it is: List of strings specifying which columns from A contain the data you want to compare
- Purpose: Tells the system which fields in A contain meaningful information for matching
- Example: ['name', 'address', 'phone'] - these are the columns that will help identify if two records represent the same entity
- Requirement: These columns should be the same as the columns for b_cols

`b_cols`: Column names from dataset B to use for comparison

- What it is: List of strings specifying which columns from B contain the data you want to compare
- Purpose: Tells the system which fields in B contain meaningful information for matching
- Example: ['name', 'address', 'phone'] - these are the columns that will help identify if two records represent the same entity
- Requirement: These columns should be the same as the columns for a_cols

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

   - Text: Names, addresses, descriptions
   - Numbers: Ages, prices, quantities
   - Categories: Status codes, types, classifications

   b) Content Analysis

   - How many words are typical?
   - Are there patterns in the text?
   - How often are values missing?

2. **Feature Selection**
   Based on the analysis, different strategies are chosen:

   a) For Text Columns:
   Short text (< 5 words, like names):

   ```python
   # Example: Comparing names
   "John Smith" vs "Smith, John"
   # Uses:
   # 1. Stripped Whitespace tokenizer to split on words
   # 2. Jaccard and Overlap Coefficient similarity functions
   ```

   Medium text (5-20 words, like addresses):

   ```python
   # Example: Comparing addresses
   "123 Main Street, Apt 4B, New York, NY 10001"
   # Uses:
   # 1. 3-gram tokenizer for character sequences
   # 2. TF-IDF similarity function for term weighting
   ```

   Long text (> 20 words, like descriptions):

   ```python
   # Example: Comparing product descriptions
   "High-quality leather office chair with adjustable height..."
   # Uses:
   # 1. Stripped Whitespace tokenizer for word-level analysis
   # 2. TF-IDF and Cosine similarity functions
   ```

   b) For Numeric Columns:

   ```python
   # Example: Comparing ages
   25 vs 26  # Uses Numeric tokenizer to extract numbers
   25 vs 75  # Computes similarity based on numeric difference

   # Example: Comparing prices
   $99.99 vs $100.00  # Numeric tokenizer handles currency symbols
   $99.99 vs $999.99  # Returns low similarity for large differences
   ```

   c) For Categorical Columns:

   ```python
   # Example: Comparing status codes
   "ACTIVE" vs "Active"  # Stripped Whitespace tokenizer normalizes case
   "COMPLETED" vs "DONE" # Jaccard similarity for exact token matching
   ```

3. **Feature Creation Example**

   ```python
   # Real-world example - note that a_cols and b_cols must be the same
   features = create_features(
       customers_df, prospects_df,
       ['name', 'address', 'revenue'],
       ['name', 'address', 'revenue']
   )

   # What happens behind the scenes:
   # 1. name columns:
   #    - Stripped Whitespace tokenizer for word-level analysis
   #    - Jaccard and Overlap Coefficient similarity functions

   # 2. address columns:
   #    - 3-gram tokenizer for character sequences
   #    - TF-IDF similarity function for term weighting

   # 3. revenue columns:
   #    - Numeric tokenizer to extract numeric values
   #    - Cosine similarity for numeric comparison
   ```

### featurize()

This function takes the features created earlier and applies them to your data. Here's how it works:

```python
def featurize(
    features: List[Callable],           # Feature functions from create_features()
    A: pd.DataFrame,                    # Your first dataset
    B: pd.DataFrame,                    # Your second dataset
    candidates: pd.DataFrame,           # Which pairs to compare
    output_col: str = 'fvs',           # Name for feature vector column
    fill_na: Any = None                 # Value for missing data
) -> pd.DataFrame
```

**Parameter Explanations:**

`features`: Feature functions for computing similarity

- What it is: List of callable feature objects returned by `create_features()`
- Purpose: These functions know how to compare specific columns between datasets A and B
- What they contain: Each feature combines a tokenizer (text processing method) and similarity function (comparison method)
- Why you need them: Raw data can't be directly compared - features convert text/numbers into comparable similarity scores

`A`: Your first dataset (same as in create_features)

- What it is: The same pandas DataFrame you used in `create_features()`
- Schema requirement: Must be identical to what you passed to `create_features()`
- Purpose: Contains the actual record data that will be tokenized and compared
- Why needed again: The feature functions need access to the raw data to compute similarities

`B`: Your second dataset (same as in create_features)

- What it is: The same pandas DataFrame you used in `create_features()`
- Schema requirement: Must be identical to what you passed to `create_features()`
- Purpose: Contains the actual record data that will be compared against dataset A
- Why needed again: The feature functions need access to the raw data to compute similarities

`candidates`: Specification of record pairs to analyze

- What it is: A pandas DataFrame that specifies exactly which record pairs should be compared for matching
- Purpose: Defines the specific comparisons you want to perform between datasets A and B
- Required columns:
  - `id2`: Record identifiers from dataset B (these should match the index or a unique ID column in B)
  - `id1_list`: Record identifiers from dataset A to compare against each B record
- Data source: These record pairs are typically provided by external blocking/candidate generation systems
- Focus: This package focuses on the matching step - determining if the provided pairs are actual matches

`output_col`: Name for the resulting feature vector column

- What it is: String that will become the column name containing computed feature vectors
- Default value: 'fvs' (short for "feature vectors")
- Purpose: Allows you to customize the output column name to match your naming conventions
- What gets stored: Each row will contain an array of similarity scores (one per feature function)

`fill_na`: Default value for missing data

- What it is: The value to use when similarity computation fails due to missing data
- Default behavior: None (which may cause errors if not handled properly)
- Common alternatives: 0.0 (no similarity), -1.0 (unknown), or numpy.nan
- Why important: Missing data is common, and the system needs a consistent way to handle it

1. **Understanding the Candidates DataFrame**
   This specifies which record pairs to analyze for matching:

   ```python
   # Example candidates DataFrame (typically from external blocking system)
   candidates = pd.DataFrame({
       'id2': [1, 1, 2, 2],                    # Record IDs from dataset B
       'id1_list': [[10], [11], [12], [13]],  # Record IDs from dataset A (as lists)
       'source': ['blocking_system', 'manual_review', 'blocking_system', 'manual_review']
   })

   # This means:
   # - Analyze if B record 1 matches A record 10
   # - Analyze if B record 1 matches A record 11
   # - Analyze if B record 2 matches A record 12
   # - Analyze if B record 2 matches A record 13
   ```

2. **Preprocessing**
   What it does: Prepares your data for comparison

   ```python
   # Example: Text standardization
   original = "JOHN's Company, LLC"
   standardized = "johns company llc"

   # Example: Number standardization
   original = "$1.2M"
   standardized = 1200000
   ```

3. **Feature Vector Generation**
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

4. **Complete Example**

   ```python
   feature_vectors = featurize(
       features,              # The features we created earlier
       customers_df,          # Your first dataset
       prospects_df,          # Your second dataset
       candidates_df,         # Pairs we want to compare
       output_col='feature_vectors',
       fill_na=0.0           # Use 0.0 for missing similarity scores
   )

   # Result DataFrame looks like:
   result = pd.DataFrame({
       'id2': [1, 1, 2, 2],
       'id1': [10, 11, 12, 13],                # id1_list gets expanded to individual id1 entries
       'feature_vectors': [
           [0.9, 0.95, 0.98],  # High similarity scores → likely match
           [0.3, 0.2, 0.1],    # Low similarity scores → likely not match
           [0.8, 0.7, 0.9],    # Medium-high scores → uncertain
           [0.1, 0.05, 0.02]   # Very low scores → definitely not match
       ],
       'source': ['blocking_system', 'manual_review', 'blocking_system', 'manual_review']  # Preserved from candidates
   })
   ```

### down_sample()

What it does: When you have too much data, this function intelligently reduces it while keeping the important patterns. Think of it like creating a representative sample of your population.

```python
def down_sample(
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Your feature vectors
    percent: float,                            # How much to keep (0.0 to 1.0)
    search_id_column: str,                     # Column with unique IDs
    score_column: str,                         # Column with similarity scores
    bucket_size: int = 1000                    # Size of processing chunks
) -> Union[pd.DataFrame, SparkDataFrame]
```

**Parameter Explanations:**

`fvs`: Your feature vectors dataset

- What it is: A DataFrame containing the output from `featurize()` - can be pandas or Spark
- Required content: Must contain feature vectors and similarity scores for record pairs
- Schema expectation: Should have columns with record IDs and a column with computed similarity scores
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

`bucket_size`: Processing chunk size

- What it is: Integer specifying how many records to process together in each batch
- Default value: 1000 records per batch
- Purpose: Controls memory usage during processing - larger values use more memory but may be faster
- When to adjust: Reduce if running out of memory, increase if processing is slow and you have enough memory

**Why you need it:**

- Large datasets are slow to process
- You want to train models on manageable amounts of data
- You need to maintain the same patterns in a smaller dataset

**How it works:**

1. **Score Analysis**: Looks at your similarity scores to understand the distribution

   ```python
   # Example score distribution in your data
   high_scores = [0.95, 0.94, 0.93, 0.92, 0.91]      # Likely matches (5%)
   medium_scores = [0.7, 0.6, 0.5, 0.4, 0.35]        # Uncertain (25%)
   low_scores = [0.3, 0.2, 0.1, 0.05, 0.02]          # Likely non-matches (70%)
   ```

2. **Stratified Sampling**: Takes samples from each score range proportionally

   ```python
   sampled_data = down_sample(
       feature_vectors,
       percent=0.1,                    # Keep 10% of data
       search_id_column='pair_id',     # Your unique pair identifier
       score_column='similarity',      # Your similarity score column
       bucket_size=500                 # Process 500 records at a time
   )

   # If you had 100,000 pairs, you'll get 10,000 pairs
   # But they'll represent the same score distribution as the original data
   ```

### create_seeds()

What it does: Creates initial training examples when you don't have any labeled data yet. Seeds are the starting point for teaching your model what matches look like.

```python
def create_seeds(
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Your feature vectors
    nseeds: int,                               # How many examples to create
    labeler: Union[Labeler, Dict],             # How to label the examples
    score_column: str                          # Column with similarity scores
) -> Union[pd.DataFrame, SparkDataFrame]
```

**Parameter Explanations:**

`fvs`: Your feature vectors dataset

- What it is: A DataFrame containing the output from `featurize()` with computed feature vectors
- Required content: Must contain feature vectors and similarity scores for unlabeled record pairs
- Data source: This is typically the same data you might pass to `down_sample()` or the full output from `featurize()`
- Purpose: Provides the pool of unlabeled record pairs from which to select examples for labeling

`nseeds`: Number of training examples to create

- What it is: Integer specifying how many labeled examples you want the function to generate
- Typical values: 100-5000 depending on your dataset size and complexity
- Purpose: Controls the size of your initial training dataset
- Trade-offs: More seeds provide better training data but require more labeling effort
- Planning consideration: Consider how much time you have for manual labeling (if using human labeler)

`labeler`: Method for determining if record pairs match

- What it is: Either a Labeler object or configuration dictionary that defines how to label record pairs
- Purpose: Provides the "ground truth" labels needed to train your machine learning model
- How it works: Takes two record IDs and returns a label indicating whether they represent the same entity
- Types available: Automatic (rule-based) or manual (human reviewer) labeling approaches

`score_column`: Similarity score column identifier

- What it is: String name of the column in your fvs DataFrame containing similarity scores
- Data requirements: Column should contain numeric values representing similarity between record pairs
- Purpose: Used to select diverse examples across different similarity levels for balanced training
- Strategy: Function uses these scores to pick high-confidence matches, high-confidence non-matches, and uncertain cases

**Why you need seeds:** Machine learning models need examples of both matches and non-matches to learn patterns.

**Understanding the Labeler:**
The labeler is what actually decides if two records match. It can be:

1. **Automatic Labeler** (using score thresholds):

   ```python
   labeler_config = {
       'type': 'threshold',
       'high_threshold': 0.8,    # Above this = definite match
       'low_threshold': 0.3      # Below this = definite non-match
       # Between 0.3 and 0.8 = uncertain, skip these
   }
   ```

2. **Manual Labeler** (human reviews):

   ```python
   class ManualLabeler(Labeler):
       def __call__(self, id1, id2):
           # Get the actual records to show to human
           record1 = get_record_from_dataset_a(id1)
           record2 = get_record_from_dataset_b(id2)

           print(f"Record A: {record1}")
           print(f"Record B: {record2}")

           response = input("Do these match? (y/n/u/s): ")
           if response == 'y': return 1.0      # Match
           elif response == 'n': return 0.0    # No match
           elif response == 'u': return 2.0    # Unsure
           elif response == 's': return -1.0   # Stop labeling
   ```

**How it works:**

1. **Selection Strategy**: Picks diverse examples across different similarity scores

   ```python
   # Selects examples like:
   # - High similarity (0.8-1.0) for likely positive examples
   # - Low similarity (0.0-0.3) for likely negative examples
   # - Medium similarity (0.4-0.7) for challenging/uncertain examples
   ```

2. **Example Process:**

   ```python
   seeds = create_seeds(
       feature_vectors,
       nseeds=500,                          # Want 500 labeled examples
       labeler=manual_labeler,              # Human will label them
       score_column='similarity_score'
   )

   # Result: DataFrame with 500 rows, each labeled as match/non-match
   # The function automatically picks diverse, informative examples
   ```

### train_matcher()

What it does: Takes your labeled examples (seeds) and teaches a machine learning model to recognize patterns that indicate matches.

```python
def train_matcher(
    model_spec: Union[Dict, MLModel],     # What type of model to train
    labeled_data: pd.DataFrame,           # Your labeled examples
    feature_col: str,                     # Column with feature vectors
    label_col: str                        # Column with match labels
) -> MLModel
```

**Parameter Explanations:**

`model_spec`: Machine learning model configuration

- What it is: Either a dictionary specifying model type and parameters, or a pre-configured MLModel object
- Purpose: Defines what type of machine learning algorithm to use and how to configure it
- Common types: Random Forest, Logistic Regression, Gradient Boosting, Neural Networks
- Why configurable: Different datasets may work better with different algorithms
- What it controls: Model architecture, hyperparameters, training behavior

`labeled_data`: Your training dataset

- What it is: A pandas DataFrame containing record pairs with known match/non-match labels
- Data source: Typically the output from `create_seeds()` or manually labeled data
- Required columns: Must contain feature vectors and ground truth labels
- Schema requirements: Should have the columns specified in feature_col and label_col parameters
- Quality importance: The quality of this data directly affects how well your model will perform

`feature_col`: Feature vector column identifier

- What it is: String name of the column in labeled_data that contains the computed feature vectors
- Expected content: Each row should contain an array/list of numeric similarity scores
- Data source: These are the feature vectors computed by `featurize()` function
- Why needed: Tells the training algorithm which column contains the input features for learning
- Typical names: 'fvs', 'feature_vectors', 'features'

`label_col`: Ground truth labels column identifier

- What it is: String name of the column containing the correct match/non-match labels
- Expected values: Typically 1.0 for matches, 0.0 for non-matches
- Data source: These labels come from human reviewers or automatic labeling rules
- Purpose: Provides the "answer key" that the model learns to predict
- Critical requirement: Labels must be accurate for the model to learn correct patterns

**Why you need it:** Once you have labeled examples, you need a model that can apply those patterns to new, unlabeled data.

**Understanding Model Specifications:**
You can specify different types of models:

1. **Random Forest** (good default choice):

   ```python
   model_spec = {
       'type': 'random_forest',
       'n_estimators': 100,        # Number of trees (more = better but slower)
       'max_depth': 10,            # How deep each tree can go (prevents overfitting)
       'random_state': 42          # For reproducible results
   }
   ```

2. **Logistic Regression** (simple and interpretable):

   ```python
   model_spec = {
       'type': 'logistic_regression',
       'C': 1.0,                   # Regularization strength (higher = less regularization)
       'max_iter': 1000            # Maximum training iterations
   }
   ```

3. **Gradient Boosting** (often highest accuracy):
   ```python
   model_spec = {
       'type': 'gradient_boosting',
       'n_estimators': 100,        # Number of boosting stages
       'learning_rate': 0.1,       # How much each tree contributes
       'max_depth': 6              # Depth of individual trees
   }
   ```

**What your labeled_data should look like:**

```python
# Example labeled_data DataFrame
labeled_data = pd.DataFrame({
    'id1': [10, 11, 12, 13, 14, 15],
    'id2': [1, 1, 2, 2, 3, 3],
    'feature_vectors': [
        [0.9, 0.95, 0.98],    # High similarity features
        [0.3, 0.2, 0.1],      # Low similarity features
        [0.8, 0.7, 0.9],      # Medium-high features
        [0.1, 0.05, 0.02],    # Very low features
        [0.85, 0.8, 0.9],     # High features
        [0.2, 0.15, 0.1]      # Low features
    ],
    'is_match': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # Labels: 1.0=match, 0.0=non-match
})
```

**Example Training Process:**

```python
# Train a model
model = train_matcher(
    model_spec={'type': 'random_forest', 'n_estimators': 100},
    labeled_data=labeled_examples,
    feature_col='feature_vectors',
    label_col='is_match'
)

# The model learns patterns like:
# "When name similarity > 0.8 AND address similarity > 0.7 → likely match"
# "When all similarities < 0.3 → likely non-match"
# "When name similarity > 0.9 but address similarity < 0.2 → uncertain"
```

### apply_matcher()

What it does: Uses your trained model to predict matches on new data that hasn't been labeled yet.

```python
def apply_matcher(
    model_spec: Union[MLModel, Any],      # Your trained model
    df: pd.DataFrame,                     # Data to make predictions on
    feature_col: str,                     # Column with feature vectors
    output_col: str                       # Where to put predictions
) -> pd.DataFrame
```

**Parameter Explanations:**

`model_spec`: Your trained machine learning model

- What it is: A trained MLModel object returned from `train_matcher()`
- Requirements: Must be a model that has already been trained on labeled data
- Purpose: Contains the learned patterns for identifying matches between records
- What it knows: How to convert feature vectors into match/non-match predictions
- Alternative formats: Can also accept scikit-learn or Spark ML model objects

`df`: Your unlabeled data to predict on

- What it is: A pandas DataFrame containing record pairs with computed feature vectors
- Data source: Typically output from `featurize()` for new, unlabeled record pairs
- Required content: Must contain a column with feature vectors (specified by feature_col parameter)
- Schema requirement: Must have the same feature vector structure as the training data
- Purpose: These are the new record pairs you want to classify as matches or non-matches

`feature_col`: Feature vector column identifier

- What it is: String name of the column in df that contains the feature vectors
- Expected content: Each row should contain an array/list of numeric similarity scores
- Consistency requirement: Feature vectors must have same structure as those used during training
- Purpose: Tells the model which column contains the input data for making predictions
- Common names: 'fvs', 'feature_vectors', 'features'

`output_col`: Prediction results column name

- What it is: String that will become the name of the new column containing predictions
- Output format: Column will contain numeric values (1.0 = match, 0.0 = non-match)
- Purpose: Allows you to customize the output column name to fit your workflow
- Additional output: Some models may also provide confidence scores in a separate column

**What it returns:**

```python
# Your input data with predictions added
result = pd.DataFrame({
    'id1': [20, 21, 22, 23],
    'id2': [3, 3, 4, 4],
    'feature_vectors': [
        [0.85, 0.9, 0.92],    # High similarity
        [0.2, 0.15, 0.1],     # Low similarity
        [0.75, 0.8, 0.85],    # Medium-high similarity
        [0.4, 0.45, 0.5]      # Medium similarity
    ],
    'predictions': [1.0, 0.0, 1.0, 0.0],        # Model predictions
    'confidence': [0.95, 0.98, 0.82, 0.65]      # How confident the model is
})
```

**Example Usage:**

```python
# Apply your trained model to new data
predictions = apply_matcher(
    model,                              # Your trained model
    new_feature_vectors,                # New data to predict on
    feature_col='feature_vectors',      # Column with features
    output_col='match_prediction'       # Where to put results
)

# Now you can see which pairs the model thinks are matches
matches = predictions[predictions['match_prediction'] == 1.0]
print(f"Found {len(matches)} potential matches")

# You can also look at confidence scores
high_confidence_matches = predictions[
    (predictions['match_prediction'] == 1.0) &
    (predictions['confidence'] > 0.8)
]
print(f"Found {len(high_confidence_matches)} high-confidence matches")
```

### label_data()

What it does: Implements active learning - intelligently selects which examples would be most helpful to label next, improving your model iteratively.

```python
def label_data(
    model_spec: Union[Dict, MLModel],     # Current model or model config
    mode: Literal["batch", "continuous"], # How to do the labeling
    labeler_spec: Union[Dict, Labeler],   # How to label examples
    fvs: pd.DataFrame,                    # Unlabeled data
    seeds: Optional[pd.DataFrame] = None  # Existing labeled data
) -> pd.DataFrame
```

**Parameter Explanations:**

`model_spec`: Model for active learning

- What it is: Either a trained MLModel object or a configuration dictionary for training a new model
- When to use trained model: If you already have a model and want to improve it with more labeled data
- When to use config dict: If you're starting fresh and want the function to train a model as it gathers labels
- Purpose: Provides the intelligence to select which examples would be most valuable to label
- How it works: Uses model uncertainty to identify examples where labels would provide most learning value

`mode`: Active learning strategy

- What it is: String specifying how to conduct the labeling process
- "batch" mode: Label a group of examples, then retrain the model, then repeat
  - Advantages: More efficient model training, better for large datasets
  - Process: Select N examples → label them → retrain model → repeat
- "continuous" mode: Label one example, update model, select next example
  - Advantages: More responsive, adapts quickly to new information
  - Process: Select 1 example → label it → update model → repeat
- Choice factors: Dataset size, computational resources, responsiveness needs

`labeler_spec`: Method for generating labels

- What it is: Either a Labeler object or configuration dictionary that defines how to label examples
- Purpose: Provides the mechanism to convert unlabeled examples into labeled training data
- Human labeler: Presents examples to human reviewer for manual classification
- Automatic labeler: Uses rules or thresholds to automatically assign labels
- Hybrid approaches: Combines automatic labeling for clear cases with human review for uncertain cases

`fvs`: Pool of unlabeled feature vectors

- What it is: A pandas DataFrame containing feature vectors for record pairs that need labels
- Data source: Typically the full output from `featurize()` before any labeling has been done
- Content requirements: Must contain feature vectors and any metadata needed for labeling decisions
- Purpose: Provides the candidate pool from which active learning will select examples for labeling

`seeds`: Initial labeled data (optional)

- What it is: A pandas DataFrame containing already-labeled examples to start the active learning process
- Data source: Could be output from `create_seeds()` or any previously labeled data
- Why optional: If None, active learning starts with no prior knowledge
- Why helpful: Provides initial model training data to bootstrap the active learning process
- Content requirements: Must contain feature vectors and labels in the same format as expected output

**Why you need it:** Instead of randomly labeling data, this focuses on examples where labeling will most improve your model's accuracy.

**How Active Learning Works:**

1. **Uncertainty Sampling**: Finds examples the model is most unsure about

   ```python
   # Model predictions with confidence
   uncertain_examples = [
       {'prediction': 0.51, 'confidence': 0.51},  # Very uncertain! (close to 0.5)
       {'prediction': 0.49, 'confidence': 0.49},  # Very uncertain! (close to 0.5)
       {'prediction': 0.95, 'confidence': 0.95},  # Very certain (high confidence)
       {'prediction': 0.05, 'confidence': 0.95}   # Very certain (high confidence)
   ]
   # Active learning would pick the first two for labeling
   ```

2. **Example Process:**

   ```python
   # Start with some seeds
   initial_labeled_data = create_seeds(feature_vectors, nseeds=100, ...)

   # Use active learning to improve iteratively
   more_labeled_data = label_data(
       model_spec=trained_model,
       mode="batch",
       labeler_spec=human_labeler,
       fvs=unlabeled_feature_vectors,
       seeds=initial_labeled_data
   )

   # Result: More labeled examples, focused on where they help most
   # The function automatically picks examples that will improve the model most
   ```

## Advanced Concepts Explained

### Understanding N-grams

In MadMatcher, the 3-gram tokenizer creates consecutive sequences of 3 characters. This helps catch partial matches and handle typos effectively.

```python
# 3-gram tokenizer example
text = "Smith"
character_3grams = ["Smi", "mit", "ith"]

text = "Smyth"
character_3grams = ["Smy", "myt", "yth"]

# The Jaccard similarity function compares these token sets:
# Intersection: {} (no common 3-grams)
# Union: {"Smi", "mit", "ith", "Smy", "myt", "yth"} (6 total)
# Jaccard = 0/6 = 0.0

# But "Smith" vs "Smiths" would have better similarity:
text1 = "Smith" -> ["Smi", "mit", "ith"]
text2 = "Smiths" -> ["Smi", "mit", "ith", "ths"]
# Intersection: {"Smi", "mit", "ith"} (3 common)
# Union: {"Smi", "mit", "ith", "ths"} (4 total)
# Jaccard = 3/4 = 0.75
```

### Understanding TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) in MadMatcher combines with tokenizers to identify important terms. Here's how it works:

```python
# Example: Using Stripped Whitespace tokenizer + TF-IDF similarity
companies = [
    "Smith Construction Company",
    "Jones Construction LLC",
    "Smith Manufacturing Inc",
    "ABC Company"
]

# Step 1: Stripped Whitespace tokenizer breaks into words
# "Smith Construction Company" -> ["smith", "construction", "company"]
# "Jones Construction LLC" -> ["jones", "construction", "llc"]

# Step 2: TF-IDF similarity function analyzes term importance
# "construction" appears in 2/4 companies - medium importance
# "smith" appears in 2/4 companies - medium importance
# "company" appears in 2/4 companies - medium importance
# "llc", "inc" appear in 1/4 companies each - high importance (rare)
# "manufacturing" appears in 1/4 companies - high importance (rare)

# Step 3: When comparing "Smith Construction Company" vs "Smith Construction LLC":
# - "smith" and "construction" get medium TF-IDF weight
# - "llc" gets high TF-IDF weight (distinctive term)
# - "company" gets medium TF-IDF weight
# Result: High similarity due to shared important terms
```

### Custom Feature Development

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

## Performance Optimization

### Computational Efficiency

```python
# Optimize feature computation for your specific similarity functions
# Since MadMatcher focuses on matching with specific tokenizers and similarity functions:

# Example: Optimize for TF-IDF computation with large candidate sets
def optimize_tfidf_matching(features, df_a, df_b, candidates_df):
    # TF-IDF similarity function benefits from batched processing
    batch_size = 5000  # Adjust based on memory available
    results = []

    for i in range(0, len(candidates_df), batch_size):
        batch = candidates_df.iloc[i:i+batch_size]
        batch_results = featurize(features, df_a, df_b, batch)
        results.append(batch_results)

    return pd.concat(results, ignore_index=True)

# Example: Optimize when using multiple similarity functions
def balance_similarity_functions(df_a, df_b, cols):
    # Some combinations work better than others:
    # - TF-IDF + Stripped Whitespace: Good for text with varying lengths
    # - Jaccard + 3-gram: Good for short text with potential typos
    # - Cosine + Numeric: Good for numerical comparisons

    fast_features = create_features(
        df_a, df_b, cols, cols,
        sim_functions=[jaccard_similarity, overlap_coefficient],  # Faster
        tokenizers=[stripped_whitespace_tokenizer]  # Simpler
    )

    return fast_features
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
   print(f"Feature vector shape: {feature_vectors['fvs'].iloc[0].shape}")
   ```

### Troubleshooting Common Issues

1. **Poor Match Quality**

   ```python
   # Check feature distributions for your specific similarity functions
   import matplotlib.pyplot as plt

   feature_vectors['avg_similarity'] = feature_vectors['fvs'].apply(lambda x: np.mean(x))
   plt.hist(feature_vectors['avg_similarity'], bins=50)
   plt.title('Distribution of Similarity Scores')
   plt.show()

   # Diagnose specific similarity functions:
   # - TF-IDF scores usually range 0.0-1.0, should have good spread
   # - Jaccard scores are 0.0-1.0, sensitive to tokenizer choice
   # - Cosine similarity typically 0.0-1.0, good for normalized comparisons

   # If most TF-IDF scores are low, try Stripped Whitespace tokenizer
   # If Jaccard scores are all very low, try 3-gram tokenizer for character-level similarity
   ```

2. **Memory Issues**

   ```python
   # Monitor memory usage during MadMatcher operations
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

   # Reduce candidate set size if needed
   # Since candidates come from external blocking systems,
   # work with blocking system to reduce candidate volume
   smaller_candidates = candidates_df.head(10000)
   feature_vectors = featurize(features, df_a, df_b, smaller_candidates)
   ```

3. **Slow Performance**

   ```python
   # Time your MadMatcher operations
   import time

   start_time = time.time()
   features = create_features(df_a, df_b, cols_a, cols_b)
   print(f"Feature creation took {time.time() - start_time:.2f} seconds")

   # Optimize by choosing faster similarity functions:
   # Fastest: Jaccard, Overlap Coefficient
   # Medium: Cosine, SIF
   # Slowest: TF-IDF (but often most accurate)

   # Use fewer similarity functions if speed is critical
   fast_features = create_features(
       df_a, df_b, cols_a, cols_b,
       sim_functions=[jaccard_similarity],  # Just one function
       tokenizers=[stripped_whitespace_tokenizer]  # Simplest tokenizer
   )
   ```
