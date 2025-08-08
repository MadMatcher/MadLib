## Similarity Functions and Tokenizers in MadLib

In MadLib, we create a set of features, then use them to convert each tuple pair into a feature vector. Roughly speaking, a feature is a way to compare two values and determine how similar they are. Think of it as answering questions like:

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
   Available similarity functions in MadLib (as of July 31, 2025):
   - TF-IDF: Term frequency-inverse document frequency similarity
   - Jaccard: Set-based similarity using intersection over union
   - SIF: Smooth inverse frequency similarity
   - Overlap Coefficient: Set overlap measure
   - Cosine: Vector space similarity between token vectors

### Understanding Your Similiarity Functions

When using MadLib, it's crucial to understand how your chosen similarity functions work:

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
