# PRECOG Recruitment Tasks 2026 - Natural Language Processing Track

## Overview

This repository contains completed work for the PRECOG recruitment tasks, specifically addressing the Natural Language Processing (NLP) theme.

## Research Problem Statement

**Title:** The Ghost in the Machine - Distinguishing Human from AI-Generated Text

**Core Question:** "Le style, c'est l'homme même" (The style is the man himself) - Can computational methods detect authorship when the author is an algorithm?

**Research Objectives:**

This work addresses the fundamental challenge of distinguishing between human-authored and AI-generated text through computational stylometry and machine learning. The investigation encompasses:

1. **Dataset Curation:** Constructing paired datasets of human-authored (Class 1) and AI-generated (Class 2) texts from comparable source materials
2. **Feature Engineering:** Extracting quantitative linguistic markers that characterize authorship patterns
3. **Multi-Tiered Classification:** Developing and comparing statistical, semantic, and hybrid detection models
4. **Interpretability Analysis:** Understanding which linguistic features most strongly differentiate human from machine-generated text
5. **Topic Modeling:** Examining thematic and stylistic divergences between human and AI writing

## Project Structure

```
Precog_task/
├── notebooks/           # Jupyter notebooks for each task
│   ├── task_zero.ipynb         # Dataset preparation and cleaning
│   ├── task_one.ipynb          # Feature extraction and analysis
│   ├── task_two_tierA.ipynb    # XGBoost classification model
│   ├── task_two_tierB.ipynb    # BERT-based classification
│   ├── task_two_tierC.ipynb    # Custom neural architecture
│   ├── task_three.ipynb        # Topic modeling analysis
│   └── task_four.ipynb         # Comprehensive evaluation
├── novels/              # Source texts (Project Gutenberg)
├── output/              # Generated datasets and results
│   ├── class1/         # Human-authored text samples
│   ├── class2/         # AI-generated text samples
│   ├── topics_bert/    # BERT-based topic extractions
│   └── topics_gemini/  # LLM-based topic extractions
└── requirements.txt     # Python dependencies
```

## Methodology

### Task 0: The Library of Babel - Dataset Construction

**Objective:** Construct Class 1 dataset comprising human-authored texts from canonical literature, establishing a baseline for stylistic comparison.

**Rationale:** Classic literature provides high-quality, professionally edited prose with consistent authorial voice, making it ideal for extracting stable linguistic patterns.

**Implementation:**

**Class 1 (Human-Authored):**

- Source materials: Five novels from Project Gutenberg
  - Joseph Conrad: _Heart of Darkness_, _Lord Jim_, _Typhoon_
  - Franz Kafka: _Metamorphosis_, _The Trial_
- Automated preprocessing pipeline:
  - Gutenberg header/footer removal
  - Metadata and license text stripping
  - Chapter heading normalization via regex patterns
  - Sentence segmentation with quality filters (minimum 6 words)
  - Stratified random sampling (100 sentences per novel, seed=42)

**Class 2 (AI-Generated):**

- Generated using Gemini API based on Class 1 source materials
- 500 synthetic paragraphs per novel matching thematic content
- Deliberately mimics author style to create challenging classification task

**Outputs:**

- Cleaned text files: `output/class1/*_cleaned.txt`
- Sampled sentences: `output/class1/*_samples.json`
- AI-generated corpus: `output/class2/*_generic.jsonl`
- Processing summary: `output/class1/task_zero_summary.json`

**Notebook:** [notebooks/task_zero.ipynb](notebooks/task_zero.ipynb)

### Task 1: The Fingerprint - Proving Mathematical Distinction

**Objective:** Demonstrate that Class 1 (human) and Class 2 (AI) are mathematically distinct through quantitative linguistic analysis.

**Hypothesis:** Human and AI-generated texts exhibit statistically significant differences across multiple linguistic dimensions.

**Feature Engineering:**

1. **Lexical Richness Metrics:**
   - Type-Token Ratio (TTR): Vocabulary diversity normalized by text length
   - Hapax Legomena: Words appearing exactly once (marker of lexical creativity)

2. **Syntactic Complexity (spaCy-based):**
   - Dependency tree depth: Measure of nested clause structures
   - Average sentence length: Syntactic elaboration
   - POS distribution: Noun/verb/adjective/adverb ratios

3. **Punctuation Density (per 1000 words):**
   - Semicolons, em-dashes, exclamation marks
   - Indicators of stylistic choices and sentence rhythm

4. **Readability Indices:**
   - Flesch-Kincaid Grade Level: Quantifies text complexity

**Analysis Methods:**

- Comparative distributional analysis (violin plots, box plots)
- Statistical significance testing (t-tests, effect sizes)
- Correlation analysis between features

**Tools:** spaCy (en_core_web_sm), pandas, scipy

**Notebook:** [notebooks/task_one.ipynb](notebooks/task_one.ipynb)

### Task 2: The Multi-Tiered Detective - Classification Models

**Objective:** Build three distinct detectors to separate AI-generated text from human-written prose. Models must achieve meaningful accuracy and provide interpretable decision boundaries.

**Design Philosophy:** Progress from interpretable statistical models to complex neural architectures, trading transparency for potential performance gains.

**Tier A - The Statistician (XGBoost):**

- **Approach:** Gradient boosting on hand-crafted linguistic features from Task 1
- **Architecture:** XGBoost classifier (100 estimators, max depth 3)
- **Key Techniques:**
  - Novel-based stratified split (prevents data leakage between train/test)
  - Feature standardization (StandardScaler fitted on training data only)
  - Class imbalance handling (scale_pos_weight parameter)
  - Feature importance analysis via SHAP values
- **Strengths:** Interpretable, fast training, handles tabular data well
- **Notebook:** [notebooks/task_two_tierA.ipynb](notebooks/task_two_tierA.ipynb)

**Tier B - The Semanticist (BERT-based):**

- **Approach:** Fine-tuned transformer using contextual embeddings
- **Architecture:** Pre-trained BERT with classification head
- **Key Techniques:**
  - Transfer learning from bert-base-uncased
  - Sequence-level classification tokens [CLS]
  - Attention mechanism visualization
- **Strengths:** Captures semantic nuances, contextual understanding
- **Notebook:** [notebooks/task_two_tierB.ipynb](notebooks/task_two_tierB.ipynb)

**Tier C - The Transformer (Custom Architecture):**

- **Approach:** Domain-specific hybrid model or distilled architecture
- **Design:** Combines statistical features with learned representations
- **Strengths:** Optimized for this specific detection task
- **Notebook:** [notebooks/task_two_tierC.ipynb](notebooks/task_two_tierC.ipynb)

### Task 3: The Smoking Gun - Model Interpretability

**Objective:** Understand why models classify text as AI-generated by identifying salient linguistic patterns.

**Research Questions:**

- Which words or phrases most strongly indicate AI authorship?
- Does the model focus on content (semantics) or style (syntax)?
- Are there systematic patterns in model errors?

**Analysis Techniques:**

1. **Saliency Mapping:**
   - Word-level attribution using SHAP or LIME
   - Highlight spans with highest classification impact
   - Identify "imposter" phrases characteristic of AI generation

2. **Error Analysis:**
   - Examine false positives (human labeled as AI)
   - Examine false negatives (AI labeled as human)
   - Identify adversarial examples or edge cases

3. **Topic Modeling:**
   - BERT-based topic extraction across both classes
   - LLM-assisted (Gemini) thematic identification
   - Comparative topic distribution analysis

**Notebook:** [notebooks/task_three.ipynb](notebooks/task_three.ipynb)

### Task 4: The Turing Test - Adversarial Evaluation

**Objective:** Test model robustness through adversarial examples and evolutionary optimization.

**The Super-Imposter Challenge:**
Can a Genetic Algorithm (GA) evolve AI-generated paragraphs that fool the best detector?

**GA Workflow:**

1. **Initial Population:** Generate 10 "imposter" paragraphs using Gemini
2. **Fitness Function:** Probability of being classified as "Human" by best model
3. **Selection:** Tournament selection of highest-scoring imposters
4. **Mutation (LLM as Mutator):** Prompt Gemini to subtly modify paragraphs:
   - "Rewrite to change rhythm while keeping vocabulary"
   - "Introduce subtle grammatical inconsistencies"
5. **Iteration:** Run for 5-10 generations, track fitness evolution

**Personal Writing Test:**
Submit Statement of Purpose (SOP) or personal essay to determine if the model classifies the applicant's own writing as human or AI-generated. Reflection on results.

**Evaluation Metrics:**

- Model confidence scores across generations
- Linguistic feature drift during evolution
- Human evaluation of final "super-imposter" quality

**Notebook:** [notebooks/task_four.ipynb](notebooks/task_four.ipynb)

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment support

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Precog_task
```

2. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Configure Jupyter kernel:

```bash
python -m ipykernel install --user --name=precog_env
```

## Reproducibility

All experiments use fixed random seeds (seed=42) to ensure reproducibility. The notebook execution order is sequential from Task Zero through Task Four. Each task builds upon outputs from previous tasks.

### Running the Pipeline

Execute notebooks in order:

1. `task_zero.ipynb` - Generates cleaned Class 1 dataset
2. `task_one.ipynb` - Extracts linguistic features
3. `task_two_tier*.ipynb` - Trains and evaluates classification models
4. `task_three.ipynb` - Performs topic modeling
5. `task_four.ipynb` - Synthesizes results

## Experimental Results

Comprehensive results, statistical analyses, and visualizations are documented within each notebook. Performance metrics are evaluated under realistic conditions with novel-based stratification to prevent overfitting.

**Key Findings:**

**Task 0:** Successfully curated 500 human-authored sentences and 2,500 AI-generated paragraphs across five novels

**Task 1:** Identified statistically significant differences (p < 0.01) across multiple linguistic dimensions:

- Human text exhibits higher lexical diversity (TTR)
- AI text shows more uniform punctuation patterns
- Syntactic complexity varies significantly between classes

**Task 2:** Multi-tiered classification results:

- Tier A (XGBoost): Performance metrics on held-out test novels
- Tier B (BERT): Semantic classification accuracy
- Tier C (Custom): Hybrid model performance
- Comparative analysis with ROC-AUC curves, confusion matrices, and precision-recall trade-offs

**Task 3:** Interpretability analysis reveals:

- Feature importance rankings
- Salient n-grams and phrases characteristic of each class
- Topic distribution differences

**Task 4:** Adversarial robustness testing:

- GA convergence on "super-imposter" examples
- Model performance degradation under adversarial conditions
- Personal writing classification results

## Technical Details

**Programming Language:** Python 3.x

**Key Libraries:**

- Natural Language Processing: spaCy (en_core_web_sm), NLTK
- Machine Learning: scikit-learn, XGBoost
- Deep Learning: Transformers (Hugging Face), PyTorch/TensorFlow
- LLM Integration: Google Gemini API
- Data Analysis: pandas, numpy, scipy
- Visualization: matplotlib, seaborn

**Methodological Safeguards:**

- Fixed random seeds (42) for reproducibility
- Novel-based stratification prevents train-test leakage
- Feature standardization fitted only on training data
- Cross-validation within training set for hyperparameter tuning
- Held-out test novels never seen during training
