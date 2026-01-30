# Precog_task

## Project Status — Task Zero (Library of Babel)

Summary of work completed so far (Task Zero — prepare Class‑1 / human examples):

- **Notebook:** Implemented the cleaning and dataset-building pipeline in [notebooks/task_zero.ipynb](notebooks/task_zero.ipynb).
- **Cleaning functions added:** `read_text`, `remove_gutenberg_headers`, `basic_cleanup`, `remove_chapter_headings` (tightened to catch many variants), `clean_text`, `split_sentences`, and `build_class1_dataset`.
- **Outputs generated:** cleaned book texts and per-book samples saved under `output/task_zero` (files named `<book>_cleaned.txt`, `<book>_samples.json`) and a summary `task_zero_summary.json`.
- **Environment:** Development and notebook run inside a Python venv (created as `.venv`) registered as a Jupyter kernel (example kernel name used: Precog (venv)).

## How to reproduce / run the pipeline

1. Activate the venv (Windows PowerShell example):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Open the project in VS Code and select the corresponding Python interpreter / Jupyter kernel (use the `.venv\Scripts\python.exe`).

3. In the notebook `notebooks/task_zero.ipynb`, run the final cell that calls `build_class1_dataset(...)` (or run the cell labeled `#### BUILD DATASET`). Example parameters are already set in the notebook:

```python
base = r'c:\Users\eisas\OneDrive\Desktop\PROJECTS\Precog_task\novels'
out = r'c:\Users\eisas\OneDrive\Desktop\PROJECTS\Precog_task\output\task_zero'
summary = build_class1_dataset(base, out, sample_size=100)
print('Task-zero cleaning complete. Summary:', summary)
```

4. After running, check outputs in `output/task_zero`.

## What the cleaning does (concise)

- **Header/footer removal:** Strips Project Gutenberg header/footer when present.
- **Boilerplate removal:** Removes obvious license lines, short all-caps headers and URLs.
- **Chapter-heading removal:** Aggressive regex heuristics remove common chapter/book/part headings (case-insensitive, title-case 'Chapter Two' variants, short all-caps headings). This was iterated to remove leftover markers like "Chapter", "Chapter Two", "CHAPTER I.".
- **Sentence extraction:** Heuristic splitter that keeps sentences of at least a configurable word length (default 6) and drops enumerated list items.
- **Sampling & export:** Random, seeded sampling (seed=42) to produce per-book sample JSON and a project summary.

## Files of interest

- Notebook: [notebooks/task_zero.ipynb](notebooks/task_zero.ipynb)
- Novels folder: `novels/` (source `.txt` files)
- Output folder: `output/task_zero/`

## Next recommended steps

- Re-run the final cell to regenerate cleaned files after any tuning.
- Inspect `output/task_zero/*_cleaned.txt` for any remaining formatting artifacts and report any variants to further tune `remove_chapter_headings`.
- (Optional) Replace the heuristic sentence splitter with `nltk`/`spacy` for higher-quality splitting if needed.

---

If you'd like, I can run the pipeline now and show a diff of before/after for one or two books, or further tighten the chapter-heading patterns based on specific residual examples you provide.
