# Little Book on Deep Learning – Chapter Splits

This repo splits `little_book_on_deep_learning.pdf` into per-chapter folders, each with:
- `chapter.pdf` – the page subset for that chapter
- `chapter.txt` – extracted text (plain, with page separators)

## Structure
- `split_chapters.py` – splitting/extraction script
- `chapters/NN_slug/` – output folders (already generated)
- `.venv/` – local virtual environment with required deps

## Chapter ranges (1-based, inclusive)
1. Foreword: 8–9  
2. Machine Learning: 11–19  
3. Efficient Computation: 20–24  
4. Training: 25–57  
5. Model Components: 58–97  
6. Architectures: 98–116  
7. Prediction: 117–137  
8. Synthesis: 138–145  
9. The Compute Schism: 146–157  
10. The missing bits: 158–163  
11. Bibliography: 164–175  
12. Index: 176–185  

## Usage
Assumes macOS with Python 3 available.

```bash
# activate venv (already created and populated)
source .venv/bin/activate

# regenerate everything (outputs to ./chapters)
python split_chapters.py

# generate specific chapters by slug
python split_chapters.py --only machine_learning training
```

Outputs go under `chapters/NN_slug/` with `chapter.pdf` and `chapter.txt`.

## Notes
- Text extraction uses pdfplumber with character deduping to reduce repeated letters.
- If you need markdown instead of plain text, or adjusted ranges/naming, update `CHAPTERS` in `split_chapters.py`.
