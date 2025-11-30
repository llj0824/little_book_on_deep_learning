# Learning Context: Little Book on Deep Learning

This repository serves as a structured workspace for learning from François Fleuret's [Little Book on Deep Learning](https://fleuret.org/public/lbdl.pdf). 

The content has been processed and organized to facilitate focused study and to provide granular context for AI assistants (like Gemini) to help explain concepts, summarize chapters, or answer specific questions.

## Repository Purpose

The primary goal is to break down the dense material of the book into manageable, machine-readable chunks. This allows for:
1.  **Context-Aware AI Assistance:** Giving the AI specific chapter text to ensure accurate answers based strictly on the book's content.
2.  **Structured Learning:** Tracking progress chapter by chapter.
3.  **Quick Reference:** Easily locating specific topics using the generated folder structure.

## Key Files & Structure

*   **`gemini_book_overview.md`**: A comprehensive global overview and roadmap of the book. It contains summaries, key concepts, and dependency maps for every chapter. Use this to orient yourself or prime the AI before diving into a specific section.
*   **`chapters/`**: The core content. The book has been split into individual folders (e.g., `01_machine_learning/`, `05_architectures/`). Each contains:
    *   `chapter.pdf`: **PRIMARY SOURCE.** The specific pages for that chapter. Please refer to this file for all figures, diagrams, and visual context. It is the source of truth for visual content.
    *   `chapter_gemini_copyedited.md`: The text content of the chapter, copy-edited for clarity.
    *   `chapter_notes.md`: Notes and summaries generated during our study sessions.
*   **`little_book_on_deep_learning.pdf`**: The original full source PDF.
*   **`split_chapters.py`**: The Python utility used to generate the `chapters/` directory.

## Usage

### 1. Getting an Overview
Start by reading `gemini_book_overview.md` or feeding it to your AI assistant to establish a shared understanding of the book's scope and notation conventions.

### 2. Deep Diving a Topic
When studying a specific concept (e.g., "How does a Transformer work?"):
1.  Navigate to the relevant folder (e.g., `chapters/05_architectures/`).
2.  Open the `chapter.pdf` for reading.
3.  If you have questions, provide the content of `chapter_gemini_copyedited.md` to your AI assistant as context.

### 3. Regenerating Content (Optional)
If you need to re-run the extraction process (e.g., to adjust page ranges):

```bash
# Activate the environment
source .venv/bin/activate

# Run the split script
python split_chapters.py
```