# University NLP Project

A Natural Language Processing coursework project written in Python, with a Persian right-to-left web interface and an interactive command-line program. It groups three practical exercises in one interactive program: text preprocessing, document classification with multinomial Naive Bayes, and information retrieval with TF-IDF and vector-space ranking.

## UI preview

![Vajeh NLP Studio demo: text processing, word frequency, classification, dark mode, and document search](docs/assets/ui-demo.gif)

The demo uses the running local UI and real processing results. The interface is in Persian; the NLP algorithms process English text.

Start the web interface after installing the Python requirements below:

```bash
python ui/server.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). See the [UI guide](ui/README.md) for limits and tests.

## What it includes

- Tokenization for common text patterns such as emails, URLs, dates, numbers, hashtags, mentions, words, and punctuation.
- Lowercase normalization, token-frequency reporting, Porter stemming, and stop-word removal.
- Multinomial Naive Bayes classification over the included training and test documents.
- TF-IDF document indexing and ranked retrieval for interactive queries.
- Vector-space retrieval using TF-IDF weighted vectors and cosine similarity.
- JSON result files for submitted TF-IDF and vector-space queries.

## Repository layout

```text
.
├── Nlp-project.py
└── Dataset/
    ├── TextProcessing/                 # Sample text used by preprocessing tasks
    ├── Spelling Dataset/               # Supporting spelling-correction material
    ├── Classification-Train And Test/  # Training and test documents by class
    └── IR/
        └── IR Documents to Index/      # Corpus used by retrieval exercises
```

The project expects the `Dataset` directory to remain next to `Nlp-project.py`.

## Requirements

- Python 3
- NumPy
- NLTK

Install the Python packages with:

```bash
python -m pip install numpy nltk
```

`PorterStemmer` is used directly and does not require downloading an NLTK corpus.

## Run

From the repository root:

```bash
python Nlp-project.py
```

The main menu exposes these sections:

```text
1. Text processing
2. Text classification
3. Information retrieval
4. Exit
```

### Text processing

The text-processing menu runs against `Dataset/TextProcessing/61085-0.txt` and writes its output into the repository root.

| Option | Result file |
| --- | --- |
| Tokenize | `Tokenize_result.txt` |
| Lowercase folding | `Lowercase_result.txt` |
| Token frequency | `Token_frequncy_result.txt` |
| Stemming | `stemmming_result.txt` |
| Stop-word removal | `stop_word_removal_result.txt` |

### Text classification

The classifier reads the folders inside `Dataset/Classification-Train And Test`, builds a vocabulary from the available training documents, then evaluates a randomly selected test document. It reports the predicted class and the calculated probabilities.

The implementation uses additive smoothing and log-space scores to avoid underflow during probability multiplication.

### Information retrieval

Two retrieval approaches are available:

- **TF-IDF model:** builds an index for the IR corpus and ranks matching documents.
- **Vector-space model:** builds weighted document vectors and ranks results by cosine similarity.

Enter queries in the terminal. Type `exit` when prompted for a query to return to the previous menu. Query result files are written to the output directory created by the selected retrieval flow.

## Notes

- This is an educational implementation that favors readable, explicit algorithms over production-scale optimization.
- The included datasets can be large. Initial indexing time depends on the number of documents selected by the program.
- Dataset paths resolve relative to the Python script. Generated command-line output files use the current working directory.

## License

No license file is currently included in this repository. All rights remain with the repository owner unless a license is added.
