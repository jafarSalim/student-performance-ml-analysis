# Student Performance Machine Learning Analysis

A Python machine-learning project for analyzing and predicting student pass/fail performance using the UCI Student Performance dataset.

The project compares custom implementations of Naive Bayes classifiers, Principal Component Analysis (PCA), and semi-supervised learning methods. It also exports numerical results and a visual comparison dashboard.

## Features

- Loads the real `student-mat.csv` dataset by default.
- Supports `student-por.csv` as an alternative dataset.
- Converts the final grade (`G3`) into a binary pass/fail target using `G3 >= 10`.
- Evaluates three Naive Bayes variants:
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
- Evaluates three PCA implementations:
  - Standard PCA
  - SVD-based PCA
  - Incremental PCA
- Evaluates three semi-supervised approaches:
  - Self-Training Naive Bayes
  - Co-Training Naive Bayes
  - Label Propagation Naive Bayes
- Uses a stratified 70/30 train-test split.
- Fits feature scaling only on the training set to avoid test-set leakage.
- Uses stratified 5-fold cross-validation for the custom Naive Bayes models.
- Automatically exports result tables and a dashboard image.

## Repository Files

| File | Description |
|---|---|
| `student_performance_ml.py` | Main Python program |
| `student-mat.csv` | Mathematics student dataset |
| `student-por.csv` | Portuguese language student dataset |
| `comprehensive_ml_results.csv` | Detailed algorithm metrics |
| `ml_methods_comparison.csv` | Method comparison summary |
| `dataset_and_best_results.csv` | Dataset statistics and best-performing methods |
| `ml_analysis_dashboard.png` | Generated visualization dashboard |
| `results_summary.txt` | Console output from a verified run on `student-mat.csv` |
| `dataset_source.txt` | Dataset source information |
| `requirements.txt` | Python dependencies |
| `LICENSE` | Source-code license |

## Installation

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Run

Keep `student_performance_ml.py` and the CSV files in the same directory, then run:

```bash
python student_performance_ml.py
```

The program uses `student-mat.csv` first. If it is unavailable, it tries `student-por.csv`. If neither file exists, the program can generate a synthetic demonstration dataset.

## Verified Results

The included results were generated from the real `student-mat.csv` dataset with 395 records and 30 preprocessed input features.

Key results from the included run:

| Method | Test Accuracy |
|---|---:|
| Gaussian Naive Bayes | 0.6975 |
| Multinomial Naive Bayes | 0.6807 |
| Bernoulli Naive Bayes | 0.6975 |
| Standard PCA + Gaussian NB | 0.6639 |
| SVD PCA + Gaussian NB | 0.6639 |
| Incremental PCA + Gaussian NB | 0.6639 |
| Self-Training NB | 0.6387 |
| Co-Training NB | 0.6387 |
| Label Propagation NB | 0.6050 |

Naive Bayes 5-fold cross-validation results:

- Gaussian NB: `0.7031 ± 0.0442`
- Multinomial NB: `0.7101 ± 0.0346`
- Bernoulli NB: `0.6668 ± 0.0425`

These values are provided for reproducibility and should not be interpreted as universal performance guarantees.

## Dataset

This project uses the **Student Performance** dataset from the UCI Machine Learning Repository. The two CSV files describe student achievement in Mathematics and Portuguese language courses and contain demographic, social, school-related, and grade attributes.

See `dataset_source.txt` for the dataset page.

## Notes

The original project stored the Python program inside a `.txt` file. This repository converts it into a normal `.py` module and fixes the dataset filename so that the real `student-mat.csv` file is actually loaded. The evaluation code was also adjusted so that scaling is fitted only on the training set and custom classifiers receive real stratified cross-validation scores.

## License

The source code in this repository is released under the MIT License. The included dataset remains subject to the terms of its original provider.
