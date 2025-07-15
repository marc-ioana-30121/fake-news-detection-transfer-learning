# Fake News Detection with Transfer Learning and Data Augmentation

This project explores the integration of transfer learning techniques with data augmentation for fake news detection.

## Project Structure

```
fake_news_detection/
├── data/
│   ├── raw/              # Original datasets (LIAR, FakeNewsNet)
│   └── processed/        # Preprocessed data
├── models/               # Saved models
├── experiments/          # Experiment configurations
├── results/              # Results and plots
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architectures
│   ├── training/        # Training scripts
│   ├── evaluation/      # Evaluation scripts
│   └── augmentation/    # Data augmentation methods
├── config.yaml          # Configuration file
└── requirements.txt     # Python dependencies
```

## Setup

1. Activate virtual environment:
   ```bash
   source fake_news_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download language models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Place your dataset zip files in the project root
2. Run the data exploration script
3. Start experimenting with models

## Research Questions

- What are the most efficient ways to integrate data augmentation techniques with transfer learning?
- How does the order and strategy of combining augmentation and transfer learning influence model performance?

## Datasets

- LIAR: A benchmark dataset for fake news detection
- FakeNewsNet: A comprehensive dataset of fake and real news

## Contact

Your project for exploring transfer learning in textual data augmentation.
