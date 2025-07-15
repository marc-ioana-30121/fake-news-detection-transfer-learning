#!/usr/bin/env python3
"""
Create project structure for fake news detection
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    print("📁 Creating project structure...")
    
    # Create main directories
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "experiments",
        "results",
        "notebooks",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/augmentation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
        
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/augmentation/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def create_config_file():
    """Create configuration file"""
    config_content = """# Configuration for Fake News Detection Project

# Model Configuration
model:
  name: "bert-base-uncased"
  max_length: 512
  num_labels: 2

# Training Configuration
training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500
  weight_decay: 0.01
  gradient_accumulation_steps: 1

# Data Configuration
data:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42

# Augmentation Configuration
augmentation:
  methods: ["back_translation", "synonym_replacement", "paraphrasing"]
  augmentation_ratio: 0.5
  max_augmented_samples: 1000
  
# Paths
paths:
  data_dir: "./data"
  raw_data_dir: "./data/raw"
  processed_data_dir: "./data/processed"
  model_dir: "./models"
  results_dir: "./results"
  
# Logging
logging:
  use_wandb: false
  project_name: "fake-news-detection"
  log_dir: "./logs"
"""
    
    with open("config.yaml", "w") as f:
        f.write(config_content)
    print("✅ Created: config.yaml")

def create_requirements_file():
    """Create requirements.txt file with exact versions"""
    requirements = """torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers==4.36.2
datasets>=2.16.0
tokenizers>=0.15.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
nltk>=3.8.0
spacy>=3.7.0
textattack>=0.3.10
tqdm>=4.66.0
jupyter>=1.0.0
ipywidgets>=8.1.0
wandb>=0.16.0
pyyaml>=6.0.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Created: requirements.txt")

def create_readme():
    """Create README file"""
    readme_content = """# Fake News Detection with Transfer Learning and Data Augmentation

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
"""

    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✅ Created: README.md")

def main():
    print("🚀 Setting up Fake News Detection Project Structure\n")
    
    create_project_structure()
    print()
    create_config_file()
    create_requirements_file()
    create_readme()
    
    print("\n🎉 Project structure created successfully!")
    print("\nNext steps:")
    print("1. Place your LIAR and FakeNewsNet zip files in this directory")
    print("2. Run the data exploration script")
    print("3. Start with Jupyter notebook: jupyter notebook")
    
    # Show created structure
    print("\n📁 Created directory structure:")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        if level < 3:  # Don't go too deep
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                if not file.startswith('.') and level < 2:
                    print(f"{subindent}{file}")

if __name__ == "__main__":
    main()
