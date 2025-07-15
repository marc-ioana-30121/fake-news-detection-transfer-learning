"""
Getting Started Script - Complete Setup for Fake News Detection Project
Run this script step by step to set up your environment and explore your datasets
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_project_structure():
    """Create project directory structure"""
    print("📁 Setting up project structure...")
    
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
    
    print("✅ Project structure created")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """torch>=2.1.0
transformers==4.36.2
datasets==2.16.1
tokenizers==0.15.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
textattack==0.3.10
nltk==3.8.1
spacy==3.7.2
matplotlib==3.8.2
seaborn==0.13.0
wandb==0.16.1
tqdm==4.66.1
jupyter==1.0.0
ipywidgets==8.1.1
accelerate==0.25.0
evaluate==0.4.1"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Requirements.txt created")

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    # Install main packages
    if not run_command("pip install -r requirements.txt", "Installing main packages"):
        return False
    
    # Download NLTK data
    nltk_downloads = [
        "import nltk; nltk.download('punkt')",
        "import nltk; nltk.download('stopwords')",
        "import nltk; nltk.download('wordnet')",
        "import nltk; nltk.download('omw-1.4')"
    ]
    
    for download in nltk_downloads:
        run_command(f'python -c "{download}"', f"Downloading NLTK data")
    
    # Download SpaCy model
    run_command("python -m spacy download en_core_web_sm", "Downloading SpaCy model")
    
    return True

def create_test_script():
    """Create installation test script"""
    test_script = '''
import torch
import transformers
import textattack
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import nltk
import spacy

def test_installation():
    print("🔍 Testing installation...")
    
    # Test PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
    
    # Test Transformers
    print(f"✅ Transformers version: {transformers.__version__}")
    
    # Test BERT model loading
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        print("✅ BERT model loaded successfully")
    except Exception as e:
        print(f"❌ BERT loading failed: {e}")
    
    # Test TextAttack
    print(f"✅ TextAttack version: {textattack.__version__}")
    
    # Test SpaCy
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence.")
        print("✅ SpaCy model loaded successfully")
    except Exception as e:
        print(f"❌ SpaCy loading failed: {e}")
    
    print("\\n🎉 Installation test complete!")

if __name__ == "__main__":
    test_installation()
'''
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    print("✅ Test script created")

def create_config_file():
    """Create configuration file"""
    config = """# Model Configuration
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
  
# Paths
paths:
  data_dir: "./data"
  model_dir: "./models"
  results_dir: "./results"
"""
    
    with open("config.yaml", "w") as f:
        f.write(config)
    print("✅ Configuration file created")

def setup_wandb():
    """Setup Weights & Biases"""
    print("🔧 Setting up Weights & Biases...")
    print("Please run 'wandb login' manually and follow the instructions")
    print("You can get your API key from: https://wandb.ai/authorize")

def main():
    """Main setup function"""
    print("🚀 Starting Fake News Detection Project Setup...\n")
    
    # Step 1: Create project structure
    setup_project_structure()
    
    # Step 2: Create requirements file
    create_requirements_file()
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check the errors above.")
        return
    
    # Step 4: Create test script
    create_test_script()
    
    # Step 5: Create config file
    create_config_file()
    
    # Step 6: Test installation
    print("🧪 Testing installation...")
    if run_command("python test_installation.py", "Running installation test"):
        print("✅ All installations successful!")
    else:
        print("❌ Some tests failed. Please check the installation.")
    
    # Step 7: Setup instructions
    print("\\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\\nNext steps:")
    print("1. Place your LIAR dataset zip file in the project root")
    print("2. Place your FakeNewsNet dataset zip file in the project root")
    print("3. Update the paths in the data loader script")
    print("4. Run: python -c 'from data_loader_explorer import DatasetExplorer; explorer = DatasetExplorer(); explorer.load_and_explore_all(\"your_liar.zip\", \"your_fakenews.zip\")'")
    print("5. Start Jupyter notebook: jupyter notebook")
    print("6. Setup W&B: wandb login")
    print("\\nProject structure:")
    print("📁 Your project is ready in the current directory")
    print("📊 Data will be processed in ./data/processed/")
    print("🤖 Models will be saved in ./models/")
    print("📈 Results will be saved in ./results/")
    
    # Show current directory structure
    print("\\nCurrent directory structure:")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            if not file.startswith('.'):
                print(f"{subindent}{file}")

if __name__ == "__main__":
    main()
