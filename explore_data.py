#!/usr/bin/env python3
"""
Simple Data Loader and Explorer for LIAR and FakeNewsNet datasets
"""

import pandas as pd
import numpy as np
import zipfile
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

def extract_and_explore_datasets():
    """Extract and explore your datasets"""
    print("🚀 Fake News Dataset Explorer")
    print("=" * 50)
    
    # Check for zip files in current directory
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    print(f"Found zip files: {zip_files}")
    
    if not zip_files:
        print("❌ No zip files found in current directory.")
        print("Please copy your LIAR and FakeNewsNet zip files here.")
        return
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Extract all zip files
    for zip_file in zip_files:
        print(f"\n📦 Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                extract_path = f"data/raw/{zip_file.replace('.zip', '')}"
                zip_ref.extractall(extract_path)
                print(f"✅ Extracted to {extract_path}")
                
                # List contents
                print("Contents:")
                for root, dirs, files in os.walk(extract_path):
                    level = root.replace(extract_path, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:10]:  # Show first 10 files
                        print(f"{subindent}{file}")
                    if len(files) > 10:
                        print(f"{subindent}... and {len(files) - 10} more files")
                        
        except Exception as e:
            print(f"❌ Error extracting {zip_file}: {e}")
    
    # Now explore the extracted data
    print("\n🔍 Exploring extracted data...")
    explore_extracted_data()

def explore_extracted_data():
    """Explore the extracted data files"""
    raw_dir = Path("data/raw")
    
    if not raw_dir.exists():
        print("❌ No raw data directory found")
        return
    
    # Find all CSV, TSV, JSON files
    data_files = []
    for ext in ['*.csv', '*.tsv', '*.txt', '*.json']:
        data_files.extend(raw_dir.rglob(ext))
    
    print(f"Found {len(data_files)} data files:")
    for file in data_files:
        print(f"  📄 {file}")
    
    # Try to load and analyze each file
    datasets = {}
    
    for file_path in data_files:
        print(f"\n📊 Analyzing {file_path.name}...")
        try:
            # Determine file type and load accordingly
            if file_path.suffix in ['.csv', '.tsv', '.txt']:
                # Try different separators
                for sep in ['\t', ',', '|']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, nrows=5)  # Read first 5 rows to test
                        if df.shape[1] > 1:  # If we got multiple columns, it worked
                            df_full = pd.read_csv(file_path, sep=sep)
                            print(f"✅ Loaded as {sep}-separated file: {df_full.shape}")
                            print(f"Columns: {list(df_full.columns)}")
                            
                            # Try to identify fake news data patterns
                            if identify_fake_news_dataset(df_full, file_path.name):
                                datasets[file_path.name] = df_full
                            break
                    except:
                        continue
                        
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded JSON with {len(data) if isinstance(data, list) else 1} items")
                
                # Convert to DataFrame if possible
                if isinstance(data, list) and len(data) > 0:
                    df = pd.json_normalize(data)
                    if identify_fake_news_dataset(df, file_path.name):
                        datasets[file_path.name] = df
                        
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
    
    # Analyze identified datasets
    if datasets:
        print(f"\n🎯 Successfully identified {len(datasets)} fake news datasets!")
        analyze_datasets(datasets)
    else:
        print("\n❌ No fake news datasets identified. Manual inspection needed.")
        print("Try running: python3 -c \"import pandas as pd; df = pd.read_csv('path_to_your_file', sep='\\t'); print(df.head())\"")

def identify_fake_news_dataset(df, filename):
    """Try to identify if this is a fake news dataset"""
    # Look for common fake news dataset patterns
    fake_news_indicators = [
        'label', 'fake', 'real', 'true', 'false', 'statement', 'claim', 
        'news', 'article', 'text', 'content', 'title', 'headline'
    ]
    
    columns_lower = [col.lower() for col in df.columns]
    
    # Check if we have potential fake news columns
    has_label = any('label' in col for col in columns_lower)
    has_text = any(indicator in col for col in columns_lower for indicator in ['text', 'statement', 'content', 'title', 'news'])
    
    if has_label and has_text:
        print(f"🎯 Identified as fake news dataset!")
        return True
    
    # Check for LIAR dataset pattern (specific format)
    if df.shape[1] >= 3 and 'liar' in filename.lower():
        print(f"🎯 Identified as LIAR dataset!")
        return True
    
    return False

def analyze_datasets(datasets):
    """Analyze the identified datasets"""
    print("\n📈 Dataset Analysis")
    print("=" * 50)
    
    for name, df in datasets.items():
        print(f"\n📊 Analyzing {name}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
        
        # Look for labels
        potential_label_cols = [col for col in df.columns if 'label' in col.lower()]
        if potential_label_cols:
            label_col = potential_label_cols[0]
            print(f"\nLabel distribution in '{label_col}':")
            print(df[label_col].value_counts())
        
        # Save processed version
        output_path = f"data/processed/{name.replace('.', '_')}_processed.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved processed data to: {output_path}")
        
        # Create simple visualization
        try:
            plt.figure(figsize=(10, 6))
            
            if potential_label_cols:
                # Plot label distribution
                plt.subplot(1, 2, 1)
                df[label_col].value_counts().plot(kind='bar')
                plt.title(f'Label Distribution - {name}')
                plt.xlabel('Labels')
                plt.ylabel('Count')
                
            # Plot text length if we can find text column
            text_cols = [col for col in df.columns if any(indicator in col.lower() for indicator in ['text', 'statement', 'content', 'title'])]
            if text_cols:
                text_col = text_cols[0]
                plt.subplot(1, 2, 2)
                text_lengths = df[text_col].astype(str).apply(len)
                plt.hist(text_lengths, bins=30, alpha=0.7)
                plt.title(f'Text Length Distribution - {name}')
                plt.xlabel('Text Length (characters)')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plot_path = f"results/{name.replace('.', '_')}_analysis.png"
            os.makedirs('results', exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved analysis plot to: {plot_path}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

def main():
    """Main function"""
    print("Welcome to the Fake News Dataset Explorer!")
    print("This script will help you extract and explore your datasets.")
    
    # Check if we're in the right directory
    if not os.path.exists('fake_news_env'):
        print("⚠️  You might not be in the project directory.")
        print("Make sure you're in the directory where you created the virtual environment.")
    
    # Extract and explore
    extract_and_explore_datasets()
    
    print("\n🎉 Data exploration complete!")
    print("\nNext steps:")
    print("1. Check the 'data/processed/' directory for your processed datasets")
    print("2. Check the 'results/' directory for analysis plots")
    print("3. Start Jupyter notebook to begin modeling: jupyter notebook")

if __name__ == "__main__":
    main()
