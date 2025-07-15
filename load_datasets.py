#!/usr/bin/env python3
"""
Specific Data Loader for LIAR and FakeNewsNet datasets
Based on the actual structure we found in your data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_liar_dataset():
    """Load LIAR dataset with proper column names"""
    print("📋 Loading LIAR Dataset...")
    
    # LIAR dataset column names (based on the format)
    liar_columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
        'state_info', 'party_affiliation', 'barely_true_counts', 
        'false_counts', 'half_true_counts', 'mostly_true_counts', 
        'pants_on_fire_counts', 'context'
    ]
    
    datasets = {}
    
    # Load each split
    for split in ['train', 'valid', 'test']:
        file_path = f"data/raw/liar_dataset/{split}.tsv"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep='\t', header=None, names=liar_columns)
                
                # Convert to binary classification
                df['binary_label'] = df['label'].apply(lambda x: 1 if x in ['false', 'pants-fire', 'barely-true'] else 0)
                df['binary_label_text'] = df['binary_label'].map({0: 'real', 1: 'fake'})
                
                datasets[split] = df
                print(f"✅ Loaded {split}: {len(df)} samples")
                print(f"   Label distribution: {df['binary_label'].value_counts().to_dict()}")
                
            except Exception as e:
                print(f"❌ Error loading {split}: {e}")
    
    return datasets

def load_fakenews_dataset():
    """Load FakeNewsNet dataset"""
    print("\n📋 Loading FakeNewsNet Dataset...")
    
    # Define the files we found
    files_info = {
        'buzzfeed_fake': 'data/raw/FakeNewsNet/BuzzFeed_fake_news_content.csv',
        'buzzfeed_real': 'data/raw/FakeNewsNet/BuzzFeed_real_news_content.csv',
        'politifact_fake': 'data/raw/FakeNewsNet/PolitiFact_fake_news_content.csv',
        'politifact_real': 'data/raw/FakeNewsNet/PolitiFact_real_news_content.csv'
    }
    
    all_data = []
    
    for source_type, file_path in files_info.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Add labels based on filename
                df['source_type'] = source_type
                df['binary_label'] = 1 if 'fake' in source_type else 0
                df['binary_label_text'] = 'fake' if 'fake' in source_type else 'real'
                df['source'] = 'buzzfeed' if 'buzzfeed' in source_type else 'politifact'
                
                all_data.append(df)
                print(f"✅ Loaded {source_type}: {len(df)} samples")
                
            except Exception as e:
                print(f"❌ Error loading {source_type}: {e}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Split into train/val/test (80/10/10)
        np.random.seed(42)
        shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        n_total = len(shuffled_df)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        datasets = {
            'train': shuffled_df[:n_train],
            'valid': shuffled_df[n_train:n_train+n_val],
            'test': shuffled_df[n_train+n_val:]
        }
        
        print(f"\n📊 FakeNewsNet Split Summary:")
        for split, df in datasets.items():
            print(f"   {split}: {len(df)} samples")
            print(f"   Label distribution: {df['binary_label'].value_counts().to_dict()}")
        
        return datasets
    
    return {}

def analyze_and_visualize_datasets(liar_data, fakenews_data):
    """Create comprehensive analysis and visualizations"""
    print("\n📈 Creating Analysis and Visualizations...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Analyze LIAR dataset
    if liar_data:
        print("\n🔍 LIAR Dataset Analysis:")
        
        # Combine all splits for analysis
        liar_combined = pd.concat([liar_data[split] for split in liar_data.keys()], 
                                keys=liar_data.keys()).reset_index(level=0)
        liar_combined.rename(columns={'level_0': 'split'}, inplace=True)
        
        # Basic statistics
        print(f"Total LIAR samples: {len(liar_combined)}")
        print(f"Original labels: {liar_combined['label'].value_counts().to_dict()}")
        print(f"Binary labels: {liar_combined['binary_label'].value_counts().to_dict()}")
        
        # Text length analysis
        liar_combined['statement_length'] = liar_combined['statement'].astype(str).apply(len)
        print(f"Average statement length: {liar_combined['statement_length'].mean():.1f} characters")
        
        # Save processed data
        for split, df in liar_data.items():
            df.to_csv(f'data/processed/liar_{split}.csv', index=False)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LIAR Dataset Analysis', fontsize=16)
        
        # 1. Label distribution
        liar_combined['binary_label_text'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
        axes[0,0].set_title('Binary Label Distribution')
        axes[0,0].set_xlabel('Label')
        axes[0,0].set_ylabel('Count')
        
        # 2. Original label distribution
        liar_combined['label'].value_counts().plot(kind='bar', ax=axes[0,1], rot=45)
        axes[0,1].set_title('Original Label Distribution')
        axes[0,1].set_xlabel('Label')
        
        # 3. Split distribution
        liar_combined['split'].value_counts().plot(kind='bar', ax=axes[0,2])
        axes[0,2].set_title('Split Distribution')
        axes[0,2].set_xlabel('Split')
        
        # 4. Statement length distribution
        liar_combined['statement_length'].hist(bins=50, ax=axes[1,0], alpha=0.7)
        axes[1,0].set_title('Statement Length Distribution')
        axes[1,0].set_xlabel('Length (characters)')
        axes[1,0].set_ylabel('Frequency')
        
        # 5. Length by label
        sns.boxplot(data=liar_combined, x='binary_label_text', y='statement_length', ax=axes[1,1])
        axes[1,1].set_title('Statement Length by Label')
        
        # 6. Party affiliation
        if 'party_affiliation' in liar_combined.columns:
            party_counts = liar_combined['party_affiliation'].value_counts().head(10)
            party_counts.plot(kind='bar', ax=axes[1,2], rot=45)
            axes[1,2].set_title('Top 10 Party Affiliations')
        
        plt.tight_layout()
        plt.savefig('results/liar_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Saved LIAR analysis to: results/liar_analysis.png")
        
    # Analyze FakeNewsNet dataset
    if fakenews_data:
        print("\n🔍 FakeNewsNet Dataset Analysis:")
        
        # Combine all splits
        fakenews_combined = pd.concat([fakenews_data[split] for split in fakenews_data.keys()], 
                                    keys=fakenews_data.keys()).reset_index(level=0)
        fakenews_combined.rename(columns={'level_0': 'split'}, inplace=True)
        
        # Basic statistics
        print(f"Total FakeNewsNet samples: {len(fakenews_combined)}")
        print(f"Label distribution: {fakenews_combined['binary_label'].value_counts().to_dict()}")
        print(f"Source distribution: {fakenews_combined['source'].value_counts().to_dict()}")
        
        # Text analysis
        fakenews_combined['title_length'] = fakenews_combined['title'].astype(str).apply(len)
        fakenews_combined['text_length'] = fakenews_combined['text'].astype(str).apply(len)
        print(f"Average title length: {fakenews_combined['title_length'].mean():.1f} characters")
        print(f"Average text length: {fakenews_combined['text_length'].mean():.1f} characters")
        
        # Save processed data
        for split, df in fakenews_data.items():
            df.to_csv(f'data/processed/fakenews_{split}.csv', index=False)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FakeNewsNet Dataset Analysis', fontsize=16)
        
        # 1. Label distribution
        fakenews_combined['binary_label_text'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
        axes[0,0].set_title('Binary Label Distribution')
        axes[0,0].set_xlabel('Label')
        
        # 2. Source distribution
        fakenews_combined['source'].value_counts().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Source Distribution')
        axes[0,1].set_xlabel('Source')
        
        # 3. Split distribution
        fakenews_combined['split'].value_counts().plot(kind='bar', ax=axes[0,2])
        axes[0,2].set_title('Split Distribution')
        axes[0,2].set_xlabel('Split')
        
        # 4. Title length distribution
        fakenews_combined['title_length'].hist(bins=50, ax=axes[1,0], alpha=0.7)
        axes[1,0].set_title('Title Length Distribution')
        axes[1,0].set_xlabel('Length (characters)')
        
        # 5. Text length by label
        sns.boxplot(data=fakenews_combined, x='binary_label_text', y='text_length', ax=axes[1,1])
        axes[1,1].set_title('Text Length by Label')
        
        # 6. Cross-tabulation
        cross_tab = pd.crosstab(fakenews_combined['source'], fakenews_combined['binary_label_text'])
        cross_tab.plot(kind='bar', ax=axes[1,2], stacked=True)
        axes[1,2].set_title('Source vs Label Distribution')
        axes[1,2].legend(title='Label')
        
        plt.tight_layout()
        plt.savefig('results/fakenews_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Saved FakeNewsNet analysis to: results/fakenews_analysis.png")
    
    plt.show()

def create_combined_dataset(liar_data, fakenews_data):
    """Create a combined dataset for training"""
    print("\n🔗 Creating Combined Dataset...")
    
    combined_datasets = {}
    
    for split in ['train', 'valid', 'test']:
        combined_data = []
        
        # Add LIAR data
        if split in liar_data:
            liar_split = liar_data[split][['statement', 'binary_label', 'binary_label_text']].copy()
            liar_split.rename(columns={'statement': 'text'}, inplace=True)
            liar_split['dataset'] = 'liar'
            combined_data.append(liar_split)
        
        # Add FakeNewsNet data
        if split in fakenews_data:
            fn_split = fakenews_data[split][['text', 'binary_label', 'binary_label_text']].copy()
            fn_split['dataset'] = 'fakenews'
            combined_data.append(fn_split)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_datasets[split] = combined_df
            
            # Save combined dataset
            combined_df.to_csv(f'data/processed/combined_{split}.csv', index=False)
            print(f"✅ Combined {split}: {len(combined_df)} samples")
            print(f"   Label distribution: {combined_df['binary_label'].value_counts().to_dict()}")
    
    return combined_datasets

def main():
    """Main function to load and analyze all datasets"""
    print("🚀 Comprehensive Dataset Loading and Analysis")
    print("=" * 60)
    
    # Load datasets
    liar_data = load_liar_dataset()
    fakenews_data = load_fakenews_dataset()
    
    if not liar_data and not fakenews_data:
        print("❌ No datasets loaded successfully!")
        return
    
    # Analyze and visualize
    analyze_and_visualize_datasets(liar_data, fakenews_data)
    
    # Create combined dataset
    combined_data = create_combined_dataset(liar_data, fakenews_data)
    
    print("\n🎉 Dataset Loading and Analysis Complete!")
    print("=" * 60)
    print("\n📁 Files created:")
    print("   📊 Individual datasets in data/processed/")
    print("   📊 Combined datasets in data/processed/")
    print("   📈 Analysis plots in results/")
    
    print("\n📈 Dataset Summary:")
    if liar_data:
        total_liar = sum(len(df) for df in liar_data.values())
        print(f"   LIAR Dataset: {total_liar} total samples")
    
    if fakenews_data:
        total_fn = sum(len(df) for df in fakenews_data.values())
        print(f"   FakeNewsNet: {total_fn} total samples")
    
    if combined_data:
        total_combined = sum(len(df) for df in combined_data.values())
        print(f"   Combined Dataset: {total_combined} total samples")
    
    print("\n🚀 Ready for model training!")
    print("Next steps:")
    print("1. Review the analysis plots in results/")
    print("2. Check processed data in data/processed/")
    print("3. Start building your baseline model")

if __name__ == "__main__":
    main()
