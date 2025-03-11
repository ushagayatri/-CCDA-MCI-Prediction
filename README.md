# -CCDA-MCI-Prediction

CCDA-MCI-Prediction/
│── data/
│   ├── raw/                  # Raw audio/text datasets
│   ├── processed/             # Processed feature datasets
│   ├── external/              # External knowledge (ConceptNet embeddings)
│
│── notebooks/
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_Feature_Extraction.ipynb
│   ├── 3_Knowledge_Augmentation.ipynb
│   ├── 4_Model_Training.ipynb
│   ├── 5_Model_Evaluation.ipynb
│
│── src/
│   ├── data_preprocessing.py   # Speech-to-text, linguistic feature extraction
│   ├── feature_extraction.py   # NLP-based text features
│   ├── knowledge_augmentation.py   # ConceptNet integration
│   ├── ccda_model.py           # Attention-based transformer model
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Model evaluation metrics
│
│── requirements.txt            # Dependencies
│── README.md                   # Repository documentation
│── config.yaml                 # Configuration settings
│── train.sh                    # Shell script for training
│── evaluate.sh                  # Shell script for evaluation
