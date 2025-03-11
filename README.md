# -CCDA-MCI-Prediction

This repository is related to this paper:
Lokala, Usha, et al.(2024). Artificial intelligence-based detection of early cognitive impairment using language, speech, and demographic features: Model development and validation. BMC Medical Informatics and Decision Making, 2024.

Instructions to Run the Repository
pip install -r requirements.txt
python src/data_preprocessing.py
python src/feature_extraction.py
python src/train.py
python src/evaluate.py
This repository implements the full pipeline for MCI detection using CCDA. Let me know if you need custom modifications or additional feature

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
