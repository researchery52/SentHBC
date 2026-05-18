# SentBHC: Multimodal Hybrid Sentiment Analysis Framework

This repository contains the official implementation code and pre-rendered execution outputs for the **SentBHC** framework, a hybrid multimodal neural network designed for cross-domain sentiment analysis. 

SentBHC systematically fuses textual semantic representations (extracted via BERT) and acoustic paralinguistic features (extracted via Mel-Frequency Cepstral Coefficients - MFCCs) to classify sentiment profiles from digital multimedia streams into negative, neutral, and positive classes.

---

## 📂 Repository Structure

The repository is structured to ensure full scientific replication while maintaining strict adherence to double-blind peer review protocols:

* **`01_voice_feature_extraction.ipynb`**: The primary Jupyter Notebook containing the data ingestion pipeline, feature alignment, and the balanced training loop for the hybrid fusion architecture. *(Note: Execution outputs are cleared in the source code to maintain repository scannability and eliminate visualization time-outs).*
* **`01_voice_feature_extraction_executed_outputs.pdf`**: A fully rendered, static visual proof of the notebook execution. This document contains all live training logs, loss convergence metrics, confusion matrices, and classification reports for immediate web review.
* **`sentiment_analyzer.py`**: A standalone inference script optimized for local deployment (e.g., Anaconda environments) to test cross-domain generalization using real-time streaming pipelines.

---

## 🔗 Project Assets & Data Availability Statement

The data, pre-trained model checkpoints, and analytical logs supporting the findings of this study are openly available within this repository and its secure hosted mirrors. Raw or proprietary multimedia files are not redistributed in order to maintain strict compliance with the YouTube platform's terms of service and intellectual property frameworks. 

Instead, independent researchers can access the completely anonymized, de-identified text segments, derived acoustic feature matrices, trained model weights, and comprehensive evaluation matrices via the secure Google Drive links provided below:

* 📦 **[Download De-Identified Dataset (.zip)](https://drive.google.com/file/d/1EaMTvh-jEmZ9TKaVhRjyb3bmuBgZ2LqB/view?usp=sharing)** — Contains anonymized `.wav` audio segments and the synchronized transcription `.csv` metadata grid.
* 🧠 **[Download Pre-Trained Model Checkpoint (.pth)](https://drive.google.com/file/d/1tJkJodCCqhWctdJc7M5k3JKIyZNpozcb/view?usp=sharing)** — Contains the optimized weights (`model_v2_may.pth`) generated at the final convergence epoch of the SentBHC network.
* 📊 **[Access Experimental Artifacts & Reports Folder](https://drive.google.com/drive/folders/1zCMGaXZOGX8HxKFG8TQm5I-7nskJlIck?usp=drive_link)** — Contains full-resolution evaluation outputs, including execution logs, classification report text matrices, confusion matrix plots, and cross-domain linguistic word cloud visualizations.

---

## 🛠️ Prerequisites & Installation

Deployment of the framework locally requires an environment meeting the following configuration setup:

```bash
# Clone the anonymous repository
git clone [https://github.com/YOUR_ANONYMOUS_USER/YOUR_REPO_NAME.git](https://github.com/YOUR_ANONYMOUS_USER/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install required deep learning and signal processing dependencies
pip install torch transformers librosa pandas numpy matplotlib seaborn yt-dlp wordcloud tqdm
