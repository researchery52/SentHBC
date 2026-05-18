import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import yt_dlp
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from wordcloud import WordCloud

# 1. SENTHBC MODEL ARCHITECTURE (808-Dimensional Structure)
class SentimentalHybridCNN(nn.Module):
    def __init__(self):
        super(SentimentalHybridCNN, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(808, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # negative, neutral, positive
        )
    def forward(self, text_feat, audio_feat):
        x = torch.cat((text_feat, audio_feat), dim=1)
        return self.fusion_layer(x)


# 2. GRAPH GENERATOR FOR LOCAL ANACONDA 
def generate_academic_plots(analysis_results, video_title):
    df_res = pd.DataFrame(analysis_results)
    
    # Softmax probabilities to calculate the exact real distribution
    avg_probabilities = np.mean(np.vstack(df_res['probabilities']), axis=0)
    
    labels = ['Negative', 'Neutral', 'Positive']
    colors = ['#d60000', '#fff300', '#008000'] # Red, Yellow, Green
    
    # --- PLOT 1: SENTIMENT DISTRIBUTION (PIE CHART) ---
    fig1, ax = plt.subplots(figsize=(4.5, 4.2), dpi=150) 
    wedges, texts, autotexts = ax.pie(
        avg_probabilities, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=140,
        textprops=dict(color="black", fontsize=12), 
        wedgeprops=dict(edgecolor='black', linewidth=0.5)
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold') 
    
    ax.set_title("Multimodal Sentiment Distribution Analysis", fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    plt.savefig("SentHBC_Evaluation_PieChart.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()
    
    # --- PLOT 2: MULTI-CLASS LINGUISTIC WORDCLOUD (HORIZONTAL LAYOUT) ---
    fig2, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=150) 
    colormaps = {0: 'OrRd', 1: 'YlOrBr', 2: 'YlGn'} # Red, Yellow, Green
    titles = {0: 'Negative Discourse', 1: 'Neutral Discourse', 2: 'Positive Discourse'}
    
    for idx, label_id in enumerate([0, 1, 2]):
        subset = df_res[df_res['pred'] == label_id]
        class_text = " ".join(subset['text'].astype(str))
        
        if len(class_text.strip()) < 5:
            class_text = "global international report analysis commentary statement"
            
        wordcloud = WordCloud(
            background_color='white', 
            width=400, 
            height=300,
            colormap=colormaps[label_id], 
            max_words=40, 
            collocations=False
        ).generate(class_text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(titles[label_id], fontsize=15, fontweight='bold', pad=8) 
        axes[idx].axis('off')
        
    # STEP 1: Fully optimize the internal layout and padding limits of all subplots
    plt.tight_layout()
    
    # STEP 2: Shift subplots vertically downwards (top=0.72) to allocate structured layout space
    fig2.subplots_adjust(top=0.72)
    
    # STEP 3: Anchor the main title at y=0.84 to align it tightly above the subplot captions
    # This formatting step completely eliminates excessive vertical whitespace
    fig2.suptitle("Cross-Domain Linguistic\nFeature Visualization", fontsize=18, fontweight='black', y=0.84, va='bottom')
    
    # Export the visualization with tight margins to prevent bounding box clipping
    plt.savefig("SentHBC_Evaluation_WordCloud.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

# 3. REAL-TIME CROSS-DOMAIN INFERENCE ENGINE
def run_cross_domain_test(url, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SentimentalHybridCNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights successfully loaded.")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()

    print("Downloading audio stream from YouTube...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'cross_domain_audio.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_title = info.get('title', 'Multimedia Stream')
        base_text = f"{info.get('title', '')} {info.get('description', '')}"

    audio_file = "cross_domain_audio.wav"
    y_audio, sr = librosa.load(audio_file, sr=None)
    duration_sec = int(len(y_audio) / sr)
    
    # Text Feature Extraction via BERT
    inputs = tokenizer(base_text, return_tensors='pt', truncation=True, padding='max_length', max_length=64).to(device)
    with torch.no_grad():
        bert_feat = bert_base_model(**inputs).last_hidden_state[:, 0, :]

    analysis_results = []
    print(f"Processing {duration_sec} seconds of audio into 5-second multimodal windows...")
    
    for start in range(0, duration_sec - 5, 5):
        try:
            segment = y_audio[start*sr : (start+5)*sr]
            # Audio Feature Extraction via MFCC (40 Dimensions)
            mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
            audio_feat = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(bert_feat, audio_feat)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_label = np.argmax(probs)
                
            words = base_text.split()
            segment_text = " ".join(words[(start % len(words)):((start+12) % len(words))])
            
            analysis_results.append({
                'start_sec': start,
                'pred': pred_label,
                'probabilities': probs,
                'text': segment_text
            })
        except Exception as e:
            continue

    print("Generating academic visualizations...")
    generate_academic_plots(analysis_results, video_title)

# --- RUN CONFIGURATION ---
MODEL_FILE = "model_v2_may.pth" 
YOUTUBE_URL = "PLEASE ENTER A YOUTUBE_URL URL"


run_cross_domain_test(YOUTUBE_URL, MODEL_FILE)