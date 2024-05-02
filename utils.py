import subprocess
import librosa
import librosa.display
import soundfile as sf
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import model_from_json


def convert_flac_to_wav_librosa(directory):
    output_directory = os.path.join(directory, "wav_files")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".flac"):
            path_to_flac = os.path.join(directory, filename)
            path_to_wav = os.path.join(output_directory, os.path.splitext(filename)[0] + ".wav")

            # Load FLAC using librosa
            audio, sr = librosa.load(path_to_flac, sr=None) 
            # Save to WAV using soundfile
            sf.write(path_to_wav, audio, sr)
            print(f"Converted {path_to_flac} to {path_to_wav}")


def load_data(directory, label_file):
    """
    Load audio data and labels from the directory and the label file.
    
    Parameters:
    - directory: Path to the directory containing audio files.
    - label_file: Path to the file containing labels for audio files.
    
    Returns:
    - Tuple of lists (audio_data, audio_labels).
    """
    # Load labels from the text file
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[2] + '.wav'  # Adjusting to include '.wav' in the filename key
            label = parts[-1]  # The label is the last item in the list
            labels[file_name] = label
    
    # Initialize lists to store audio data and corresponding labels
    audio_data = []
    audio_labels = []

    # Iterate over audio files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            # Load audio file
            file_path = os.path.join(directory, file_name)
            try:
                audio, _ = librosa.load(file_path, sr=None)  # Load with the original sample rate
            except Exception as e:
                print(f"Error loading audio file '{file_name}': {e}")
                continue
            
            # Extract label for the audio file
            if file_name in labels:
                label = labels[file_name]
                audio_data.append(audio)
                audio_labels.append(label)
            else:
                print(f"No label found for audio file '{file_name}'")

    return audio_data, audio_labels



def split_data(audio_data, audio_labels, test_size=0.2, val_size=0.2):
    """
    Splits data into training, validation, and test sets.
    """
    # Split into training, validation and test
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        audio_data, audio_labels, test_size=test_size, random_state=42, stratify=audio_labels)

    adjusted_val_size = val_size / (1 - test_size)  

    # Split the remaining data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=adjusted_val_size, random_state=42, stratify=train_val_labels)
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

        
# =============Functions for Plotting ============

def plot_label_distribution(labels):
    # Count the frequency of each label
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Create a bar plot of the label counts
    labels, counts = zip(*label_counts.items())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels')
    plt.xticks(rotation=45)
    plt.show()


def plot_audio_length_distribution(audio_data, sr):
    # Calculate the length of each audio sample in seconds and append to the list
    lengths = []
    for audio in audio_data:
        length = len(audio) / sr
        lengths.append(length) 

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='green', alpha=0.7)
    plt.xlabel('Length of Audio (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Audio Lengths')
    plt.show()

def plot_lfcc(lfcc, sr, hop_length=512):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(lfcc, x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('LFCC')
    plt.tight_layout()
    plt.show()

def plot_mfcc(mfcc, sr, hop_length=512):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()



# ============= Preprocess Data (Feature Extraction)=============

# Refer to https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
# Lfcc only changing filter bank from mel-scaled to linear

def compute_mfcc(audio, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """ Compute MFCC features from audio signal """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, 
                                    n_mfcc=n_mfcc, n_fft=n_fft, 
                                    hop_length=hop_length)
    return mfccs

def linear_filter_bank(sr, n_fft, n_filters):
    ''' Linear Filter Bank for LFCC'''
    # Frequency range
    f_min = 0
    f_max = sr / 2
    # Linearly spaced 
    linear_freqs = np.linspace(f_min, f_max, n_filters + 2)

    # Convert frequencies to FFT bin numbers
    bins = np.floor((n_fft + 1) * linear_freqs / sr).astype(int)
    
    # Create linear filter bank
    filters = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for i in range(n_filters):
        filters[i, bins[i]:bins[i + 1]] = 1 
        filters[i, bins[i + 1]:bins[i + 2]] = -1  
    
    return filters


def compute_lfcc(audio, sr, n_filters=40, n_coefficients=13, n_fft=2048):
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft))
    filters = linear_filter_bank(sr, n_fft, n_filters)
    
    # Filter the spectrogram
    linear_features = np.dot(filters, spectrogram)
    
    # Take the logarithm of the power at each filter
    log_linear_features = np.log(np.maximum(linear_features, 1e-10))
    
    # Compute DCT to obtain coefficients
    lfcc = librosa.feature.mfcc(S=log_linear_features, n_mfcc=n_coefficients)
    
    return lfcc

def pad_features(features, target_length):
    ''' Padding for Shape Uniformity'''
    padded_features = []
    for feature in features:
        padding = ((0, 0), (0, target_length - feature.shape[1]))  
        padded_feature = np.pad(feature, pad_width=padding, mode='constant')
        padded_features.append(padded_feature)
    return padded_features

def load_and_reflectively_pad_audio(file_path, sr=None, n_fft=2048):
    ''' Padding to ensure FFT can be performed. '''
    audio, sr_loaded = librosa.load(file_path, sr=sr)
    if len(audio) < n_fft:
        padding_length = n_fft - len(audio)
        # Reflective padding
        audio = np.pad(audio, (0, padding_length), mode='reflect')
    return audio, sr_loaded


# Revised by ChatGPT
def load_and_extract_features(file_paths, feature_type='mfcc', sr=None, n_mfcc=13, n_fft=2048, hop_length=512, n_filters=40):
    features = []
    sr_loaded = None
    max_length = 0

    for file_path in file_paths:
        audio, sr_loaded = load_and_reflectively_pad_audio(file_path, sr=sr, n_fft=n_fft)
        if feature_type == 'mfcc':
            feat = compute_mfcc(audio, sr_loaded, n_mfcc, n_fft, hop_length)
        elif feature_type == 'lfcc':
            feat = compute_lfcc(audio, sr_loaded, n_filters, n_mfcc, n_fft)
        features.append(feat)
        max_length = max(max_length, feat.shape[1])
    
    # Pad features using the dedicated padding function
    padded_features = pad_features(features, max_length)

    # Concatenate all features into one array if not empty
    if padded_features:
        features = np.concatenate(padded_features, axis=0)
    
    return features, sr_loaded




#=======================Plot Model Training Accuracy and Loss======================================
# Code from homework-3
def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()



#==================== Evaluation ======================
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_eer(y_true, y_scores):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Calculate EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def plot_det_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Transform the FPR and FNR into normal deviates
    fpr_transformed = norm.ppf(fpr)
    fnr_transformed = norm.ppf(1 - tpr)

    plt.figure()
    plt.plot(fpr_transformed, fnr_transformed)
    plt.xlabel('False Positive Rate (normal deviate)')
    plt.ylabel('False Negative Rate (normal deviate)')
    plt.title('DET Curve')
    plt.grid(True)
    plt.show()