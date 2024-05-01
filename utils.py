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
from tcn import TCN
from keras.models import model_from_json


# def convert_flac_to_wav_librosa(directory):
#     output_directory = os.path.join(directory, "wav_files")
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     for filename in os.listdir(directory):
#         if filename.endswith(".flac"):
#             path_to_flac = os.path.join(directory, filename)
#             path_to_wav = os.path.join(output_directory, os.path.splitext(filename)[0] + ".wav")

#             # Load FLAC using librosa
#             audio, sr = librosa.load(path_to_flac, sr=None) 
#             # Save to WAV using soundfile
#             sf.write(path_to_wav, audio, sr)
#             print(f"Converted {path_to_flac} to {path_to_wav}")


def load_data(directory, label_file):
    """
    Load audio data and labels from a directory and a label file.
    
    Parameters:
    - directory: Path to the directory containing audio files.
    - label_file: Path to the file containing labels for audio files.
    
    Returns:
    - Tuple of lists (audio_data, audio_labels) containing audio signal arrays and corresponding labels.
    """
    # Load labels from the text file
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[2] + '.flac'  # Adjusting to include '.wav' in the filename key
            label = parts[-1]  # The label is the last item in the list
            labels[file_name] = label
    
    # Initialize lists to store audio data and corresponding labels
    audio_data = []
    audio_labels = []

    # Iterate over audio files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.flac'):
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

def load_flac_data(directory, label_file):
    """
    Load audio data and labels from a directory and a label file.
    
    Parameters:
    - directory: Path to the directory containing audio files.
    - label_file: Path to the file containing labels for audio files.
    
    Returns:
    - Tuple of lists (audio_data, audio_labels) containing audio signal arrays and corresponding labels.
    """
    # Load labels from the text file
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[2] + '.flac'  # Adjusting to include '.flac' in the filename key
            label = parts[-1]  # The label is the last item in the list
            labels[file_name] = label
    
    # Initialize lists to store audio data and corresponding labels
    audio_data = []
    audio_labels = []

    # Iterate over audio files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.flac'):
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

def load_flac_data_reduced(directory, label_file):
    """
    Load audio data and labels from a directory and a label file.
    
    Parameters:
    - directory: Path to the directory containing audio files.
    - label_file: Path to the file containing labels for audio files.
    
    Returns:
    - Tuple of lists (audio_data, audio_labels) containing audio signal arrays and corresponding labels.
    """
    # Load labels from the text file
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[2] + '.flac'  # Adjusting to include '.flac' in the filename key
            label = parts[-1]  # The label is the last item in the list
            labels[file_name] = label
    
    # Count the number of "deepfake" and "bonafide" labels
    num_deepfake = sum(1 for label in labels.values() if label == 'deepfake')
    num_bonafide = sum(1 for label in labels.values() if label == 'bonafide')
    
    # Calculate the number of "deepfake" and "bonafide" files to retain based on the 40% requirement
    num_to_retain_deepfake = int(0.2 * num_deepfake)
    num_to_retain_bonafide = int(1 * num_bonafide)
    
    # Initialize lists to store audio data and corresponding labels
    audio_data = []
    audio_labels = []

    # Iterate over audio files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.flac'):
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
                
                # Check if the label is "deepfake" or "bonafide" and if the respective quota is not exceeded
                if (label == 'deepfake' and num_to_retain_deepfake > 0) or (label == 'bonafide' and num_to_retain_bonafide > 0):
                    audio_data.append(audio)
                    audio_labels.append(label)
                    
                    # Reduce the respective quota
                    if label == 'deepfake':
                        num_to_retain_deepfake -= 1
                    else:
                        num_to_retain_bonafide -= 1
            else:
                print(f"No label found for audio file '{file_name}'")

    return audio_data, audio_labels

def copy_selected_flac_files(directory, label_file):
    """
    Copy selected audio files from a directory to two new directories based on the provided labels.
    
    Parameters:
    - directory: Path to the directory containing audio files.
    - label_file: Path to the file containing labels for audio files.
    
    Returns:
    - None
    """
    # Load labels from the text file
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[2] + '.flac'  # Adjusting to include '.flac' in the filename key
            label = parts[-1]  # The label is the last item in the list
            labels[file_name] = label
    
    # Count the number of "deepfake" and "bonafide" labels
    num_deepfake = sum(1 for label in labels.values() if label == 'deepfake')
    num_bonafide = sum(1 for label in labels.values() if label == 'bonafide')
    
    # Calculate the number of "deepfake" and "bonafide" files to retain based on the 40% requirement
    num_to_retain_deepfake = int(0.2 * num_deepfake)
    num_to_retain_bonafide = int(1 * num_bonafide)
    
    # Initialize lists to store selected file names
    selected_files_deepfake = []
    selected_files_bonafide = []

    # Iterate over audio files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            # Extract label for the audio file
            if file_name in labels:
                label = labels[file_name]
                
                # Check if the label is "deepfake" or "bonafide" and if the respective quota is not exceeded
                if (label == 'deepfake' and num_to_retain_deepfake > 0) or (label == 'bonafide' and num_to_retain_bonafide > 0):
                    if label == 'deepfake':
                        selected_files_deepfake.append(file_name)
                        num_to_retain_deepfake -= 1
                    else:
                        selected_files_bonafide.append(file_name)
                        num_to_retain_bonafide -= 1
            else:
                print(f"No label found for audio file '{file_name}'")

    # Create two new directories
    new_directory_1 = '/teamspace/studios/this_studio/jap_testset'  # Replace 'directory_1' with your desired directory name
    new_directory_2 = '/teamspace/studios/this_studio/jap_valset'  # Replace 'directory_2' with your desired directory name
    os.makedirs(new_directory_1, exist_ok=True)
    os.makedirs(new_directory_2, exist_ok=True)
    
    # Determine the number of files to copy into each directory
    num_files_per_directory = min(len(selected_files_deepfake), len(selected_files_bonafide)) // 2
    
    # Shuffle the selected files
    random.shuffle(selected_files_deepfake)
    random.shuffle(selected_files_bonafide)
    
    # Copy selected audio files into the new directories
    for file_name in selected_files_deepfake[:num_files_per_directory]:
        source_path = os.path.join(directory, file_name)
        destination_path = os.path.join(new_directory_1, file_name)
        shutil.copy(source_path, destination_path)
    
    for file_name in selected_files_bonafide[:num_files_per_directory]:
        source_path = os.path.join(directory, file_name)
        destination_path = os.path.join(new_directory_1, file_name)
        shutil.copy(source_path, destination_path)
    
    for file_name in selected_files_deepfake[num_files_per_directory:]:
        source_path = os.path.join(directory, file_name)
        destination_path = os.path.join(new_directory_2, file_name)
        shutil.copy(source_path, destination_path)
    
    for file_name in selected_files_bonafide[num_files_per_directory:]:
        source_path = os.path.join(directory, file_name)
        destination_path = os.path.join(new_directory_2, file_name)
        shutil.copy(source_path, destination_path)
    
    # Create label files for each directory
    with open(os.path.join(new_directory_1, 'labels.txt'), 'w') as f:
        for file_name in selected_files_deepfake[:num_files_per_directory]:
            label = labels[file_name]
            f.write(f"{label}\n")
        
        for file_name in selected_files_bonafide[:num_files_per_directory]:
            label = labels[file_name]
            f.write(f"{label}\n")
    
    with open(os.path.join(new_directory_2, 'labels.txt'), 'w') as f:
        for file_name in selected_files_deepfake[num_files_per_directory:]:
            label = labels[file_name]
            f.write(f"{label}\n")
        
        for file_name in selected_files_bonafide[num_files_per_directory:]:
            label = labels[file_name]
            f.write(f"{label}\n")


def split_data(audio_data, audio_labels, test_size=0.2, val_size=0.2):
    """
    Splits data into training, validation, and test sets.
    test_size: proportion of data to reserve for the test set.
    val_size: proportion of the remaining data to reserve for validation.
    """
    # First, split into training+validation and test
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        audio_data, audio_labels, test_size=test_size, random_state=42, stratify=audio_labels)

    # Calculate the adjusted validation size from the remaining data
    adjusted_val_size = val_size / (1 - test_size)  # Adjust val_size proportionally

    # Split the remaining data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=adjusted_val_size, random_state=42, stratify=train_val_labels)
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

import shutil

def test_val_split(audio_data, audio_labels, test_size=0.5):
    """
    Splits Japanese dev data into validation, and test sets (splits the dev set into two equal parts)
    """

    # Split the remaining data into training and validation sets
    test_data, val_data, test_labels, val_labels = train_test_split(
        audio_data, audio_labels, test_size=test_size, random_state=42, stratify=audio_labels)
    
    return  test_data, val_data, test_labels, val_labels

def move_bonafide_files(directory, label_file, target_directory):
    """
    Moves all audio files labeled as 'bonafide' to a specified directory.
    
    Parameters:
    - directory: The directory where audio files are stored.
    - label_file: The file containing labels for the audio files.
    - target_directory: The directory to move 'bonafide' labeled files to.
    """
    # Load labels from the text file
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[2]  # Assume filename is the third item in the list
            label = parts[-1]     # Assume label is the last item in the list
            labels[file_name] = label

    # Check and move files labeled as 'bonafide'
    for file_name, label in labels.items():
        if label.lower() == 'bonafide':
            source_path = os.path.join(directory, file_name)
            target_path = os.path.join(target_directory, file_name)
            
            # Create target directory if it doesn't exist
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
            
            # Move the file
            shutil.move(source_path, target_path)
            print(f"Moved '{file_name}' to '{target_directory}'")


def process_dataset(dataset_dir, timestamp_path, output_dir):
    # Load timestamps
    timestamps = load_timestamps(timestamp_path)
    
    # Process each file as per timestamps
    for file_info in timestamps:
        audio_path = os.path.join(dataset_dir, file_info['file_name'])
        segments = segment_audio(audio_path, file_info['start_time'], file_info['end_time'])
        
        # Save each segment
        for idx, segment in enumerate(segments):
            save_path = os.path.join(output_dir, f"{file_info['file_name']}_{idx}.wav")
            save_audio(segment, save_path)


def augment_and_concatenate_bonafide(audio_data, audio_labels, output_dir, pitch_shift_steps=2):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    augmented_data = []
    augmented_labels = []

    for i, (audio, label) in enumerate(zip(audio_data, audio_labels)):
        if label == "bonafide":
            # Perform pitch shifting
            for shift_step in range(-pitch_shift_steps, pitch_shift_steps + 1):
                if shift_step != 0:  # Don't shift by 0 steps
                    augmented_audio = librosa.effects.pitch_shift(audio, sr=44100, n_steps=shift_step)
                    
                    # Save the augmented audio file in .flac format
                    output_filename = f"bonafide_augmented_{i}_shift_{shift_step}.wav"
                    output_path = os.path.join(output_dir, output_filename)
                    sf.write(output_path, augmented_audio, 44100, subtype='PCM_24')

                    # Append augmented data and labels
                    augmented_data.append(output_path)
                    augmented_labels.append("bonafide")

    # Concatenate augmented data with original data
    concatenated_data = audio_data + augmented_data
    concatenated_labels = audio_labels + augmented_labels

    return concatenated_data, concatenated_labels

def append_directories(source_dir, target_dir, source_labels, target_labels):
    # List all audio files in the source directory
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.flac')]
    
    # List all audio files in the target directory
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.flac')]

    # Append audio files from source directory to target directory
    for file in source_files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        shutil.copyfile(source_path, target_path)

    # Append labels from source label list to target label list
    target_labels.extend(source_labels)

    return target_labels

def create_bonafide_labels(audio_dir):
    # List all audio files in the directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]

    # Create "bonafide" labels for the audio files
    audio_labels = ["bonafide"] * len(audio_files)

    return audio_labels


        
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
    lengths = [len(audio)/sr for audio in audio_data]  # Convert sample counts to seconds

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

#reference at this: (add the github link)

def compute_mfcc(audio, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """ Compute MFCC features from audio signal """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, 
                                    n_mfcc=n_mfcc, n_fft=n_fft, 
                                    hop_length=hop_length)
    return mfccs
# refer to: 
#if not work, use MFCC from librosa first
def linear_filter_bank(sr, n_fft, n_filters):
    # Frequency range
    f_min = 0
    f_max = sr / 2
    # Linearly spaced frequencies between f_min and f_max
    linear_freqs = np.linspace(f_min, f_max, n_filters + 2)
    # Convert frequencies to FFT bin numbers
    bins = np.floor((n_fft + 1) * linear_freqs / sr).astype(int)
    
    # Create filter bank
    filters = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for i in range(n_filters):
        filters[i, bins[i]:bins[i + 1]] = 1  # Rising edge
        filters[i, bins[i + 1]:bins[i + 2]] = -1  # Falling edge
    
    return filters


def compute_lfcc(audio, sr, n_filters=40, n_coefficients=13, n_fft=2048):
    """ Compute LFCC features from audio signal """
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft))
    filters = linear_filter_bank(sr, n_fft, n_filters)
    
    # Filter the spectrogram
    linear_features = np.dot(filters, spectrogram)
    
    # Take the logarithm of the power at each filter
    log_linear_features = np.log(np.maximum(linear_features, 1e-10))
    
    # Compute DCT to obtain LFCCs
    lfcc = librosa.feature.mfcc(S=log_linear_features, n_mfcc=n_coefficients)
    
    return lfcc

def pad_features(features, target_length):
    padded_features = []
    for feature in features:
        padding = ((0, 0), (0, target_length - feature.shape[1]))  # Pad second dimension
        padded_feature = np.pad(feature, pad_width=padding, mode='constant')
        padded_features.append(padded_feature)
    return padded_features

def load_and_reflectively_pad_audio(file_path, sr=None, n_fft=2048):
    audio, sr_loaded = librosa.load(file_path, sr=sr)
    if len(audio) < n_fft:
        padding_length = n_fft - len(audio)
        # Reflective padding
        audio = np.pad(audio, (0, padding_length), mode='reflect')
    return audio, sr_loaded


#for batch processing
# def load_and_extract_features(file_paths, feature_type='mfcc', sr=None, n_mfcc=13, n_fft=1024, hop_length=512, n_filters=40):
#     features = []
#     max_length = 0  # Track the maximum length of the feature arrays

#     # First, extract features and find the maximum length
#     for file_path in file_paths:
#         audio, sr_loaded = librosa.load(file_path, sr=sr)
#         if feature_type == 'mfcc':
#             feat = librosa.feature.mfcc(y=audio, sr=sr_loaded, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#         elif feature_type == 'lfcc':
#             feat = compute_lfcc(audio, sr_loaded, n_filters, n_mfcc, n_fft)  # Assume compute_lfcc is defined elsewhere
#         features.append(feat)
#         if feat.shape[1] > max_length:
#             max_length = feat.shape[1]

#     # Pad each feature array to the maximum length
#     features_padded = [pad_features(feat, max_length) for feat in features]

#     # Convert the list of padded features to a numpy array
#     features_array = np.array(features_padded)
    
#     return features_array, sr_loaded

#revised by ChatGPT
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




#=======================================================================
#plot model training loss
#code from homework-3
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



#=====================================================

def extract_yamnet_embedding(wav_data, yamnet):
    """
    Run YAMNet to extract embeddings from the wav data.

    Parameters
    ----------
    wav_data : np.ndarray
        The audio signal to be processed.
    yamnet : tensorflow.keras.Model
        The pre-trained YAMNet model.

    Returns
    -------
    np.ndarray
        The extracted embeddings from YAMNet.
    """
    # Hint: check the tensorflow models to see how YAMNET should be used
    # YOUR CODE HERE
     # Ensure wav_data is mono by averaging if it's not
    scores, embeddings, spectrogram = yamnet(wav_data)
    
    return embeddings

def reload_tcn(model_path, weights_path, optimizer, loss, metrics):
    """
    Reload a TCN model from a JSON file and restore its weights. 
    Preferred method when dealing with custom layers.

    Parameters
    ----------
    model_path : str
        The path to the JSON file containing the model architecture.
    weights_path : str
        The path to the model weights file.
    optimizer : str or tf.keras.optimizers.Optimizer
        The optimizer to use when compiling the model.
    loss : str or tf.keras.losses.Loss
        The loss function to use when compiling the model.
    metrics : list of str or tf.keras.metrics.Metric
        The list of metrics to use when compiling the model.

    Returns
    -------
    reloaded_model : tf.keras.Model
        The reloaded model with the restored weights.

    Example
    -------
    >>> model_path = 'path/to/saved_model.json'
    >>> weights_path = 'path/to/saved_weights.h5'
    >>> optimizer = 'adam'
    >>> loss = 'sparse_categorical_crossentropy'
    >>> metrics = ['accuracy']
    >>> reloaded_model = reload_tcn(model_path, weights_path, optimizer, loss, metrics)
    """
    # Load the best checkpoint of the model from json file (due to custom layers)
    loaded_json = open(model_path, 'r').read()
    reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

    reloaded_model.compile(optimizer=optimizer,
                            loss=loss, 
                            metrics=metrics)
    # restore weights
    reloaded_model.load_weights(weights_path)

    return reloaded_model

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