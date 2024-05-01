# group_5_DL4M1
Loading Required Libraries and Modules: The script imports utilities from a custom module named utils and various libraries for handling audio data such as librosa, soundfile, and others for data manipulation and visualization.
Data Conversion (Optional): There's a provision for converting audio files from one format to another, though it's commented out, suggesting it might be a one-time setup step.
Loading Data: The script includes functionality to load audio files and corresponding labels from specified directories.
Data Inspection: There are examples showing how to play an audio file, display its spectrogram, and print some basic statistics about the loaded data.
Data Splitting: The audio data is divided into training, validation, and test sets.
Visualization: Functions are called to plot label distribution and audio length distribution.
Feature Extraction and Model Training:
Extraction of embeddings from audio using a pre-trained model (YAMNet model hosted on TensorFlow Hub).
A temporal convolutional network (TCN) model is defined and trained on these embeddings.
Model Evaluation: The trained model is evaluated, and the performance metrics (such as EER, loss, and accuracy) are calculated and displayed.
Saving and Loading Models: The script includes steps to save and later reload the trained model for further evaluation or deployment.


About
## Description

This project is inspired by the 2024 Singing Deepfake Detection Challenge. It contains a basic pipeline designed to identify and flag deepfake content in Mandarin. This pipeline uses CNN architecture to analyze audio files and detect discrepancies that may suggest manipulation. The dataset we used was derived from the Controlled SVDD dataset. The original CtrSVDD dataset only provides deepfake audio files and some bonafide files. The rest of the bonafide files are downloaded separately from the following datasets: Oniku, Ofuton, Kiritan, and JVS-MuSiC. After downloading, the bonafide audio files were segmented based on the requirements in timestamps. We performed the segmentation by using this line of code in the terminal:


python segment.py /path_to_the_bonafide_folder/ /path_to_related_timestamps.txt /path_to_where_to_store_segmented_bonafide_audio

The segmentation file helped us to shorten the audio files to 4 to 7 seconds. 

The obtained full dataset is significant in size, over 20 GB. Therefore, we decided to customize the dataset to make it smaller and suitable for small prototype testing. The method we chose for customization was to annotate the full dataset based on the language and organize the data into two new datasets (one for Mandarin and one for Japanese). Because of the imbalance between bonafides and deepfakes in the datasets, we created functions for data augmentation and performed them on our Japanese dataset. We chose the Japanese dataset to perform data augmentation because of its smaller size. After augmentation, our bonafide & deepfake audio files are almost balanced (50:50). However, due to labeling issues, unknown reasons for different corrupted files showing in different setups (macbook locally vs. virtual coding environment), and limitation of time, the Japanese dataset was not able to be fully digested and processed in the local environment. It only worked in virtual coding environments such as Lightning AI. Therefore, we eventually decided to use the Mandarin dataset that is much larger in size for this project. However, we have uploaded the code and how the Japanese dataset is organized in this GitHub project; see file: _________________


Installation & System Setting:


To install the necessary dependencies, follow the requirements in ‘requirements.txt’ under the folder ‘environment’.
Warning: 
We tested and ran this project on macbook M1. The whole project has not been tested fully on virtual environments. If running this project, recommend running it locally. 


