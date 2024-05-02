## Japanese Dataset
See Japanese dataset in lightning AI link:
    https://lightning.ai/live-session/1bc4b830-565d-4f30-99bc-eaba451052f2

## Description
This project is inspired by the 2024 Singing Deepfake Detection Challenge. It contains a pipeline designed to identify and flag deepfake content in Mandarin and Japanese. This pipeline uses CNN architecture to analyze singing files and detect discrepancies that may suggest manipulation. The dataset we used was derived from the Controlled SVDD dataset. 

## System Setting
Since our groupmates work on both datasets parallelly, our model for the Mandarin singing deepfake detection pipeline is performed locally on a Macbook with an M1 chip, and the Japanese dataset singing deepfake detection pipeline is performed on Lightning AI. Besides these two differences, these two pipelines went through a very similar process. 

The setup requirement for both local and virtual environments is in the requiementes.yml file under the environment folder. 

## Dataset
The original CtrSVDD dataset only provides deepfake audio files and some bonafide files. The rest of the bonafide files are downloaded separately from the following datasets: Oniku, Ofuton, Kiritan, and JVS-MuSiC. After downloading, the bonafide audio files were segmented based on the requirements in timestamps. We performed the segmentation by using this line of code in the terminal:

‘’’
python segment.py /path_to_the_bonafide_folder/ /path_to_related_timestamps.txt /path_to_where_to_store_segmented_bonafide_audio
‘’’

The segmentation file helped us to shorten the audio files to 4 to 7 seconds. 

The obtained full dataset is significant in size, over 20 GB. We experienced difficulties in importing the dataset into our virtual coding environment for processing. Therefore, we decided to customize the dataset to make it smaller and suitable for small prototype testing.  The method we chose for customization was to annotate the full dataset based on the language and organize the data into two new datasets (one for Mandarin and one for Japanese). Because of the imbalance between bonafides and deepfakes in both datasets, we created functions for data augmentation and performed them on our Japanese dataset. After the data augmentation, the Japanese dataset reaches an almost 50:50 balance between the number of bonafide and deepfake audio files, contrasting with the imbalanced nature of the Mandarin dataset. 

During the data preprocessing stage, we at first attempted to convert all the FLAC files in both datasets to WAV format so we could play and listen to them in the virtual environment. However, due to the cost of time, we only converted the files in the Mandarin dataset. Thus, the Mandarin dataset consists of WAV files while the Japanese dataset consists of FLAC files now. 


## Data splitting
For the Japanese dataset, the provided training set was first balanced by augmenting only the bonafide audio, since there was a majority of deepfake audio. This balanced set was then used as a whole for training. The provided development set was split into two equal parts that were used as the test set and validation set respectively. Given that the provided development set was equal in size to the provided training set, we balanced the test set and evaluation set by simply eliminating the excess deepfake audio. 

## Feature Extraction
We used LFCC as our feature; this feature was chosen because it was required by the challenge. We implemented the algorithm for LFCC with Hann windowing and a linear filter bank we designed manually. Necessary padding was applied to ensure STFT can be performed and the uniformity in the shape of features. After all the LFCC features had been extracted, we stored this information in h5 files and then converted them to npy files when used. 

## Model Structure
Both datasets were fed into the same CNN model, which features three convolutional layers. The architecture also incorporates two maxpooling layers to reduce the height dimensions and a 'same' padding strategy to maintain the width throughout the convolutions. Following this, the network flattens the output and channels it through two dense layers for binary classification, using a sigmoid activation in the output layer. 

## Result & Evaluation
We used the EER(Equal Error Rate) metric to evaluate both our models. To obtain the EER value, a DET (Detection Error Tradeoff) curve was plotted. The DET curve is a visualization of the tradeoff between the number of false acceptances and false rejections made by the model, with each of these values plotted on either axis. We chose this metric over accuracy because it provides a more comprehensive evaluation of the model’s performance by taking into account the distribution of the two classes and the performance of the model on each class. The point on this curve where the false acceptance rate is equivalent to the false rejection rate was determined as the EER. For our Mandarin model, we obtained an EER of 75% at a threshold of 0.85. This means that the model falsely accepts bonafide audio as deepfake audio or vice-versa in 75% of its predictions. However, for our Japanese dataset, we obtained a significantly lower (better) EER of 50% at the same threshold. We therefore infer that data balancing and data augmentation made a huge (positive) difference to our model’s performance. 
