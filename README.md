# Speech Emotion Recognition System

This repository contains a Python-based Speech Emotion Recognition (SER) system that can predict emotions from audio input. The system uses an LSTM-based model trained on audio features to classify emotions such as neutral, happy, sad, angry, fearful, disgust, and surprised.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Creating a Virtual Environment](#creating-a-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Usage](#usage)
- [Training Data](#training-data)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Audio Feature Extraction**: Extracts MFCC, Chromagram, Mel Spectrogram, Spectral Contrast, and Tonnetz features from audio.
- **LSTM Model**: Utilizes a Sequential LSTM model for emotion classification.
- **Real-time Prediction**: Ability to record audio from a microphone and predict emotion in real-time.
- **File Prediction**: Predicts emotion from a given audio file.
- **Model Training & Evaluation**: Includes functionality to train the model, evaluate its performance, and visualize training history and confusion matrix.

## Requirements

To run this project, you need Python 3.x and the following libraries:

- librosa
- sounddevice
- tensorflow
- matplotlib
- seaborn
- numpy
- pandas
- scikit-learn

## Setup

Follow these steps to set up the project in a virtual environment.

### Creating a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

1. Navigate to your project directory:
   ```bash
   cd your_project_directory
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
   This command creates a folder named `venv` in your project directory, containing the virtual environment.

3. Activate the virtual environment:
   
   **On Windows:**
   ```bash
   .\venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   You should see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

### Installing Dependencies

Once your virtual environment is active, install the required libraries:

```bash
pip install librosa sounddevice tensorflow matplotlib seaborn numpy pandas scikit-learn
```

## Usage

1. Run the main script:
   ```bash
   python speech_emotion_recognition.py
   ```

2. Follow the menu options:
   - **1. Download datasets**: This option will attempt to download the necessary audio datasets. For now, it is configured to download only the TESS dataset. You might need a `dataset_downloader.py` script in the same directory for this to work.
   - **2. Train model with TESS dataset**: Trains the emotion recognition model using the TESS dataset. You will be prompted to enter the number of epochs and batch size.
   - **3. Record and predict emotion**: Records audio from your microphone for a few seconds and then predicts the emotion.
   - **4. Predict emotion from file**: Prompts you to enter the path to an audio file and then predicts the emotion from that file.
   - **5. View Model Architecture**: Displays the summary of the trained model's architecture.
   - **6. Exit**: Exits the program.

## Training Data

Currently, only the TESS (Toronto Emotional Speech Set) dataset is used for training. The `speech_emotion_recognition.py` script is set up to skip RAVDESS and CREMA-D datasets during the load_data phase.

## Results

After training, the system generates plots for training history and a confusion matrix.

### Training History
This plot shows the model's accuracy and loss during training and validation over epochs.

### Confusion Matrix
The confusion matrix visualizes the performance of the classification model, showing the counts of correct and incorrect predictions for each emotion.

### Emotion Prediction Example
An example of a predicted emotion waveform and its corresponding probability distribution.

## Future Work

- **More Diverse Training Data**: Incorporate RAVDESS, CREMA-D, and other datasets to improve generalization.
- **Model Optimization**: Experiment with different model architectures (e.g., CNN-LSTM, attention mechanisms) and hyperparameters.
- **Real-time Performance**: Optimize for faster real-time prediction.
- **Robustness**: Improve the model's ability to detect recorded audio emotion precisely, as more training is needed for higher accuracy on diverse real-world audio.
- **User Interface**: Develop a more interactive graphical user interface (GUI) for easier use.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
