# speech_emotion_recognition.py

import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

class SpeechEmotionRecognizer:
    def __init__(self):
        # Define emotions to detect
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Parameters for audio recording
        self.sample_rate = 22050
        self.duration = 3  # seconds
        
        # Create model or load pre-trained weights if available
        self.model = None 
        
        print("Speech Emotion Recognition System initialized")
    
    # def build_model(self, feature_dim):
    #     """Build the LSTM model for emotion classification with the correct input dimension"""
    #     model = Sequential()
    #     model.add(LSTM(128, input_shape=(1, feature_dim), return_sequences=True))
    #     model.add(Dropout(0.3))
    #     model.add(LSTM(128, return_sequences=False))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(len(self.emotions), activation='softmax'))
        
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     print(f"Model built successfully with input dimension: {feature_dim}")
    #     return model


    def build_model(self, feature_dim):
        model = Sequential()
        
        # More suitable architecture for smaller datasets
        model.add(Dense(256, activation='relu', input_shape=(1, feature_dim)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(len(self.emotions), activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', 
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    metrics=['accuracy'])
        return model
    

    def extract_features(self, audio_path=None, audio_data=None):
        """Extract audio features: MFCC, Chromagram, Mel Spectrogram, Spectral Contrast, Tonnetz"""
        
        if audio_path is not None:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        elif audio_data is not None:
            # Use provided audio data
            y = audio_data
            sr = self.sample_rate
        else:
            raise ValueError("Either audio_path or audio_data must be provided")
        
        # Apply preprocessing: Noise removal can be added here
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract features
        # MFCCs | spectral envelope
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chromagram | musical notes like C# Bb etc
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Mel Spectrogram | power or amplitude then in decibles
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectral Contrast | distinguish peaks for noise vs sounds
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Tonnetz | harmonic features
        harmonics = librosa.effects.harmonic(y=y)
        tonnetz = librosa.feature.tonnetz(y=harmonics, sr=sr)
        
        # Combine all features
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel_spec_db, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])
        
        return features, y
    
    def load_data(self, data_folders):
        """Load audio datasets and extract features"""
        features = []
        labels = []
        
        print("Loading and processing datasets...")
        
        # Define emotion mapping for each dataset
        ravdess_emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        tess_emotion_map = {
            'neutral': 'neutral',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fear': 'fearful',
            'disgust': 'disgust',
            'ps': 'surprised'  # 'ps' stands for 'pleasant surprised'
        }
        
        crema_emotion_map = {
            'NEU': 'neutral',
            'HAP': 'happy',
            'SAD': 'sad',
            'ANG': 'angry',
            'FEA': 'fearful',
            'DIS': 'disgust'
        }
        
      # Process files from each dataset
        for folder in data_folders:
            if folder == "RAVDESS":
                # Comment out RAVDESS for now - will train on it later
                print("Skipping RAVDESS dataset for now.")
                continue
                
            elif folder == "TESS":
                # TESS dataset structure:
                # TESS/OAF_angry/OAF_back_angry.wav
                # where OAF = Old Adult Female, YAF = Young Adult Female
                tess_dir = os.path.join(os.getcwd(), folder)
                for emotion_dir in os.listdir(tess_dir):
                    emotion_path = os.path.join(tess_dir, emotion_dir)
                    if os.path.isdir(emotion_path):
                        # Extract emotion from directory name
                        found_emotion = None
                        for emotion_name in tess_emotion_map:
                            if emotion_name in emotion_dir.lower():
                                found_emotion = tess_emotion_map[emotion_name]
                                break
                        
                        if found_emotion is None or found_emotion not in self.emotions:
                            continue
                            
                        # Process files in this emotion directory
                        for file in os.listdir(emotion_path):
                            if file.endswith(".wav"):
                                file_path = os.path.join(emotion_path, file)
                                
                                try:
                                    file_features, _ = self.extract_features(audio_path=file_path)
                                    features.append(file_features)
                                    labels.append(self.emotions.index(found_emotion))
                                    print(f"Processed TESS file: {file}")
                                except Exception as e:
                                    print(f"Error processing {file}: {e}")
            
            elif folder == "CREMA-D":
                # Comment out CREMA-D for now - will train on it later
                print("Skipping CREMA-D dataset for now.")
                continue
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.emotions))
        
        print(f"Data loading complete. {len(features)} samples loaded.")
        return features, labels
    
    def train_model(self, data_folders, epochs=50, batch_size=32):
        """Train the emotion recognition model"""
        # Load data
        X, y = self.load_data(data_folders)
        
        if len(X) == 0:
            print("No data loaded. Please check your dataset paths.")
            return None
        
        # Build model with correct input dimension
        feature_dim = X.shape[1]
        print(f"Feature dimension from data: {feature_dim}")
        self.model = self.build_model(feature_dim)
        
        # Try to load weights if they exist
        if os.path.exists('best_model.h5'):
            try:
                self.model.load_weights('best_model.h5')
                print("Loaded pre-trained model weights.")
            except Exception as e:
                print(f"Could not load model weights: {e}. Using newly initialized model.")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        # Create checkpoint to save best model
        checkpoint = ModelCheckpoint('best_model.h5', 
                                    monitor='val_accuracy', 
                                    save_best_only=True, 
                                    mode='max',
                                    verbose=1)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', 
                                      patience=10, 
                                      restore_best_weights=True,
                                      verbose=1)
        
        # Train the model
        print(f"Training model with {X_train.shape[0]} samples...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, early_stopping]
        )
        
        # Load the best model
        self.model.load_weights('best_model.h5')
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and generate confusion matrix"""
        # Predict emotions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Get the actual emotions present in the test set
        present_emotions = np.unique(y_true)
        present_emotion_labels = [self.emotions[i] for i in present_emotions]
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Display confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=present_emotion_labels, 
                    yticklabels=present_emotion_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Display classification report - only for present emotions
        report = classification_report(y_true, y_pred_classes, 
                                    target_names=present_emotion_labels,
                                    labels=present_emotions)
        print("Classification Report:")
        print(report)
        
        # Save classification report to file
        with open('classification_report.txt', 'w') as f:
            f.write(report)
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        plt.figure(figsize=(12, 5))
    
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def record_audio(self):
        """Record audio from the microphone"""
        print(f"Recording audio for {self.duration} seconds...")
        
        try:
            # Record audio
            audio_data = sd.rec(int(self.duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1)
            
            # Wait until recording is finished
            sd.wait()
            
            # Convert to mono if needed and flatten
            audio_data = audio_data.flatten()
            
            print("Recording complete!")
            return audio_data
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            print("Make sure your microphone is connected and working.")
            return None
    
    def predict_emotion(self, audio_path=None):
        """Predict emotion from audio file or recorded audio"""
        try:
            if audio_path is None:
                # Record audio
                audio_data = self.record_audio()
                if audio_data is None:
                    return None, None
                features, audio_signal = self.extract_features(audio_data=audio_data)
            else:
                # Extract features from file
                features, audio_signal = self.extract_features(audio_path=audio_path)
            
            # If model hasn't been created yet, we need to build it with the right dimensions
            if self.model is None:
                feature_dim = features.shape[0]
                print(f"Building model with input feature dimension: {feature_dim}")
                self.model = self.build_model(feature_dim)
                
                # Try to load weights if they exist
                if os.path.exists('best_model.h5'):
                    try:
                        self.model.load_weights('best_model.h5')
                        print("Loaded pre-trained model weights.")
                    except Exception as e:
                        print(f"Could not load model weights: {e}. Cannot predict without training first.")
                        return None, None
            
            # Reshape for LSTM input [samples, timesteps, features]
            features = features.reshape(1, 1, features.shape[0])
            
            # Predict emotion
            prediction = self.model.predict(features)[0]
            
            # Get predicted emotion
            predicted_emotion = self.emotions[np.argmax(prediction)]
            
            # Display results
            self.display_results(audio_signal, prediction, predicted_emotion)
            
            return predicted_emotion, prediction
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None
    
    def display_results(self, audio_signal, prediction, predicted_emotion):
        """Display waveform and emotion probabilities"""
        plt.figure(figsize=(12, 6))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(audio_signal)
        plt.title(f'Waveform - Predicted Emotion: {predicted_emotion}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        # Plot emotion probabilities
        plt.subplot(2, 1, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.emotions)))
        plt.bar(self.emotions, prediction, color=colors)
        plt.title('Emotion Probabilities')
        plt.xlabel('Emotions')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('emotion_prediction.png')
        
        # Show the plot if not in a headless environment
        try:
            plt.show()
        except:
            plt.close()
        
        # Print probabilities
        print("\nEmotion Probabilities:")
        for emotion, prob in zip(self.emotions, prediction):
            print(f"{emotion}: {prob:.4f}")

def main():
    # Create the emotion recognizer
    ser = SpeechEmotionRecognizer()
    
    # Options menu
    while True:
        print("\nSpeech Emotion Recognition System")
        print("1. Download datasets")
        print("2. Train model with TESS dataset")
        print("3. Record and predict emotion")
        print("4. Predict emotion from file")
        print("5. View Model Architecture")
        print("6. Exit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            try:
                # Check if dataset_downloader.py exists
                if not os.path.exists("dataset_downloader.py"):
                    print("Dataset downloader script not found. Please make sure you have dataset_downloader.py in the current directory.")
                    continue
                
                print("\nDownloading datasets...")
                os.system("python dataset_downloader.py --tess")  # Only download TESS for now
                print("Dataset downloaded successfully!")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
            
        elif choice == '2':
            # Check if datasets are available
            datasets = ["TESS"]  # Just use TESS for now
            available_datasets = []
            
            for dataset in datasets:
                if os.path.exists(dataset):
                    available_datasets.append(dataset)
            
            if not available_datasets:
                print("No datasets found. Please download the datasets first (option 1).")
                continue
            
            print(f"Training with available datasets: {', '.join(available_datasets)}")
            
            # Ask for number of epochs
            try:
                epochs = int(input("Enter number of epochs (default: 50): ") or "50")
                batch_size = int(input("Enter batch size (default: 32): ") or "32")
            except ValueError:
                print("Invalid input. Using default values.")
                epochs = 50
                batch_size = 32
            
            # Train the model
            ser.train_model(available_datasets, epochs=epochs, batch_size=batch_size)
            
        elif choice == '3':
            emotion, probabilities = ser.predict_emotion()
            if emotion:
                print(f"\nPredicted emotion: {emotion}")
            
        elif choice == '4':
            audio_path = input("Enter path to audio file: ")
            if os.path.isfile(audio_path):
                emotion, probabilities = ser.predict_emotion(audio_path)
                if emotion:
                    print(f"\nPredicted emotion: {emotion}")
            else:
                print(f"File not found: {audio_path}")
                
        elif choice == '5':
            # Display model architecture
            if ser.model is None:
                print("\nModel has not been built yet. Train the model first or predict an emotion.")
            else:
                print("\nModel Architecture:")
                ser.model.summary()
            
        elif choice == '6':
            print("Exiting program...")
            break
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()