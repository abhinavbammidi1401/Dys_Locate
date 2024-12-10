import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your model (ensure the model is saved in a .h5 format)
model = tf.keras.models.load_model('D://Abhinav//Test//SPR_Project//dysarthria_detection_model.h5')  # Update with the correct path

# Function to preprocess the audio
def preprocess_audio(audio_file):
    # Load audio
    audio, sr = librosa.load(audio_file, sr=None)

    # Extract 20 MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc, axis=1)  # Use mean for feature reduction

    return audio, sr, mfcc

# Function to plot MFCCs
def plot_mfccs(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title("MFCCs")
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot spectrogram
def plot_spectrogram(audio, sr):
    stft = librosa.stft(audio)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit Interface
st.title("Dys-Locate")

st.write("Upload an audio file to evaluate speech and predict the health condition:")

# File upload section
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    # Process the audio
    audio, sr, mfcc_features = preprocess_audio(audio_file)
    
    # Normalize the features (optional, depending on your model's training)
    normalized_features = mfcc_features / np.max(mfcc_features)
    
    # Reshape to match the model input shape (1, 20)
    reshaped_features = normalized_features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(reshaped_features)
    
    # Display result
    predicted_label = np.argmax(prediction, axis=1)[0]  # If output is one-hot encoded
    if predicted_label == 0:
        st.write("Prediction: Healthy Speech")
    else:
        st.write("Prediction: Possible Speech Disorder (e.g., Dysarthria)")

    # Display prediction details
    st.write(f"Prediction result: {predicted_label}")
    
    # Add option to display MFCCs or Spectrogram
    st.write("Select visualizations to display:")
    show_mfcc = st.checkbox("Show MFCCs")
    show_spectrogram = st.checkbox("Show Spectrogram")

    if show_mfcc:
        st.write("MFCCs of the audio:")
        plot_mfccs(audio, sr)

    if show_spectrogram:
        st.write("Spectrogram of the audio:")
        plot_spectrogram(audio, sr)