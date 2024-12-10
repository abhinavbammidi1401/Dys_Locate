# Dys-Locate: A System for Dysarthric Speech Detection and Transcription

Dys-Locate is a machine learning-based system designed to detect and transcribe dysarthric speech, enhancing accessibility for individuals with speech impairments. Leveraging the TORGO dataset and advanced speech processing techniques, Dys-Locate aims to provide accurate and reliable transcription solutions for better communication and understanding.

## Features
- **Automatic Dysarthria Detection**: Identifies whether speech is dysarthric or not.
- **Speech Transcription**: Transcribes dysarthric speech with improved accuracy.
- **Interactive Interface**: A Streamlit-based web application for real-time detection and transcription.

## Project Structure

- README.md - Project Documentation
- proj.ipynb - Jupyter notebook for the model building
- torgo - Dataset direcotry
- app.py - Streamlit interface for the application
- requirements.txt - Python dependencies
- dysarthia_detection_model.h5 - the final trained model
- mfcc_data.pkl - pickle file containing the MFCCs for future use
- processed_data.pkl - pickle file containing the preprocessed audio file

## Getting Started
### **Prerequisistes**
- Python 3.8 or above
- GPU for trianing (optional)
- TORGO dataset (download from TORGO website)

### **Note**: You will have to download the dataset on your own.

### **Installation**
1. Clone the repository:
```bash
git clone https://github.com/your-repo/Dys_Locate.git
cd Dys_Locate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset preparation
1. Place the **TORGO** dataset in the same direcotry.
2. Run the Jupyter notebook after replacing the path of the TORGO dataset.


## Usage
### Training the Model
Train the dysarthia detection model using the Jupyter notebook.

### Running the Streamlit app
Launch the Streamlit interface for real-time interaction:
```bash
streamlit run app.py
```

## Results
- **Detection accuracy**: 95%

## Future work:
- Expanding the dataset to include diverse dysarthria cases.
- Improving transcription for severe dysarthric cases.
- Integrating Dys-Locate into assistive devices and mobile applications.

## Contributing
We welcome contributions! Please follow the contribution guidelines to submit issues or pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The TORGO dataset creators for their invaluable resource.
- The open-source community for the tools and libraries used.