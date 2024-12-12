# import streamlit as st
# import librosa

# st.write("Deep Learning Final Project")

# upload = st.file_uploader("Upload a file", type = ["mp3", "wav", "m4a"])
# if upload:
#     #process the file
#     pass


#line of code to get the requirements txtx
#pip freeze>requirements.txt


import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
import librosa

# Load the OpenAI WhisperProcessor (used for preprocessing audio)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load your fine-tuned model and tokenizer

model_path = "./checkpoint-1400"  # Replace with your fine-tuned model path
model = WhisperForConditionalGeneration.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path)

# Function to load and preprocess the audio file
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

# Streamlit UI for file upload
st.title("Fine-Tuned Whisper Transcription")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a","ogg"])

if uploaded_file:
    # Show file details
    st.write(f"Processing file: {uploaded_file.name}")
    
    # Save the uploaded file temporarily
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and preprocess the audio file
    audio_data = load_audio("temp_audio_file", target_sr=16000)
    
    # Process the audio using the WhisperProcessor
    inputs = processor(audio_data, return_tensors="pt", sampling_rate=16000)
    
    # Move model and inputs to the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Generate transcription with your fine-tuned model
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    
    # Decode the predicted transcription using your tokenizer
    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    
    # Output the transcription
    st.write("Transcription:")
    st.text_area("Transcribed Text", transcription[0], height=200)

