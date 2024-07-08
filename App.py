import os
import tkinter as tk
from tkinter import ttk, PhotoImage
import speech_recognition as sr
import torch
import torchaudio
import tempfile
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from PIL import Image, ImageTk

# Get the path to the model directory relative to the executable
base_path = os.path.dirname(__file__)
model_directory = os.path.join(base_path, 'model')

# Load the model and processor from the local directory
model = Wav2Vec2ForCTC.from_pretrained(model_directory)
processor = Wav2Vec2Processor.from_pretrained(model_directory)

def recognize_speech(audio):
    try:
        # Save the audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio.get_wav_data())

        # Load the temporary audio file using PySoundFile
        speech_array, sampling_rate = sf.read(temp_audio_file.name)

        # Convert to Torch tensor and cast to Double
        speech_tensor = torch.tensor(speech_array, dtype=torch.double)

        # Resample if needed
        if sampling_rate != 16_000:  # Adjust to model's expected sampling rate
            resampler = torchaudio.transforms.Resample(sampling_rate, 16_000, dtype=torch.float64)  # Set type to double precision
            # Cast to Double before resampling
            speech_tensor = speech_tensor.double()
            speech_tensor = resampler(speech_tensor).squeeze()

        # Preprocess the audio
        inputs = processor(speech_tensor, sampling_rate=16_000, return_tensors="pt", padding=True)

        # Run the model
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # Decode the prediction
        predicted_ids = torch.argmax(logits, dim=-1)
        recognized_text = processor.batch_decode(predicted_ids)[0]

        print("You said:", recognized_text)

        # Update the text box with the recognized text
        text_entry.delete(1.0, tk.END)
        text_entry.insert(tk.END, recognized_text)

        # Change the button back to blue when transcription is done
        button.configure(style='Blue.TButton')
    except Exception as e:
        print("Error during speech recognition:", e)

def start_countdown_and_transcribe(audio):
    countdown_time = 3  # Set the countdown time to 60 seconds

    def countdown():
        nonlocal countdown_time
        if countdown_time > 0:
            seconds = countdown_time
            timer_label.config(text=f"{seconds:02}")
            countdown_time -= 1
            root.after(1000, countdown)
        else:
            # Reset the timer display to "00:00"
            timer_label.config(text="00")
            # Transcribe the captured audio
            recognize_speech(audio)

    countdown()

def capture_audio():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
            # Change the button to red when listening
            button.configure(style='Red.TButton')
            start_countdown_and_transcribe(audio)
    except Exception as e:
        print("Error during audio capture:", e)

def on_hover(event):
    button.configure(style='Red.TButton')

def on_leave(event):
    button.configure(style='Blue.TButton')

def on_click(event):
    button.configure(style='Red.TButton')

def change_bg_color():
    current_color = root.cget("bg")
    try:
        current_color_value = int(current_color[1:], 16)
        new_color_value = current_color_value + 1
        new_color = "#" + hex(new_color_value)[2:].zfill(6)  # Convert back to hexadecimal format
    except ValueError:
        new_color = "#1e90ff"  # Default color if extraction fails
    root.configure(bg=new_color)
    root.after(100, change_bg_color)  # Repeat after 100 milliseconds

# Create main application window
root = tk.Tk()
title = "Speech To Text Luganda Desktop Version"
root.wm_title(title)

# Set window title
title_text = "Speech To Text Luganda Desktop Version"
root.title(title_text)

# Set window size and background color
root.geometry("950x600")

# Start the function to change background color
change_bg_color()

# Create a frame
frame = ttk.Frame(root)
frame.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))

# Create custom styles for the button
style = ttk.Style()
style.configure('Red.TButton', font=('TkDefaultFont', 14, 'bold'), foreground='white', background='red')
style.configure('Blue.TButton', font=('TkDefaultFont', 14, 'bold'), foreground='blue')

# Load the image for the button using Pillow
image_path = "C:/Users/jkavuma/Desktop/icon.png"  # Replace with the correct path to your image
if os.path.exists(image_path):
    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((100, 100), Image.LANCZOS)  # Resize the image to fit the button
    image = ImageTk.PhotoImage(pil_image)
else:
    image = None

# Create a text entry box with a larger font size
text_entry = tk.Text(frame, height=30, width=10, font=("Calibri", 40))  # Adjust the height and width as needed
text_entry.grid(column=0, row=0, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))

# Create a button with an image
if image:
    button = ttk.Button(frame, text="Nyiga Oyogele", command=capture_audio, style='Blue.TButton', image=image,  compound=tk.LEFT, width=50)
else:
    button = ttk.Button(frame, text="Nyiga Oyogele", command=capture_audio, style='Blue.TButton',  compound=tk.LEFT, width=50)

button.grid(column=0, row=1, pady=10, sticky=(tk.W, tk.E))

# Bind hover and click events to the button
button.bind("<Enter>", on_hover)
button.bind("<Leave>", on_leave)
button.bind("<Button-1>", on_click)

# Create a label to display the timer
timer_label = tk.Label(frame, font=("Helvetica", 24), fg="#0000ff")
timer_label.grid(column=0, row=2, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))

# Add row and column weights to make the text entry box expand with the window
frame.rowconfigure(0, weight=1)  # The text box row
frame.columnconfigure(0, weight=1)
frame.rowconfigure(1, weight=0)  # The button row
frame.rowconfigure(2, weight=0)  # The timer row

# Make the frame expand with the window
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Run the application
root.mainloop()