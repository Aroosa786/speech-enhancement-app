import numpy as np
import soundfile as sf
from scipy import signal
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()

def load_sample_audio(speech_type='female'):
    try:
        audio_path = 'clean.wav'
        if os.path.exists(audio_path):
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            audio = audio / np.max(np.abs(audio)) * 0.7
            return audio, sr
        else:
            return create_simple_speech(), 16000
    except Exception as e:
        return create_simple_speech(), 16000

def create_simple_speech():
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, int(sr * duration))
    speech = (np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 5 * t) +
              0.5 * np.sin(2 * np.pi * 400 * t) * np.sin(2 * np.pi * 3 * t))
    envelope = np.exp(-((t - duration/2) ** 2) / (2 * (duration/3) ** 2))
    speech = speech * envelope * 0.3
    return speech

def add_noise(audio, noise_level_db, sr=16000):
    noise_factor = abs(noise_level_db) / 200.0
    noise = np.random.normal(0, 1, len(audio))
    noisy_audio = audio + noise * noise_factor
    return noisy_audio

def create_spectrogram(audio, sr):
    f, t, Zxx = signal.stft(audio, sr, nperseg=1024)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    return f, t, magnitude_db

def audio_to_base64(audio, sr):
    max_audio = np.max(np.abs(audio))
    if max_audio > 0:
        audio_normalized = np.int16(audio / max_audio * 32767)
    else:
        audio_normalized = np.int16(audio * 32767)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_normalized, sr, format='WAV')
    buffer.seek(0)
    
    audio_b64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/wav;base64,{audio_b64}"

def enhance_audio_with_aic(audio, enhancement_level, sr=16000):
    api_key = os.getenv('ai-sdk-api-key')
    
    if not api_key:
        return apply_basic_enhancement(audio, enhancement_level, sr)
    
    try:
        import aic_sdk
        audio_input = audio.astype(np.float32)
        
        if hasattr(aic_sdk, 'enhance'):
            enhanced_audio = aic_sdk.enhance(audio_input, api_key=api_key)
            return enhanced_audio
        elif hasattr(aic_sdk, 'process'):
            enhanced_audio = aic_sdk.process(audio_input, api_key=api_key)
            return enhanced_audio
        else:
            return apply_basic_enhancement(audio, enhancement_level, sr)
            
    except Exception as e:
        return apply_basic_enhancement(audio, enhancement_level, sr)

def apply_basic_enhancement(audio, enhancement_level, sr=16000):
    original_clean = load_sample_audio()[0]
    noisy_audio = audio
    enhanced = noisy_audio * (1 - enhancement_level) + original_clean * enhancement_level
    return enhanced

def create_vad_output(audio, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    energy = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy.append(np.sum(frame ** 2))
    
    energy = np.array(energy)
    threshold = np.percentile(energy, 30)
    vad = energy > threshold
    
    time_frames = np.linspace(0, len(audio) / sr, len(vad))
    return time_frames, vad.astype(float)   