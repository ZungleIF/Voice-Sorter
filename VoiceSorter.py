import os
import ffmpeg
from tqdm import tqdm

import torchaudio
import torch

from pyannote.audio import Pipeline
from pyannote.audio import Inference
from pyannote.core import Segment

from joblib import Parallel, delayed

import pickle
def save_results(diarization, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(diarization, f)

def load_results(input_path):
    with open(input_path, 'rb') as f:
        diarization = pickle.load(f)
    return diarization

def process_video_file(video_filename):    
    video_file = os.path.join(video_directory, video_filename)
    audio_file = os.path.join(output_directory, f"{os.path.splitext(video_filename)[0]}.wav")
    result_file = os.path.join(output_directory, f"{os.path.splitext(video_filename)[0]}_diarization.pkl")

    
    # Look for a result_file.
    if os.path.exists(result_file):
        diarization = load_results(result_file)
    # Extract audio of the video file.
    else:
        try:
            ffmpeg.input(video_file).output(audio_file).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print(f"Error occurred while processing {video_file}:")
            print(e.stderr.decode())

        print(f"{video_filename} in the pipeline")
        diarization = diarization_pipeline(audio_file)
        save_results(diarization, result_file)

    # Load the audio.
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.to(device)

    total_segments = len(list(diarization.itertracks(yield_label=True)))
    progress_bar = tqdm(total=total_segments, desc=f"Processing {video_filename}", unit="segment")


    for turn, _, _ in diarization.itertracks(yield_label=True):
        progress_bar.update(1)

        start_time = turn.start
        end_time = turn.end

        # Setting the minimum length of a segment.
        if end_time - start_time >= segment_min:
            segment = Segment(start_time, end_time)

            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)

            # Trim the audio
            speaker_waveform = waveform[:, start_sample:end_sample]

            segment_embedding = embedding_model({"waveform": speaker_waveform, "sample_rate": sample_rate})
            segment_embedding = torch.tensor(segment_embedding.data)

            best_speaker = None
            best_similarity = -float('inf')
            for speaker, embedding in speaker_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(segment_embedding, embedding, dim=-1).mean()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker

            if best_similarity < similarity_threshold:
                # print(f"Skipping segment from {start_time} to {end_time} due to low similarity ({best_similarity})")
                continue

            # Make directories for each speaker.
            speaker_dir = os.path.join(output_directory, best_speaker)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)

            speaker_file_path = os.path.join(speaker_dir, f"{os.path.splitext(video_filename)[0]}_{start_sample}_{end_sample}.wav")
            torchaudio.save(speaker_file_path, speaker_waveform.cpu(), sample_rate)

    progress_bar.close()

similarity_threshold = 0.5
segment_min = 0.75

video_directory = "path/to/video_directory"
output_directory = "path/to/output_directory"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)



# Use GPU if CUDA is available. (Needs to install pytorch accordingly.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pipeline for diarization.
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="YOUR_TOKEN_HERE")
diarization_pipeline = diarization_pipeline.to(device)

# Load model for embedding.
embedding_model = Inference("pyannote/embedding", use_auth_token="YOUR_TOKEN_HERE", device=device)


# Sample audio files for each speakers.
sample_directory = "path/to/sample_directory"
speaker_embeddings = {}

# Generating Embeddence
for sample_filename in os.listdir(sample_directory):
    if sample_filename.endswith(".wav"):
        speaker = os.path.splitext(sample_filename)[0]
        sample_file = os.path.join(sample_directory, sample_filename)
        embedding = embedding_model(sample_file)
        # Converting SlidingWindowFeature to Tensor
        speaker_embeddings[speaker] = torch.tensor(embedding.data)


video_filenames = [f for f in os.listdir(video_directory) if f.endswith(".mkv") or f.endswith(".mp4")]

Parallel(n_jobs=4, backend='threading')(
    delayed(process_video_file)(video_filename) for video_filename in video_filenames
)

print("All video files processed and saved.")