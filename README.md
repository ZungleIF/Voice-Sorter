# Voice Sorter

####  Automatically trim and sort videos by the voice of speakers using diarization feature of [pyannote-audio](https://github.com/pyannote/pyannote-audio).

## Dependencies
Python 3.10.0
```
pip install pyannote.audio
pip install torch, torchaudio, torchvision
pip install joblib
```
## Usage

In `VoiceSorter.py`, please edit `video_directory`, `output_directory`, `sample_directory`, and `user_token`.

`video_directory`: Where your videos are located.  
`output_directory`: Where you want the outputs to be.  
`sample_directory`: Where sample clips should be.  

`user_token`: Your Huggingface access token. How to get the access token is described below.

Take each sample clip in .wav format of the voices you want to sort out. Short as 1-2 secs is also fine.   
Rename them to name of each speakers. Now, Run the code on CMD. For example,  
``` 
python "C:\Users\PC\Downloads\VoiceSorter.py"
```

You can also tweak some parameters such as `similarity_threshold`, `segment_min`.

## Getting the access token

1. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions.
2. Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) user conditions.
3. Accept [`pyannote/embedding`](https://huggingface.co/pyannote/embedding) user conditions.
4. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).