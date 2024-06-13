# Voice Sorter

####  Distinguish and sort videos by the voice of speakers using diarization feature of [pyannote-audio](https://github.com/pyannote/pyannote-audio).

## Dependencies
Python 3.10.0
```
pip install pyannote.audio
pip install torch, torchaudio, torchvision
pip install joblib
```
## Usage

In `VoiceSorter.py`, please edit `video_directory`, `output_directory` and `sample_directory`.
`video_directory`: Where your videos are located.
`output_directory`: Where you want the outputs to be.
`sample_directory`: Where sample clips should be.

Take each sample clip in .wav format of the voices you want to sort out. Short as 1-2 secs is also okay. 
Rename them to name of each speakers.
Now, Run the code on CMD. For example,
``` python "C:\Users\PC\Downloads\VoiceSorter.py" ```

You can also tweak some parameters such as `similarity_threshold`, `segment_min`