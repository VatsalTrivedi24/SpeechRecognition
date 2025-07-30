import os
from pydub import AudioSegment
mp3_file_path = 'voice_data/Mr.Warren Buffet/warren buffet.mp3'  
output_folder = 'voice_data/Mr.Warren Buffet' 
os.makedirs(output_folder, exist_ok=True)


audio = AudioSegment.from_mp3(mp3_file_path)
clip_duration = 3 * 1000 

for i in range(1500):
    start_time = i * clip_duration
    end_time = start_time + clip_duration
    if start_time >= len(audio):
        break

    clip = audio[start_time:end_time]

    output_file_path = os.path.join(output_folder, f'clip_{i + 1}.wav')
    clip.export(output_file_path, format='wav')

print("Clips created")