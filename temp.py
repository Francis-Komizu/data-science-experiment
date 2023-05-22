import os
import shutil

filelist_path = 'filelist.txt'
src_dir = 'wavs/genshin'
dst_dir = 'wavs/test_data'

wav_files = os.listdir(src_dir)

new_lines = {}
with open(filelist_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        file_name, speaker_num = line.split('|')
        file_name = file_name.replace('DUMMY/', '')
        speaker_num = speaker_num.strip('\n')
        new_lines.update({file_name: speaker_num})


for wav_file in wav_files:
    src_path = os.path.join(src_dir, wav_file)
    speaker_num = new_lines[wav_file]
    dst_file = f'{speaker_num}-{wav_file}'
    dst_path = os.path.join('wavs/test_data', dst_file)
    shutil.copy(src_path, dst_path)


