import cv2
import gc
import json
import os
import sys
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import click
import numpy as np
import pytesseract
import requests
from PIL import ImageDraw, ImageFont, Image
from skimage.metrics import structural_similarity as ssim
from moviepy.editor import VideoFileClip


# Global constants
MAX_CACHE_SIZE = 100

# Global variables
lock = threading.Lock()
translation_cache = {}
cache_dtype_shape = {}
previous_translations = {}



def calculate_remaining_time(start_time, total_frames, current_frame):
    # Calculate the remaining time for a process

    elapsed_time = time.time() - start_time
    remaining_frames = total_frames - current_frame
    estimated_time_left = (elapsed_time / current_frame) * remaining_frames
    
    hours, rem = divmod(estimated_time_left, 3600)
    minutes, seconds = divmod(rem, 60)
    
    return int(hours), int(minutes), int(seconds)


def convert_np_to_bytes(array):
    # Convert numpy array to bytes

    return array.tobytes()

def convert_bytes_to_np(bytes_data, dtype, shape):
    # Convert bytes back to numpy array

    return np.frombuffer(bytes_data, dtype=dtype).reshape(shape)

def remove_non_alphabetic(text):
    # Remove non-alphabetic characters from text

    cleaned_text = re.sub(r"[^a-zA-Z !?.]", "", text)
    return cleaned_text

def adjust_image_size(image, width=640, height=480):
    # Adjust the size of an image

    resized_image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return resized_image


def reset_cache():
    # Reset the translation cache

    global translation_cache, cache_dtype_shape, previous_translations
    with lock:
        translation_cache = {}
        cache_dtype_shape = {}
        previous_translations = {}

@lru_cache(maxsize=1000)
def translate_with_google(text, target_language):
    # Translate text using Google Translate
    base_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={}&dt=t&q={}"
    url = base_url.format(target_language, text)
    response = requests.get(url)
    result = json.loads(response.text)
    if result[0]:
        return result[0][0][0]
    else:
        return ''


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def translate_image_text(image_path, lang='en', to_lang='ru'):
    global translation_cache, cache_dtype_shape, previous_translations

    img = cv2.imread(image_path)
    img = adjust_image_size(img, width=720, height=480)  # Resize image to 640x480
    img = cv2.convertScaleAbs(img, alpha=4.0, beta=0)
   
    height, width, _ = img.shape

    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height * .89), int(width)
    img[start_row:end_row , start_col:end_col] = cv2.GaussianBlur(img[start_row:end_row , start_col:end_col], (99, 99), 30)
    
    start_row, start_col = int(height * .9), int(width * .9)
    end_row, end_col = int(height), int(width)
    img[start_row:end_row , start_col:end_col] = cv2.GaussianBlur(img[start_row:end_row , start_col:end_col], (99, 99), 30)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    height, width = thresh.shape
    start_row, start_col = int(height * .6), int(0)
    end_row, end_col = int(height), int(width)
    cropped_img = thresh[start_row:end_row , start_col:end_col]

    # Check if the frame is already in the cache
    with lock:
        for cached_img_bytes, cached_translation in translation_cache.items():
            cached_img = convert_bytes_to_np(cached_img_bytes, cache_dtype_shape[cached_img_bytes][0], cache_dtype_shape[cached_img_bytes][1])
            similarity = ssim(cropped_img, cached_img)
            if similarity > 0.95:
                print(f'Using cached translation: {cached_translation}')
                return cached_translation

    text = remove_non_alphabetic(pytesseract.image_to_string(cropped_img, lang='eng'))
    # If the text has already been translated, use the saved translation
    if text in previous_translations:
        print(f'Using previous translation: {previous_translations[text]}')
        return previous_translations[text]

    print(f'subtitle text: {text}')
    
    translation = translate_with_google(text, to_lang)
    print(f'translate text: {translation}')

    # Save the translation result in the cache
    with lock:
        cropped_img_bytes = convert_np_to_bytes(cropped_img)
        translation_cache[cropped_img_bytes] = translation
        cache_dtype_shape[cropped_img_bytes] = (cropped_img.dtype, cropped_img.shape)
        previous_translations[text] = translation

    return translation




def add_text_to_image(image_path, text):
    # Load the image using PIL
    img_pil = Image.open(image_path)
    draw = ImageDraw.Draw(img_pil)

    # Load a font that supports Cyrillic
    font = ImageFont.truetype('arial.ttf', 50)

    # Calculate the size of the text
    bbox = draw.textbbox((10, int(img_pil.size[1]*.9)), text, font=font)
    text_width = (bbox[2] - bbox[0]) 
    text_height = bbox[3] - bbox[1]

    center_x = (img_pil.size[0] - text_width) / 2
    center_y = (img_pil.size[1] - text_height) - 20

    # Create a rectangle around the text
    rectangle_start = (center_x - 10, center_y-10)
    rectangle_end = (center_x+10 + text_width, center_y+10 + text_height)
    draw.rectangle([rectangle_start, rectangle_end], fill='black')

    # Overlay the text on the image
    draw.text((center_x, center_y), text, font=font, fill='white')

    # Convert the PIL image back to an OpenCV image
    img = np.array(img_pil)

    # Convert the colors back to RGB order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save the image
    #cv2.imwrite('translated_image.jpg', img)
    return img


def extract_video_frames(video_path, frames_dir):
    # Extract frames from a video

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 100 == 0:
            print(f"Extracting frame {count} of {total_frames}")
        cv2.imwrite(os.path.join(frames_dir, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1

def process_video_frame(frame_file, frames_dir, translated_frames_dir, lang='en', to_lang='ru'):
    # Check if the translated frame already exists
    translated_frame_path = os.path.join(translated_frames_dir, frame_file)
    if os.path.exists(translated_frame_path):
        print(f"Frame {frame_file} has already been translated. Skipping.")
        return

    frame_path = os.path.join(frames_dir, frame_file)
    text = translate_image_text(frame_path, lang, to_lang)
    img = add_text_to_image(frame_path, text)
    cv2.imwrite(translated_frame_path, img)

def translate_video_frames(frames_dir, translated_frames_dir):
    count_i = 0
    start_time = time.time()
    frame_files = os.listdir(frames_dir)
    total_frames = len(frame_files)
    
    # Split the list of files into chunks of 100 files
    chunks = [frame_files[i:i + 100] for i in range(0, len(frame_files), 100)]
    
    for chunk in chunks:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_video_frame, chunk, [frames_dir]*len(chunk), [translated_frames_dir]*len(chunk))
            for i, _ in enumerate(results, 1):
                count_i += 1
                if i % 15 == 0:
                    hours_left, minutes_left, seconds_left = calculate_remaining_time(start_time, total_frames, count_i)
                    print(f"{count_i}/{total_frames} frames have been processed. There are about {hours_left}h {minutes_left}m {seconds_left}s left.")



def translate_frames_efficiently(frames_dir, translated_frames_dir, create_subtitle=None): #extreme skipping frame
    frame_count = 0
    start_time = time.time()
    frame_files = os.listdir(frames_dir)
    frame_files = sorted(frame_files, key=lambda name: int(name[5:-4]))
    
    total_frames = len(frame_files)

    translated_subtitles = []

    # Split the list of files into chunks of 30 files
    chunks = [frame_files[i:i + 30] for i in range(0, len(frame_files), 30)]
    
    for chunk in chunks:
        # Translate text from the 15th frame in each block
        frame_path = os.path.join(frames_dir, chunk[14])

        text_15 = None
        if os.path.exists(os.path.join(translated_frames_dir, chunk[14])):
            print(f'Skip AP frame.')
        else:
            text_15 = translate_image_text(frame_path, lang='en', to_lang='ru')
            if create_subtitle is not None:
                translated_subtitles.append((frame_count, text_15))
            
        for i, frame_file in enumerate(chunk):
            frame_count += 1
            # For all 30 frames in the block, use the translation from the 15th frame

            #check
            translated_frame_path = os.path.join(translated_frames_dir, frame_file)
            if os.path.exists(translated_frame_path):
                print(f"Frame {frame_file} has already been translated. Skipping.")
                continue
            if text_15 is None:
                text_15 = translate_image_text(frame_path, lang='en', to_lang='ru')

            text = text_15

            img = add_text_to_image(os.path.join(frames_dir, frame_file), text)
            cv2.imwrite(os.path.join(translated_frames_dir, frame_file), img)
                
            if i % 15 == 0:
                hours_left, minutes_left, seconds_left = calculate_remaining_time(start_time, total_frames, frame_count)
                print(f"{frame_count}/{total_frames} frames have been processed. There are about {hours_left}h {minutes_left}m {seconds_left}s left.")

        if create_subtitle is not None:
            with open(create_subtitle, 'w', encoding='utf-8') as f:
                for i, (frame_number, subtitle) in enumerate(translated_subtitles):
                    start_time_sub = frame_number / 30  # Start time of the subtitle in seconds
                    end_time_sub = (frame_number + 15) / 30  # End time of the subtitle in seconds
                    f.write(f"{i+1}\n{start_time_sub:.3f} --> {end_time_sub:.3f}\n{subtitle}\n\n")

def load_frames(translated_frames_dir, frame_files):
    return [cv2.imread(os.path.join(translated_frames_dir, frame_file)) for frame_file in frame_files]

def compile_video(translated_frames_dir, output_video_path):
    BATCH_RATE = 500

    start_time = time.time()
    frame_files = os.listdir(translated_frames_dir)
    frame_files = sorted(frame_files, key=lambda name: int(name[5:-4]))
    total_frames = len(frame_files)
    frame = cv2.imread(os.path.join(translated_frames_dir, frame_files[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, total_frames, BATCH_RATE):
            batch_files = frame_files[i:i+BATCH_RATE]
            frames = load_frames(translated_frames_dir, batch_files)
            for j, frame in enumerate(frames):
                if (i+j+1) % 100 == 0:
                    hours_left, minutes_left, seconds_left = calculate_remaining_time(start_time, total_frames, i+j+1)
                    print(f"{i+j+1} frames have been processed. There are about {hours_left}h {minutes_left}m {seconds_left}s left.")
                if (i+j) % 100 == 0:
                    print(f"Assembling frame {i+j} of {total_frames}")
                video.write(frame)

            del frames
            gc.collect()
    video.release()


def assemble_video_with_sound(video_with_audio, video_path):
    output_file = video_path[0:-4]+'_assembled.mp4'
    video_with_audio = VideoFileClip(video_with_audio)
    audio = video_with_audio.audio

    video_path = VideoFileClip(video_path)

    final_video = video_path.set_audio(audio)
    final_video.write_videofile(output_file,codec='libx264')

    print(f'{output_file} - [OK]')




###############################CLOCK INTERFACE#############################

@click.group()
def cli():
    pass

@click.command()
@click.option('--video', prompt='Video file', help='The path to the video file.')
@click.option('--output', prompt='Output directory', help='The directory to save the frames.')
def extract(video, output):
    extract_video_frames(video, output)

@click.command()
@click.option('--frames', prompt='Frames directory', help='The directory containing the frames.')
@click.option('--output', prompt='Output directory', help='The directory to save the translated frames.')
@click.option('--mode', prompt='Mode [1] default, [2] fast', help='The mode to use for translation.')
@click.option('--subtitle', prompt='Subtitle file (optional)', help='The file to save the subtitles.', default=None)
def translate(frames, output, mode, subtitle):
    if mode == '1':
        translate_video_frames(frames, output)
    elif mode == '2':
        translate_frames_efficiently(frames, output, subtitle)

@click.command()
@click.option('--frames', prompt='Frames directory', help='The directory containing the frames.')
@click.option('--output', prompt='Output video file', help='The path to the output video file.')
def assemble(frames, output):
    compile_video(frames, output)

@click.command()
@click.option('--audio', prompt='Video file with audio', help='The path to the video file with audio.')
@click.option('--video', prompt='Video file without audio', help='The path to the video file without audio.')
def sound(audio, video):
    assemble_video_with_sound(audio, video)

cli.add_command(extract)
cli.add_command(translate)
cli.add_command(assemble)
cli.add_command(sound)


###########################################################################

def main():
    print('--- menu --- \n1 - extract frame\n2 - translate frame\n3 - assemble video\n4 - assemble audio in video\nany - exit')
    select = input('select: ')
    inp = dir_out = None
    if select== '1':
        inp, dir_out = input('video: '), input('dir out: ')
        extract_video_frames(inp, dir_out)
    elif select == '2':
        sub_file = None
        type_mode = input('select mode [1] default, [2] fast: ')
        inp, dir_out = input('dir frames: '), input('dir out: ')
        if type_mode == '1':
            translate_video_frames(inp, dir_out)
        elif type_mode == '2':
            if input('create subtitle file? Y/n: ').lower() == 'y':
                sub_file = input('filename: ')
            translate_frames_efficiently(inp, dir_out, sub_file)
        else:
            print('error argument')
            return None
    elif select == '3':
        inp, dir_out = input('frames dir: '), input('video out: ')
        compile_video(inp, dir_out)
    elif select == '4':
        inp, dir_out = input('video_with_audio: '), input('video_without_audio: ')
        assemble_video_with_sound(inp, dir_out)
    else:
        print('exiting...')





if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        cli()
