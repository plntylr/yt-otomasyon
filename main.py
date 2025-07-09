from openai import OpenAI
from moviepy.editor import *
import requests
import gradio as gr
import os
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO
from pydub import AudioSegment
import pydub.utils
import shutil
import uuid

# --- Ayarlar ---
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe"
imagemagick_path = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path
AudioSegment.converter = ffmpeg_path
pydub.utils.which = lambda name: ffmpeg_path if name == "ffmpeg" else ffprobe_path if name == "ffprobe" else None

# --- Kontrolller ---
assert os.path.isfile(ffmpeg_path), "ffmpeg bulunamadı"
assert os.path.isfile(ffprobe_path), "ffprobe bulunamadı"
assert os.path.isfile(imagemagick_path), "ImageMagick (magick.exe) bulunamadı"

# --- API anahtarları ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Tarot Yorumu ---
def get_tarot_comment(card_name):
    system_prompt = (
        "Sen bir tarot uzmanısın. Kullanıcılara mistik, derin ve sezgisel tarot yorumları yapıyorsun. "
        "Yorumlarını doğrudan kartın ruhsal anlamına odaklanarak yaz. "
        "Giriş cümleleri kullanma. 'İşte yorumunuz:', 'Elbette', 'Yorum:', 'Bu kart şunu temsil eder' gibi ifadelerle başlama. "
        "Instagram postu için etkileyici, kısa ve ruhsal bir tonda yaz. Kısa paragraflar, güçlü imgeler ve spiritüel dil kullan."
    )

    user_prompt = f"Kart: {card_name}"

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.85,
        max_tokens=300
    )

    raw_text = response.choices[0].message.content
    return clean_response(raw_text)


def clean_response(text):
    # Başlangıçta çıkan tipik ifadeleri temizler
    blacklist = ["Elbette", "İşte yorum", "Yorum:", "Bu kart", "Kart:", "İşte size"]
    text = text.strip()
    for phrase in blacklist:
        if text.lower().startswith(phrase.lower()):
            text = text[len(phrase):].lstrip(": \n")
            break
    return text.strip()

    system_prompt = (
        "Sen bir tarot uzmanısın. Kullanıcılara sezgisel, mistik, etkileyici ve kısa tarot yorumları sunuyorsun. "
        "Instagram için uygun, doğrudan tarot kartının enerjisini ve rehberliğini anlatan bir dille yazıyorsun. "
        "Yanıtlarına 'Elbette', 'İşte yorum', 'Kart şunu temsil eder', 'işte yorumunuz:'gibi ':' kullanmanı gerektirecek ifadelerle başlama. sadece yorumuu yaz "
        "Sadece yorum yap. Giriş veya açıklama yazma."
    )

    user_prompt = f"Kart: {card_name}"

    response = client.chat.completions.create(
        model="gpt-4.1-nano",  # veya "gpt-3.5-turbo" gibi farklı bir model de olabilir
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.85,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

# --- Ses ---
def generate_voice(text, filename=None):
    if filename is None:
        filename = f"voice_{uuid.uuid4().hex}.wav"
    response = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=text
    )
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename
# --- Görsel ---
def generate_sd_image(prompt):
    output_path = f"tarot_{uuid.uuid4().hex}.png"
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    payload = {
        "prompt": prompt,
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 768,
        "sampler_index": "Euler a"
    }
    r = requests.post(url, json=payload).json()
    img_data = base64.b64decode(r['images'][0].split(",", 1)[-1])
    Image.open(BytesIO(img_data)).save(output_path)
    return output_path

# --- Metni Parçala ---
def split_text(text, max_len=250):
    parts, current = [], ""
    for sentence in text.split(". "):
        if len(current) + len(sentence) < max_len:
            current += sentence + ". "
        else:
            parts.append(current.strip())
            current = sentence + ". "
    if current:
        parts.append(current.strip())
    return parts

# --- Sesi Parçala ---
def split_audio(audio_path, text_chunks):
    audio = AudioSegment.from_file(audio_path)
    total_chars = sum(len(t) for t in text_chunks)
    audio_chunks, current = [], 0
    for chunk in text_chunks:
        duration = len(audio) * (len(chunk) / total_chars)
        audio_chunks.append(audio[current:current+duration])
        current += duration
    return audio_chunks

# --- 3 Görsel ---
def generate_three_images(card, index):
    prompts = [
        f"A tarot card illustration of '{card}', mystical, ornate, blue and gold",
        "Elegant mystical Instagram background, blue/gold colors",
        f"Photo-realistic tarot card '{card}' on wooden table with soft lighting"
    ]
    return [generate_sd_image(p) for p in prompts]

# --- Video Klip ---
def create_clip(images, text, audio_segment):
    wav_file = f"chunk_{uuid.uuid4().hex}.wav"
    audio_segment.export(wav_file, format="wav")
    audio = AudioFileClip(wav_file)

    durations = [audio.duration * r for r in [0.4, 0.3, 0.3]]
    clips, start = [], 0
    for i, img in enumerate(images):
        img_clip = ImageClip(img).set_duration(durations[i])
        txt_clip = TextClip(text, fontsize=24, color='white', font="Georgia-Bold",
                            method='caption', size=(img_clip.w - 80, None),
                            bg_color='rgba(0,0,0,0.5)').set_position("bottom").set_duration(durations[i])
        sub_audio = audio.subclip(start, start + durations[i])
        clips.append(CompositeVideoClip([img_clip, txt_clip]).set_audio(sub_audio))
        start += durations[i]
    return concatenate_videoclips(clips)

# --- Pipeline ---
def tarot_pipeline(card_name):
    comment = get_tarot_comment(card_name)
    voice_path = generate_voice(comment)
    chunks = split_text(comment)
    audio_chunks = split_audio(voice_path, chunks)
    clips = []
    for i, chunk in enumerate(chunks):
        imgs = generate_three_images(card_name, i)
        clips.append(create_clip(imgs, chunk, audio_chunks[i]))
    final = concatenate_videoclips(clips)
    output = "tarot_reels.mp4"
    final.write_videofile(output, fps=24)
    return comment, output

# --- Arayüz ---
with gr.Blocks() as demo:
    gr.Markdown("## Tarot Reels Video Üretici")
    card = gr.Textbox(label="Kart Adı", placeholder="The High Priestess")
    comment = gr.Textbox(label="Tarot Yorumu")
    video = gr.Video(label="Video")
    btn = gr.Button("Üret")
    btn.click(fn=tarot_pipeline, inputs=card, outputs=[comment, video])
demo.launch(server_port=7862)
