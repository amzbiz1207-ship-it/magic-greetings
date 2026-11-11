"""Image composition uses only the uploaded pixels, Pillow and FFmpeg."""
from functools import lru_cache
from pathlib import Path
import io
import json
import math
import random
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
W, H = 1080, 1350
MAX_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
TEMPLATES = json.loads((ROOT / "templates.json").read_text())
THEMES = {item["id"]: item for item in TEMPLATES}
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


class PhotoError(ValueError):
    pass


def load_photo(data):
    if len(data) > MAX_BYTES:
        raise PhotoError("file_size")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise PhotoError("format")
            if source.width * source.height > 20_000_000:
                raise PhotoError("pixels")
            source.load()
            photo = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", photo.size, "white")
            background.alpha_composite(photo)
            return background.convert("RGB")
    except PhotoError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PhotoError("pixels") from None
    except Exception:
        raise PhotoError("format") from None


@lru_cache(maxsize=1)
def face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Face detector unavailable")
    return detector


def inspect(photo, category="universal"):
    """Conservative initial heuristics. Manual preview remains essential."""
    issues = []
    if min(photo.size) < 600:
        issues.append("resolution")
    scale = min(1, 1000 / max(photo.size))
    reduced = photo.resize((round(photo.width * scale), round(photo.height * scale)))
    gray = cv2.cvtColor(np.array(reduced), cv2.COLOR_RGB2GRAY)
    if gray.mean() < 35 or gray.mean() > 245:
        issues.append("light")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 25:
        issues.append("blur")
    portrait = category in {"women", "men", "children", "grandchildren", "grandparents"}
    group = category in {"family", "couples"}
    faces = []
    if portrait or group:
        faces = face_detector().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(45, 45)
        )
        if len(faces) == 0:
            issues.append("face_missing")
        elif portrait and len(faces) != 1:
            issues.append("face_count")
        elif min(min(w, h) / scale for x, y, w, h in faces) < 100:
            issues.append("face_small")
        elif any(cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var() < 20
                 for x, y, w, h in faces):
            issues.append("face_blur")
    return issues, len(faces)


@lru_cache(maxsize=96)
def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / filename), size)


def wrap(draw, text, selected_font, max_width):
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=selected_font) <= max_width:
                line = candidate
                continue
            if line:
                lines.append(line)
            line = ""
            for character in word:
                if draw.textlength(line + character, font=selected_font) > max_width:
                    lines.append(line)
                    line = ""
                line += character
        lines.append(line)
    return lines


def text_block(image, text, box, maximum, minimum, color, bold=False):
    draw = ImageDraw.Draw(image)
    x, y, width, height = box
    for size in range(maximum, minimum - 1, -1):
        selected_font = font(size, bold)
        lines = wrap(draw, text, selected_font, width)
        line_height = size + 10
        if len(lines) * line_height <= height:
            break
    else:
        raise PhotoError("text_long")
    top = y + (height - len(lines) * line_height) / 2
    for index, line in enumerate(lines):
        draw.text((x + width / 2, top + index * line_height), line,
                  font=selected_font, fill=color, anchor="mt")


@lru_cache(maxsize=8)
def background(template_id):
    template = THEMES[template_id]
    with Image.open(ROOT / template["background"]) as source:
        return ImageOps.fit(source.convert("RGB"), (W, H))


def photo_placement(photo, template_id, size=100, horizontal=50, vertical=50):
    x, y, width, height = THEMES[template_id]["photo"]
    target = (max(1, round(width * size / 100)), max(1, round(height * size / 100)))
    fitted = ImageOps.contain(photo.convert("RGB"), target, Image.Resampling.LANCZOS)
    left = x + round((width - fitted.width) * horizontal / 100)
    top = y + round((height - fitted.height) * vertical / 100)
    return fitted, left, top


def render(photo, theme, title, message, name="", size=100, horizontal=50, vertical=50):
    template = THEMES[theme]
    card = background(theme).copy()
    draw = ImageDraw.Draw(card)
    x, y, width, height = template["photo"]
    draw.rounded_rectangle((x-6, y-6, x+width+6, y+height+6),
                           radius=6, fill=template["accent"])
    draw.rectangle((x, y, x+width, y+height), fill=template["mat"])
    fitted, left, top = photo_placement(photo, theme, size, horizontal, vertical)
    card.paste(fitted, (left, top))
    # Panels ensure readable, independent text on every artwork.
    for key in ("heading", "text"):
        tx, ty, tw, th = template[key]
        draw.rounded_rectangle((tx-14, ty-8, tx+tw+14, ty+th+8),
                               radius=18, fill=template["mat"])
    text_block(card, title, template["heading"], 54, 26, template["accent"], True)
    tx, ty, tw, th = template["text"]
    if name:
        text_block(card, name, (tx, ty, tw, 90), 40, 20, template["accent"], True)
        ty += 100
        th -= 100
    text_block(card, message, (tx, ty, tw, th), 36, 22, template["ink"])
    return card


def encode_image(card, image_format):
    output = io.BytesIO()
    card.save(output, format=image_format, **({"quality": 95} if image_format == "JPEG" else {}))
    return output.getvalue()


def animate(card, template_id="gold"):
    template = THEMES[template_id]
    rng = random.Random(75)
    particles = [(rng.randrange(W), rng.randrange(H), rng.uniform(20, 65), rng.randrange(2, 6))
                 for _ in range(70)]
    with tempfile.TemporaryDirectory(prefix="magic-greetings-") as directory:
        output = Path(directory) / "greeting.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-an", "-c:v", "libx264",
            "-threads", "2", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output)
        ]
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
            try:
                for frame_number in range(144):
                    frame = card.copy()
                    time = frame_number / 24
                    if time < 0.8:
                        for key in ("heading", "text"):
                            tx, ty, tw, th = template[key]
                            bounds = (tx, ty, tx+tw, ty+th)
                            blank = Image.new("RGB", (tw, th), template["mat"])
                            frame.paste(Image.blend(blank, card.crop(bounds), time / 0.8), (tx, ty))
                    draw = ImageDraw.Draw(frame)
                    for px, py, speed, radius in particles:
                        yy = (py + speed * time) % H
                        xx = px + 10 * math.sin(time + px)
                        # Restrict all effects to the outer margins.
                        if 90 < xx < 990:
                            continue
                        light = round(205 + 50 * math.sin(time * 3 + py) ** 2)
                        draw.ellipse((xx-radius, yy-radius, xx+radius, yy+radius),
                                     fill=(light, light, min(255, light+10)))
                        if frame_number % 24 < 12:
                            draw.line((xx-5, yy, xx+5, yy), fill=template["accent"])
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                if process.wait(timeout=90) != 0:
                    raise RuntimeError("Video encoding failed")
                return output.read_bytes()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
