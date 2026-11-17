import base64
from io import BytesIO
import os
import logging

import cv2
import numpy as np
import requests
from datetime import datetime
import hashlib
import hmac
import json

import streamlit as st
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from engine import (
    FONT_DIR, ROOT, TEMPLATES, THEMES, PhotoError, animate,
    encode_image, face_detector, inspect, load_photo, render, text_block, wrap,
)

def edit_scene(image, editable, prompt, api_key):
    files = {"image": ("scene.png", encode_image(image, "PNG"), "image/png")}
    if editable is not None:
        mask = Image.new("RGBA", image.size, (0, 0, 0, 255))
        mask.putalpha(ImageOps.invert(editable))
        files["mask"] = ("edit-mask.png", encode_image(mask, "PNG"), "image/png")
    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        data={
            "model": os.environ.get("MAGIC_GREETING_IMAGE_MODEL", "gpt-image-2.6-sunburst"),
            "prompt": prompt,
            "size": f"{image.width}x{image.height}",
        },
        files=files,
        timeout=(10, 240),
    )
    response.raise_for_status()
    result = base64.b64decode(response.json()["data"][0]["b64_json"], validate=True)
    with Image.open(BytesIO(result)) as candidate:
        if candidate.size != image.size:
            raise ValueError("Unexpected image dimensions")
        edited = candidate.convert("RGB")
        if editable is None:
            return edited
        # Keep protected pixels from the original even if the provider altered them.
        return Image.composite(edited, image, editable)


def scene_input(photo, market=False):
    ratio = photo.width / photo.height
    size = (1536, 1024) if ratio > 1.2 else (1024, 1536) if ratio < .83 else (1024, 1024)
    if market:
        size = (1536, 1024)
    fitted = ImageOps.contain(photo, tuple(round(side * .82) for side in size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#263229")
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def scene_editable(size, faces, selected=None):
    editable = Image.new("L", size, 255 if selected is None else 0)
    draw = ImageDraw.Draw(editable)
    for index, (x, y, w, h) in enumerate(faces):
        if selected is None or selected == index:
            draw.ellipse((x - w*.14, y - h*.24, x + w*1.14, y + h*1.12),
                         fill=0 if selected is None else 255)
            if selected is None:
                draw.rectangle((x, y, x + w, y + h), fill=0)
    if selected is not None:
        editable = editable.filter(ImageFilter.GaussianBlur(5))
        draw = ImageDraw.Draw(editable)
        for index, (x, y, w, h) in enumerate(faces):
            if index != selected:
                draw.rectangle((x - 10, y - 12, x + w + 10, y + h + 12), fill=0)
    return editable


def detect_scene_faces(image):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return sorted(tuple(map(int, face)) for face in face_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55)
    ))


def scene_editor(photo, digest, scene, lang, title, message, name, api_key, t):
    ru = lang == "ru"
    market = scene == "market"
    digest = f"{digest}_{scene}"
    if st.session_state.get("scene_digest") != digest:
        canvas = scene_input(photo, market)
        faces = detect_scene_faces(canvas)
        issues, _ = inspect(photo, "family")
        saved = st.session_state.setdefault("scene_saved", {}).get(digest, {})
        st.session_state.update(scene_digest=digest, scene_source=canvas, scene_faces=faces,
                                scene_issues=issues, scene_versions=saved.get("versions", []),
                                scene_confirmed_faces=saved.get("faces", []))
    canvas = st.session_state["scene_source"]
    faces = st.session_state["scene_faces"]
    versions = st.session_state["scene_versions"]
    if st.session_state["scene_issues"] or not faces:
        for issue in st.session_state["scene_issues"]:
            st.warning(t["errors"][issue])
        st.warning("Выберите чёткое фото с 1–5 хорошо видимыми лицами." if ru else
                   "Choose a clear photo with one to five visible faces.")
        return
    if not versions:
        st.markdown("**Проверьте найденные лица**" if ru else "**Check detected faces**")
        st.caption("Снимите галочку с лишних отметок." if ru else "Uncheck any false detections.")
        selected_faces = []
        columns = st.columns(min(5, len(faces)))
        for index, box in enumerate(faces):
            x, y, w, h = box
            with columns[index % len(columns)]:
                st.image(canvas.crop((x, y, x + w, y + h)), width=80)
                if st.checkbox(f"Лицо {index + 1}" if ru else f"Face {index + 1}",
                               value=True, key=f"scene_face_{digest}_{index}"):
                    selected_faces.append(box)
        if market and len(selected_faces) != 2:
            st.info("Для ярмарки отметьте два лица." if ru else "Select two faces for the Christmas market.")
            return
        if not 1 <= len(selected_faces) <= 5:
            st.warning("Выберите от одного до пяти лиц." if ru else "Select one to five faces.")
            return
        faces = selected_faces
    else:
        faces = st.session_state["scene_confirmed_faces"]
    if not versions and st.button("Создать открытку" if ru else "Create card", type="primary"):
        with st.spinner(t["creating"]):
            if market:
                prompt = (
                    "Create a realistic wide holiday photograph featuring exactly the two foreground adults "
                    "in the reference photo. Keep their left-to-right order and preserve each person's "
                    "individual facial structure, expression, age, glasses, and hair. Seat them on opposite "
                    "sides of one small wooden table with two mugs at an evening Christmas market. Both look "
                    "toward the camera at their natural head angle. Include complete heads, generous headroom, "
                    "winter coats, natural proportions, and warm strings of lights. No text or frames."
                )
                edit_mask = None
            else:
                prompt = (
                    "Transform the setting into a cozy Christmas room with a lit fir tree, glass ornaments, "
                    "burgundy ribbons and festive knits. Keep people, poses, scale and positions as shown. "
                    "Preserve faces, glasses, expressions and lighting. No head movement, no text."
                )
                edit_mask = scene_editable(canvas.size, faces)
            result = edit_scene(canvas, edit_mask, prompt, api_key)
        st.session_state["scene_confirmed_faces"] = detect_scene_faces(result) if market else faces
        versions.append(result)
        st.session_state["scene_saved"][digest] = {
            "versions": versions, "faces": st.session_state["scene_confirmed_faces"]
        }
        st.rerun()
    if not versions:
        return
    left, right = st.columns(2)
    original_preview = photo.copy()
    original_preview.thumbnail((600, 440), Image.Resampling.LANCZOS)
    left.image(original_preview, caption="Исходное фото" if ru else "Original photo")
    version = st.selectbox("Версия" if ru else "Version", list(range(len(versions))),
                           index=len(versions) - 1, format_func=lambda value: str(value + 1),
                           key=f"scene_version_{digest}_{len(versions)}")
    current = versions[version]
    can_refine = len(faces) == 2 if market else bool(faces)
    if can_refine and len(versions) < 4:
        with st.expander("Поправить одно лицо" if ru else "Refine one face"):
            choices = []
            for index, (x, y, w, h) in enumerate(faces):
                choices.append(index)
                st.image(current.crop((x, y, x + w, y + h)), caption=str(index + 1), width=85)
            choice = st.selectbox("Лицо" if ru else "Face", choices,
                                  format_func=lambda value: str(value + 1))
            note = st.text_area("Что поправить?" if ru else "What should change?", max_chars=300)
            if st.button("Применить правку" if ru else "Apply refinement", disabled=not note.strip()):
                with st.spinner(t["creating"]):
                    versions.append(edit_scene(current, scene_editable(current.size, faces, choice),
                        "Only refine the visible face within the supplied mask. Keep likeness, age, scale, "
                        "and existing expressions and lighting. Requested adjustment: " + note, api_key))
                st.rerun()
    if can_refine:
        st.caption(f"Корректировок осталось: {4-len(versions)}" if ru else
                   f"Refinements left: {4-len(versions)}")
    else:
        st.caption("Локальная правка доступна, когда в результате найдены оба лица." if ru else
                   "Local refinements require both faces to be detected in the result.")
    card = render_original(current, "gold", title, message, name, christmas=True)
    preview = card.copy()
    preview.thumbnail((600, 440), Image.Resampling.LANCZOS)
    right.image(preview, caption=t["preview"])
    st.download_button(t["download"] + " PNG", encode_image(card, "PNG"),
                       "Magic_Greetings_Christmas_Market.png" if market else
                       "Magic_Greetings_Last_Ornament.png", "image/png", on_click="ignore")


st.set_page_config(page_title="Magic Greetings", page_icon="✦", layout="wide")
L = json.loads((ROOT / "locales.json").read_text())
st.markdown("""
<style>
.stApp {background:linear-gradient(150deg,#f9efe0,#fffaf3 50%,#edf3ea);color:#263229;color-scheme:light}
.block-container {max-width:1380px;padding-top:2rem}
h1 {font-family:Georgia,serif;letter-spacing:-1px;color:#173e33}
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCaptionContainer"], .stApp label, .stApp summary {color:#263229}
.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"],
.stApp [data-baseweb="select"] > div, .stApp input, .stApp textarea,
.stApp [data-testid="stFileUploaderDropzone"], .stApp button[kind="secondary"] {
  background:#fffdf8;color:#263229;border-color:#ae987d
}
.stApp [data-baseweb="select"] svg {fill:#345446}
.stApp button[kind="secondary"]:hover {background:#f8ede5;color:#703140;border-color:#703140}
.stApp button[kind="primary"] {background:#773044;border-color:#773044}
.stApp button[kind="primary"]:hover {background:#612436;border-color:#612436}
.stApp button[kind="primary"] p {color:#fffdf8}
.stApp [data-testid="stImage"] img {border-radius:10px}
.mg-hero {padding:1.4rem 1.8rem;margin-bottom:1rem;border-radius:16px;
          border:1px solid #bca371;background:linear-gradient(115deg,#153b30,#315649);color:#fff1d6}
.mg-hero h1 {margin:0;color:#fff1d6}
@media(max-width:600px) {.block-container {padding:1rem}.stApp h1{font-size:2rem}}
</style>
""", unsafe_allow_html=True)
lang = st.selectbox("Язык сайта / Website language", list(L), format_func=lambda value: L[value]["language_name"], key="ui_language")
t = L[lang]
st.markdown('<section class="mg-hero"><h1>Magic Greetings</h1></section>', unsafe_allow_html=True)

def render_original(photo, theme, title, message, name="", christmas=False):
    # PNG keeps the decoded photo at its original size, with graphics outside it.
    palette = THEMES[theme]
    padding = max(32, round(photo.width * 0.05))
    width = max(640, photo.width + 2 * padding)

    def band_height(value):
        return max(150, round(width * 0.15), 30 * len(value.splitlines()) + 38) if value.strip() else 0

    caption = "\n".join(value for value in (name, message) if value.strip())
    heading_height = band_height(title)
    caption_height = band_height(caption)
    top = padding + heading_height
    height = top + photo.height + caption_height + padding
    if width * height > 30_000_000:
        raise PhotoError("pixels")
    card = Image.new("RGB", (width, height), "#142c24" if christmas else palette["mat"])
    card.paste(photo, ((width - photo.width) // 2, top))
    maximum = min(116, max(32, round(width * 0.061)))
    for value, y, height, bold, color in (
        (title, padding, heading_height, True, palette["accent"]),
        (caption, top + photo.height, caption_height, False, palette["ink"]),
    ):
        if not height:
            continue
        if not christmas:
            text_block(card, value, (padding, y, width - 2 * padding, height),
                       maximum, 15, color, bold)
            continue
        draw = ImageDraw.Draw(card)
        filename = "DejaVuSerif-Italic.ttf" if bold else "DejaVuSerif.ttf"
        inset = max(20, round(width * 0.02))
        for size in range(min(maximum, round(width * (0.048 if bold else 0.027))), 14, -1):
            font_path = FONT_DIR / filename
            if not font_path.is_file():
                font_path = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
            selected_font = ImageFont.truetype(str(font_path), size)
            lines = wrap(draw, value, selected_font, width - 2 * padding)
            line_height = round(size * 1.3)
            if len(lines) * line_height <= height - 2 * inset:
                break
        else:
            raise PhotoError("text_long")
        center = width // 2
        start = y + (height - len(lines) * line_height) // 2
        for index, line in enumerate(lines):
            draw.text((center, start + index * line_height), line,
                      font=selected_font, fill="#f5d89b" if bold else "#fff3db", anchor="mt")
    return card

MARKET_EXAMPLE = "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"

def runtime_setting(name):
    try:
        return str(st.secrets.get(name) or os.environ.get(name, ""))
    except FileNotFoundError:
        return os.environ.get(name, "")


@st.cache_data(show_spinner=False)
def frame_example(theme):
    sample = Image.new("RGB", (640, 480), "#eee5d7")
    draw = ImageDraw.Draw(sample)
    draw.text((320, 240), "Your photo", font=ImageFont.truetype(
        str(FONT_DIR / "DejaVuSans.ttf"), 32), fill="#554b40", anchor="mm")
    card = render(sample, theme, "", "")
    card.thumbnail((560, 380), Image.Resampling.LANCZOS)
    return encode_image(card, "JPEG")


ru = lang == "ru"
st.subheader("Выберите сцену или рамку" if ru else "Choose a scene or frame")
st.caption("Нажмите «Выбрать» под картинкой. Затем загрузите фото и добавьте пожелание."
           if ru else "Choose a picture below, then upload your photo and add your message.")
api_key = runtime_setting("OPENAI_API_KEY")
preview_password = runtime_setting("MAGIC_GREETING_TEST_PASSWORD")
designs = {
    "ornament": {
        "title": "Последняя игрушка" if ru else "The Last Ornament",
        "description": "Ёлка и уютная комната, праздничные свитеры. 1–5 человек."
                       if ru else "A tree, a cozy room and festive knits. 1–5 people.",
    },
    "market": {
        "title": "Новогодняя ярмарка" if ru else "Christmas Market",
        "description": "Двое за столиком на вечерней ярмарке."
                       if ru else "Two people at a table at an evening Christmas market.",
    },
    "frame_gold": {
        "title": "Фото в золотой рамке" if ru else "Photo in a gold frame",
        "description": "Ваш снимок в декоративной рамке с пожеланием."
                       if ru else "Your photo in a decorative frame with a greeting.",
    },
    "frame_winter": {
        "title": "Фото в зимней рамке" if ru else "Photo in a winter frame",
        "description": "Ваш снимок в зимнем оформлении с пожеланием."
                       if ru else "Your photo in a winter frame with a greeting.",
    },
}
chosen = st.session_state.get("chosen_design")
columns = st.columns(2)
for index, (design_id, design) in enumerate(designs.items()):
    with columns[index % 2]:
        with st.container(border=True):
            if design_id == "market":
                st.image(base64.b64decode(MARKET_EXAMPLE), width="stretch")
                st.caption("Пример результата" if ru else "Example result")
            elif design_id == "ornament":
                st.image(str(ROOT / "assets" / "family.jpg"), width="stretch")
                st.caption("Фон для примера атмосферы сцены" if ru else "Background showing the scene atmosphere")
            else:
                st.image(frame_example("gold" if design_id == "frame_gold" else "winter"), width="stretch")
            st.markdown("**" + design["title"] + "**")
            st.caption(design["description"])
            if st.button(("✓ Выбрано" if ru else "✓ Selected") if chosen == design_id else
                         ("Выбрать" if ru else "Choose"), key="choose_" + design_id,
                         type="primary" if chosen == design_id else "secondary"):
                st.session_state["chosen_design"] = design_id
                st.rerun()
if chosen not in designs:
    st.info("Сначала выберите картинку выше." if ru else "First choose a picture above.")
    st.stop()

st.success(("Вы выбрали: " if ru else "Selected: ") + designs[chosen]["title"])
scene_mode = chosen in ("ornament", "market")
scene = chosen
original = scene_mode
selected = "winter" if chosen == "frame_winter" else "gold"
category = "all"
card_format = "static"
if scene_mode:
    if not api_key or not preview_password:
        st.info("Генерация сцен пока недоступна. Попробуйте рамку." if ru else
                "Scene generation is currently unavailable. Please try a frame.")
        st.stop()
    supplied = st.text_input("Пароль тестового доступа" if ru else "Test access password",
                             type="password", key="scene_access")
    if not hmac.compare_digest(supplied.encode(), preview_password.encode()):
        st.info("Введите пароль для проверки сцены." if ru else "Enter the password to test a scene.")
        st.stop()
    st.caption("Для сцены фотография передаётся на обработку. Все лица должны быть хорошо видны."
               if ru else "Your photo is sent for scene generation. All faces must be clearly visible.")
st.subheader("Загрузите фото" if ru else "Upload your photo")
revision = st.session_state.get("upload_revision", 0)
uploaded = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "webp"],
                            key=f"upload_{revision}", max_upload_size=15)
if not original:
    st.caption(t["hint"])
st.subheader("Поздравление на открытке" if ru else "Your greeting on the card")
gl = st.selectbox(t["greeting_language"], list(L),
                  format_func=lambda value: L[value]["language_name"], key="content_language")
g = L[gl]
preset = st.selectbox(t["preset"], ["christmas", "newyear", "custom"],
                      format_func=lambda value: t[value], key="greeting_occasion")
default_title, default_message = g["greetings"].get(preset, ("", ""))
if preset == "christmas" and gl == "ru":
    default_title, default_message = "С Рождеством Христовым!", "Мир в доме, любовь и радость в сердце."
elif preset == "christmas" and gl == "en":
    default_message = "Peace in your home, love and joy in your heart."
text_key = f"{gl}_{preset}"
st.caption("Оставьте готовый текст или напишите своё пожелание. Его можно менять после генерации."
           if ru else "Keep the suggested text or write your own. You can edit it after generation.")
name = st.text_input(t["name"], max_chars=60, key="recipient")
title = st.text_input(t["heading"], value=default_title, max_chars=100, key="heading_" + text_key)
message = st.text_area(t["message"], value=default_message,
                       max_chars=250, key="message_" + text_key)
size, horizontal, vertical = 100, 50, 50
if not original:
    with st.expander(t["adjust"]):
        size = st.slider(t["size"], 60, 100, 100, key="size")
        horizontal = st.slider(t["horizontal"], 0, 100, 50, key="horizontal")
        vertical = st.slider(t["vertical"], 0, 100, 50, key="vertical")

if st.button(t["reset"]):
    for key in list(st.session_state):
        if key.startswith(("upload_", "photo_", "result", "check_", "scene_")) or key in {
            "recipient", "size", "horizontal", "vertical"
        }:
            del st.session_state[key]
    st.session_state["upload_revision"] = revision + 1
    st.rerun()

st.caption(t["privacy"])


@st.cache_resource
def supported_characters():
    with TTFont(str(FONT_DIR / "DejaVuSans.ttf")) as selected_font:
        return set(selected_font.getBestCmap())


if not uploaded:
    st.session_state.pop("result", None)
    st.stop()

data = uploaded.getvalue()
digest = hashlib.sha256(data).hexdigest()
signature = (digest, original, selected, category, preset, gl, title, message, name, size, horizontal, vertical, card_format)
if st.session_state.get("result_signature") != signature:
    st.session_state.pop("result", None)

try:
    if st.session_state.get("photo_digest") != digest:
        st.session_state.pop("photo_image", None)
        st.session_state.pop("check_signature", None)
        st.session_state["photo_image"] = load_photo(data)
        st.session_state["photo_digest"] = digest
    photo = st.session_state["photo_image"]
    characters = name + title + message
    if any(ord(character) not in supported_characters() for character in characters if character != "\n"):
        st.error(t["unsupported"])
        st.stop()
    if scene_mode:
        scene_editor(photo, digest, scene, lang, title, message, name, api_key, t)
        st.stop()
    if original:
        card = render_original(photo, selected, title, message, name, christmas=preset == "christmas")
    else:
        card = render(photo, selected, title, message, name, size, horizontal, vertical)
    preview = card.copy()
    preview.thumbnail((320, 400), Image.Resampling.LANCZOS)
    st.image(preview, caption=t["preview"], width=320)
    if st.button(t["create"], type="primary"):
        with st.spinner(t["creating"]):
            result = {
                "png": encode_image(card, "PNG"),
                "jpg": encode_image(card, "JPEG"),
            }
            if card_format == "animated":
                result["mp4"] = animate(card, selected)
            st.session_state["result"] = result
            st.session_state["result_signature"] = signature
            st.session_state["result_filename"] = (
                f"Magic_Greetings_{selected}_{gl}_{datetime.now():%Y%m%d_%H%M%S}"
            )
    if st.session_state.get("result"):
        st.success(t["ready"])
        result = st.session_state["result"]
        if "mp4" in result:
            st.video(result["mp4"])
        for extension, contents in result.items():
            mime = {"png": "image/png", "jpg": "image/jpeg", "mp4": "video/mp4"}[extension]
            st.download_button(
                t["download"] + " " + extension.upper(), contents,
                st.session_state["result_filename"] + "." + extension, mime,
                on_click="ignore",
            )
except PhotoError as error:
    st.error(t["errors"].get(str(error), "Не удалось подготовить открытку." if lang == "ru"
                                else "Could not prepare the card."))
except (requests.RequestException, KeyError, IndexError, ValueError) as error:
    if scene_mode:
        response = getattr(error, "response", None)
        logging.getLogger(__name__).error("Scene request or rendering failed (%s, HTTP %s)",
                                          type(error).__name__, getattr(response, "status_code", None))
        st.error("Генерация не завершилась. Попробуйте ещё раз." if lang == "ru" else
                 "The scene could not be generated. Please try again.")
    else:
        st.error("Не удалось подготовить открытку." if lang == "ru" else "Could not prepare the card.")
except Exception as error:
    if scene_mode:
        logging.getLogger(__name__).error("Card preparation failed (%s)", type(error).__name__)
        st.error("Не удалось подготовить открытку к скачиванию." if lang == "ru"
                 else "Could not prepare the card for download.")
    else:
        st.error("Не удалось подготовить открытку." if lang == "ru" else "Could not prepare the card.")
