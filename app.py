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
        st.session_state.update(scene_digest=digest, scene_source=canvas, scene_faces=faces,
                                scene_issues=issues, scene_versions=[])
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
lang = st.selectbox("🌐", list(L), format_func=lambda value: L[value]["language_name"], key="ui_language")
t = L[lang]
st.markdown('<section class="mg-hero"><h1>Magic Greetings</h1></section>', unsafe_allow_html=True)

mode_labels = {
    "en": ("Card design", "Festive scene or greeting", "Photo in a festive frame"),
    "ru": ("Оформление открытки", "Праздничная сцена или поздравление", "Фото в праздничной рамке"),
    "es": ("Diseño de la tarjeta", "Foto original", "Plantilla ilustrada"),
    "fr": ("Mise en page", "Photo originale", "Modèle illustré"),
    "de": ("Kartendesign", "Originalfoto", "Illustrierte Vorlage"),
    "it": ("Stile della cartolina", "Foto originale", "Modello illustrato"),
    "pt": ("Design do cartão", "Foto original", "Modelo ilustrado"),
    "pl": ("Układ kartki", "Zdjęcie oryginalne", "Ilustrowany szablon"),
    "tr": ("Kart düzeni", "Orijinal fotoğraf", "Resimli şablon"),
    "nl": ("Kaartontwerp", "Originele foto", "Geïllustreerd sjabloon"),
}
mode_label, original_label, template_label = mode_labels[lang]
original = st.radio(mode_label, [True, False],
                    format_func=lambda value: original_label if value else template_label,
                    horizontal=True, key="original_photo")
if lang == "ru":
    st.caption("Загрузите своё фото для любого варианта. В пробной сцене меняются фон и одежда. "
               "В праздничной рамке используется весь ваш снимок.")
elif lang == "en":
    st.caption("Upload your photo for either option. The trial scene changes the setting and clothing. "
               "A festive frame uses your entire photo.")



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

SCENES = {
    "market": ("Рождественская ярмарка", "Christmas Market", "christmas", "#2c1d1b", "#f2cf93",
               "Пусть будет с кем разделить тепло.", "May there always be someone to share the warmth with."),
    "first_snow": ("Первый снег", "First Snow", "winter", "#e3e6ee", "#2d3655",
                   "Пусть дома всегда будет кому тебя обнять.", "May there always be a hug waiting for you at home."),
    "morning": ("Рождественское утро", "Christmas Morning", "christmas", "#f4ebdf", "#65402c",
                "Пусть рождественское утро начнётся с любимых голосов.", "May Christmas morning begin with the voices you love."),
    "evening": ("Новогодний вечер", "New Year’s Eve", "newyear", "#1a2038", "#f2d190",
                "Пусть сбудется то, о чём пока знаешь только ты.", "May the wish you’re keeping to yourself come true."),
}


def render_scene_layout(photo, scene, title, message, name=""):
    _, _, _, background, ink, _, _ = SCENES[scene]
    gap = max(32, min(95, round(photo.width * .046)))
    panel = max(390, min(1150, round(photo.width * .52)))
    width, height = panel + photo.width + 3 * gap, max(635, photo.height + 2 * gap)
    if width * height > 30_000_000:
        raise PhotoError("pixels")
    card = Image.new("RGB", (width, height), background)
    card.paste(photo, (panel + 2 * gap, (height - photo.height) // 2))
    title_height = round(height * .25)
    top = round(height * .16)
    caption = "\n".join(value for value in (name, message) if value.strip())
    text_block(card, title, (gap, top, panel, title_height), min(72, panel // 9), 16, ink, True)
    text_block(card, caption, (gap, top + title_height, panel, round(height * .4)),
               min(53, panel // 14), 15, ink)
    return card



category = st.selectbox(t["category"], ["all", *t["categories"]],
                        format_func=lambda key: t["all"] if key == "all" else t["categories"][key],
                        key="category")
templates = [item for item in TEMPLATES if category == "all" or category in item["categories"]]
template_ids = [item["id"] for item in templates]
if st.session_state.get("template") not in template_ids:
    st.session_state["template"] = template_ids[0]
selected = st.selectbox(t["style"], template_ids,
                        format_func=lambda key: THEMES[key]["names"][lang], key="template")
gl = st.selectbox(t["greeting_language"], list(L),
                  format_func=lambda value: L[value]["language_name"], key="content_language")
g = L[gl]
preset_options = ["newyear", "christmas", "winter", "custom"]
if original:
    preset_options.extend(SCENES)
if st.session_state.get("preset") not in preset_options:
    st.session_state["preset"] = preset_options[0]
preset = st.selectbox(t["preset"], preset_options,
                      format_func=lambda value: SCENES[value][0 if lang == "ru" else 1]
                      if value in SCENES else t[value], key="preset")
text_key = f"{gl}_{preset}_{original}"
occasion = SCENES[preset][2] if preset in SCENES else preset
default_title, default_message = g["greetings"].get(occasion, g["greetings"]["newyear"])
if original:
    wishes = {
        "en": {
            "christmas": "May your home be filled with peace, your heart with love, and may you have someone dear to share the joy of Christmas with.",
            "newyear": "May the thing you quietly hope for find its way to you this year.",
            "winter": "May there always be someone to share the warmth with.",
        },
        "ru": {
            "christmas": "Пусть в вашем доме будет мир, в сердце — любовь, а рядом — те, с кем хочется разделить рождественскую радость.",
            "newyear": "Пусть сбудется то, о чём пока знаешь только ты.",
            "winter": "Пусть будет с кем разделить тепло.",
        },
    }
    default_message = wishes.get(gl, {}).get(preset, default_message)
    if gl == "ru" and preset == "christmas":
        default_title = "С Рождеством Христовым!"
    if preset in SCENES and gl in ("ru", "en"):
        default_message = SCENES[preset][5 if gl == "ru" else 6]
        if preset == "first_snow":
            default_title = "С первым снегом!" if gl == "ru" else "Happy First Snow!"


@st.cache_data(show_spinner=False)
def gallery_image(template_id, title, message, label):
    sample = Image.new("RGB", (800, 650), "#dfd8ce")
    draw = ImageDraw.Draw(sample)
    # Gallery is explicitly an empty photo slot, never a customer's image.
    from engine import font
    draw.text((400, 325), label, fill="#615d56", font=font(26), anchor="mm")
    card = render(sample, template_id, title, message)
    card.thumbnail((360, 450), Image.Resampling.LANCZOS)
    return encode_image(card, "JPEG")


with st.expander(t["all"], expanded=not original):
    columns = st.columns(2)
    for index, item in enumerate(templates):
        columns[index % 2].image(
            gallery_image(item["id"], default_title, default_message, t["upload"]),
            caption=item["names"][lang], width="stretch",
        )

card_format = "static" if original else st.radio(t["format"], ["static", "animated"],
                       format_func=lambda value: t[value], horizontal=True, key="format")
def runtime_setting(name):
    try:
        return str(st.secrets.get(name) or os.environ.get(name, ""))
    except FileNotFoundError:
        return os.environ.get(name, "")


api_key = runtime_setting("OPENAI_API_KEY")
preview_password = runtime_setting("MAGIC_GREETING_TEST_PASSWORD")
scene_mode = False
scene = "ornament"
if api_key and preview_password and original and lang in ("en", "ru"):
    wants_scene = st.checkbox("Попробовать праздничную сцену" if lang == "ru"
                              else "Try a festive scene")
    if wants_scene:
        supplied = st.text_input("Пароль тестового доступа" if lang == "ru"
                                 else "Test access password", type="password")
        scene_mode = hmac.compare_digest(supplied.encode(), preview_password.encode())
        if not scene_mode:
            st.stop()
if scene_mode:
    scene_labels = {"ornament": "Последняя игрушка", "market": "Новогодняя ярмарка (для двоих)"} if lang == "ru" else {
        "ornament": "The Last Ornament", "market": "Christmas Market (for two)"
    }
    scene = st.selectbox("Праздничная сцена" if lang == "ru" else "Festive scene",
                         list(scene_labels), format_func=scene_labels.__getitem__)
    st.info("Загрузите чёткое фото, где все лица хорошо видны. Фото передаётся для генерации сцены."
            if lang == "ru" else
            "Upload a clear photo with all faces visible. The photo is sent for scene generation.")
    preset = "christmas"
    text_key = f"{gl}_festive_scene"
    default_title, default_message = g["greetings"]["christmas"]
    if gl == "ru":
        default_title, default_message = "С Рождеством Христовым!", "Мир в доме, любовь и радость в сердце."
    elif gl == "en":
        default_message = "Peace in your home, love and joy in your heart."
revision = st.session_state.get("upload_revision", 0)
uploaded = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "webp"],
                            key=f"upload_{revision}", max_upload_size=15)
if not original:
    st.caption(t["hint"])
name = st.text_input(t["name"], max_chars=60, key="recipient")
title = st.text_input(t["heading"], value=default_title, max_chars=100, key="heading_" + text_key)
message = st.text_area(t["message"], value=default_message if preset != "custom" else "",
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
    if not original:
        check_signature = (digest, category)
        if st.session_state.get("check_signature") != check_signature:
            st.session_state["photo_checks"] = inspect(photo, category)
            st.session_state["check_signature"] = check_signature
        issues, count = st.session_state["photo_checks"]
        if issues:
            for issue in issues:
                st.error(t["errors"][issue])
            st.stop()
    characters = name + title + message
    if any(ord(character) not in supported_characters() for character in characters if character != "\n"):
        st.error(t["unsupported"])
        st.stop()
    if scene_mode:
        scene_editor(photo, digest, scene, lang, title, message, name, api_key, t)
        st.stop()
    if preset in SCENES:
        card = render_scene_layout(photo, preset, title, message, name)
    elif original:
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
        st.error(t["errors"]["video"])
except Exception as error:
    if scene_mode:
        logging.getLogger(__name__).error("Card preparation failed (%s)", type(error).__name__)
        st.error("Не удалось подготовить открытку к скачиванию." if lang == "ru"
                 else "Could not prepare the card for download.")
    else:
        st.error(t["errors"]["video"])
