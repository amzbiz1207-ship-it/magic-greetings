from datetime import datetime
import hashlib
import json

import streamlit as st
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from engine import (
    FONT_DIR, ROOT, TEMPLATES, THEMES, PhotoError, animate,
    encode_image, inspect, load_photo, render, text_block, wrap,
)

st.set_page_config(page_title="Magic Greetings", page_icon="✦", layout="centered")
L = json.loads((ROOT / "locales.json").read_text())
st.markdown("""
<style>
.stApp {background:#faf8f5;color:#282525}
.block-container {max-width:980px;padding-top:2rem}
h1 {font-family:Georgia,serif;letter-spacing:-1px}
div[data-testid="stButton"] button[kind="primary"] {background:#345446;border-color:#345446}
div[data-testid="stImage"] img {border-radius:10px}
@media(max-width:600px) {.block-container {padding:1rem}.stApp h1{font-size:2rem}}
</style>
""", unsafe_allow_html=True)
lang = st.selectbox("🌐", list(L), format_func=lambda value: L[value]["language_name"], key="ui_language")
t = L[lang]
st.title("Magic Greetings")
st.caption("Christmas & New Year 2027")

mode_labels = {
    "en": ("Card layout", "Original photo", "Illustrated template"),
    "ru": ("Оформление открытки", "Исходное фото", "Рисованный шаблон"),
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
            selected_font = ImageFont.truetype(str(FONT_DIR / filename), size)
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
preset = st.selectbox(t["preset"], ["newyear", "christmas", "winter", "custom"],
                      format_func=lambda value: t[value], key="preset")
text_key = f"{gl}_{preset}_{original}"
default_title, default_message = g["greetings"].get(preset, g["greetings"]["newyear"])
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
        if key.startswith(("upload_", "photo_", "result", "check_")) or key in {
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
signature = (digest, original, selected, category, gl, title, message, name, size, horizontal, vertical, card_format)
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
    card = (render_original(photo, selected, title, message, name, christmas=preset == "christmas")
            if original
            else render(photo, selected, title, message, name, size, horizontal, vertical))
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
    st.error(t["errors"][str(error)])
except Exception:
    st.error(t["errors"]["video"])
