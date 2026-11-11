from datetime import datetime
import hashlib
import json

import streamlit as st
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from engine import (
    FONT_DIR, ROOT, TEMPLATES, THEMES, PhotoError, animate,
    encode_image, inspect, load_photo, render,
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
text_key = gl + "_" + preset
default_title, default_message = g["greetings"].get(preset, g["greetings"]["newyear"])


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


with st.expander(t["all"], expanded=True):
    columns = st.columns(2)
    for index, item in enumerate(templates):
        columns[index % 2].image(
            gallery_image(item["id"], default_title, default_message, t["upload"]),
            caption=item["names"][lang], width="stretch",
        )

card_format = st.radio(t["format"], ["static", "animated"],
                       format_func=lambda value: t[value], horizontal=True, key="format")
revision = st.session_state.get("upload_revision", 0)
uploaded = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "webp"],
                            key=f"upload_{revision}", max_upload_size=15)
st.caption(t["hint"])
name = st.text_input(t["name"], max_chars=60, key="recipient")
title = st.text_input(t["heading"], value=default_title, max_chars=100, key="heading_" + text_key)
message = st.text_area(t["message"], value=default_message if preset != "custom" else "",
                       max_chars=250, key="message_" + text_key)
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
signature = (digest, selected, category, gl, title, message, name, size, horizontal, vertical, card_format)
if st.session_state.get("result_signature") != signature:
    st.session_state.pop("result", None)

try:
    if st.session_state.get("photo_digest") != digest:
        st.session_state.pop("photo_image", None)
        st.session_state.pop("check_signature", None)
        st.session_state["photo_image"] = load_photo(data)
        st.session_state["photo_digest"] = digest
    photo = st.session_state["photo_image"]
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
    st.error(t["errors"][str(error)])
except Exception:
    st.error(t["errors"]["video"])
