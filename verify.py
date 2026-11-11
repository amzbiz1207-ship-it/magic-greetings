"""Run python verify.py from the project directory."""
from pathlib import Path
import io
import json
import tempfile

import numpy as np
from PIL import Image, ImageFilter
from streamlit.testing.v1 import AppTest

from engine import (
    ROOT, TEMPLATES, PhotoError, animate, encode_image, inspect,
    load_photo, photo_placement, render,
)

locales = json.loads((ROOT / "locales.json").read_text())
sample = load_photo((ROOT / "examples/sample_family.png").read_bytes())
pet = load_photo((ROOT / "examples/sample_pet.png").read_bytes())
results = []


def record(label, condition):
    assert condition, label
    results.append(label)
    print("PASS", label, flush=True)


record("Family photo accepted", not inspect(sample, "family")[0])
record("Pet accepted without human face", not inspect(pet, "pets")[0])
record("No human face rejected in portrait category", "face_missing" in inspect(pet, "women")[0])
single = sample.crop((170, 0, 790, 1024))
record("Single portrait accepted", not inspect(single, "women")[0])
record("Group rejected in single-person category", "face_count" in inspect(sample, "women")[0])
record("Low resolution rejected", "resolution" in inspect(sample.resize((300, 200)))[0])
record("Blur rejected", "blur" in inspect(sample.filter(ImageFilter.GaussianBlur(20)))[0])
record("Dark photo rejected", "light" in inspect(Image.new("RGB", (800, 800), "#050505"))[0])
record("Overexposure rejected", "light" in inspect(Image.new("RGB", (800, 800), "white"))[0])
for contents, expected in [(b"invalid", "format"), (b"0" * (15*1024*1024+1), "file_size")]:
    try:
        load_photo(contents)
        raise AssertionError(expected)
    except PhotoError as error:
        record("Invalid upload " + expected, str(error) == expected)

# Verify every generated photo region equals precisely the resized input pixels.
for template in TEMPLATES:
    for language, words in locales.items():
        card = render(sample, template["id"], words["title"], words["default_message"], "Alexandra")
        fitted, x, y = photo_placement(sample, template["id"])
        restored = card.crop((x, y, x+fitted.width, y+fitted.height))
        assert np.array_equal(np.array(restored), np.array(fitted))
        assert Image.open(io.BytesIO(encode_image(card, "PNG"))).size == (1080, 1350)
record("80 template/language combinations and exact photo pixel preservation", True)
for language, words in locales.items():
    render(sample, "gold", words["title"], (words["default_message"]+" ") * 3, "W" * 60)
record("Long 60-character name and longer messages in ten languages", True)
try:
    render(sample, "gold", "Happy New Year", "\n".join(["text"] * 50))
    raise AssertionError("Text overflow")
except PhotoError as error:
    record("Text overflow rejected", str(error) == "text_long")

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
record("App starts", not app.exception)
app.file_uploader[0].upload("family.png", (ROOT/"examples/sample_family.png").read_bytes(), "image/png").run()
record("Upload and preview", not app.exception and not app.error)
app.button[-1].click().run()
record("PNG/JPG create and download buttons", len(app.get("download_button")) == 2 and not app.error)
app.text_input(key="recipient").set_value("Changed name").run()
record("Changing name invalidates prior result", len(app.get("download_button")) == 0)
app.selectbox(key="ui_language").select("ru").run()
app.selectbox(key="content_language").select("de").run()
record("Independent UI and greeting languages", app.selectbox(key="ui_language").value == "ru" and app.selectbox(key="content_language").value == "de")
app.radio(key="format").set_value("animated").run()
app.button[-1].click().run()
record("Animated creation and three download buttons", len(app.get("download_button")) == 3 and not app.exception and not app.error)
video = app.session_state["result"]["mp4"]
(ROOT/"examples/Magic_Greetings_Animated.mp4").write_bytes(video)
for language in locales:
    app.selectbox(key="ui_language").select(language).run()
    record("UI language " + language, not app.exception)
app.selectbox(key="category").select("pets").run()
app.file_uploader[0].upload("pet.png", (ROOT/"examples/sample_pet.png").read_bytes(), "image/png").run()
record("Pet category through app", not app.error and not app.exception)
app.button[0].click().run()
record("Clear removes private image and results", "photo_image" not in app.session_state and "result" not in app.session_state)
other = AppTest.from_file(str(ROOT/"app.py")).run()
record("Independent session has no previous photo", "photo_image" not in other.session_state)
report = {"checks": len(results), "passed": results, "template_language_combinations":80}
(ROOT/"verification_results.json").write_text(json.dumps(report, indent=2))
print("ALL PASSED", len(results))
