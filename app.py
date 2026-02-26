import streamlit as st
import os

# --- CLOUD & COMPATIBILITY SETTINGS ---
# Erzwingt CPU-Nutzung und Legacy-Modus für Teachable Machine (.h5) Modelle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import tensorflow as tf

# Versuche das Legacy Keras Paket zu laden, das für .h5 TM-Modelle nötig ist
try:
    import tf_keras
    keras_loader = tf_keras
except ImportError:
    keras_loader = tf.keras

# --- KONFIGURATION ---
st.set_page_config(page_title="Digitale Fundkiste", page_icon="📦", layout="wide")

# Pfad-Management für Cloud-Server (KORRIGIERT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_FILE = os.path.join(BASE_DIR, "database.json")
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# --- CSS STYLING ---
st.markdown("""
<style>
    .fund-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid #f0f2f6;
        color: #1f1f1f;
    }
    .status-verfuegbar { color: #2e7d32; font-weight: bold; font-size: 1.1em; }
    .status-abgeholt { color: #c62828; font-weight: bold; font-size: 1.1em; }
    .stButton>button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- LOGIK-FUNKTIONEN ---
def init_system():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
    cleanup_expired_items()

def load_db():
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def cleanup_expired_items():
    db = load_db()
    now = datetime.now()
    new_db = [item for item in db if not (item.get("status") == "abgeholt" and
              item.get("claimed_at") and
              now > datetime.fromisoformat(item["claimed_at"]) + timedelta(hours=48))]
    if len(new_db) != len(db):
        current_paths = [i["image_path"] for i in new_db]
        for f in os.listdir(UPLOAD_DIR):
            full_p = os.path.join(UPLOAD_DIR, f)
            if full_p not in current_paths:
                try:
                    os.remove(full_p)
                except:
                    pass
        save_db(new_db)

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None
    try:
        # Laden über den Legacy-Loader (tf_keras)
        model = keras_loader.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # TM Format: "0 Name" -> extrahiere "Name"
            class_names = [l.strip().split(" ", 1)[-1] if " " in l.strip() else l.strip() for l in lines]
        st.success(f"✅ Modell geladen mit {len(class_names)} Klassen: {class_names}")
        return model, class_names
    except Exception as e:
        st.error(f"KI-Fehler: {e}")
        return None, None

def classify_image(image_path, model, class_names):
    if not model or not class_names:
        return "Unbekannt", 0.0
    try:
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img).astype(np.float32)
        normalized_img = (img_array / 127.5) - 1
        data = np.expand_dims(normalized_img, axis=0)

        prediction = model.predict(data, verbose=0)
        idx = np.argmax(prediction)
        return class_names[idx], float(prediction[0][idx])
    except:
        return "Fehler", 0.0

# --- APP START ---
init_system()
model, labels = load_ai_model()

st.title("📦 Digitale Fundkiste")
st.markdown("---")

tab1, tab2 = st.tabs(["📸 Fund melden", "🔍 Fundkiste durchsuchen"])

# --- TAB 1: MELDEN ---
with tab1:
    st.header("Neuen Gegenstand erfassen")
    img_file = st.file_uploader("Bild hochladen oder Foto machen", type=["jpg", "jpeg", "png"])
     
    if img_file:
        # Bildverarbeitung & Kompression
        raw_img = Image.open(img_file)
        raw_img = ImageOps.exif_transpose(raw_img)
        if raw_img.width > 1920:
            raw_img = raw_img.resize((1920, int(raw_img.height * (1920/raw_img.width))), Image.LANCZOS)
         
        fname = f"{uuid.uuid4().hex}.jpg"
        fpath = os.path.join(UPLOAD_DIR, fname)
        raw_img.convert("RGB").save(fpath, "JPEG", quality=80)
         
        st.image(raw_img, width=400, caption="Hochgeladenes Bild")
         
        # KI Analyse
        with st.spinner("KI analysiert das Fundstück..."):
            res_label, res_conf = classify_image(fpath, model, labels)
         
        # === GEÄNDERT: Jetzt reicht 70 % Sicherheit ===
        if res_conf > 0.7:
            st.success(f"KI-Vorschlag: **{res_label}** ({res_conf:.1%})")
        else:
            st.warning("KI ist sich unsicher. Bitte manuell benennen.")

        with st.form("entry_form"):
            final_cat = st.text_input("Kategorie / Gegenstand", 
                                     value=res_label if res_conf > 0.7 else "")
            location = st.text_input("Fundort")
            tags = st.text_input("Weitere Merkmale (z.B. Marke, Farbe)")
             
            if st.form_submit_button("In Fundkiste speichern", type="primary"):
                db = load_db()
                db.append({
                    "id": str(uuid.uuid4()),
                    "category": final_cat,
                    "location": location,
                    "tags": tags,
                    "image_path": fpath,
                    "status": "verfügbar",
                    "found_at": datetime.now().isoformat(),
                    "claimed_at": None
                })
                save_db(db)
                st.balloons()
                st.rerun()

# --- TAB 2: DURCHSUCHEN ---
with tab2:
    items = load_db()
    if not items:
        st.info("Aktuell befinden sich keine Gegenstände in der Fundkiste.")
    else:
        # Neueste zuerst
        for item in reversed(items):
            with st.container():
                st.markdown('<div class="fund-card">', unsafe_allow_html=True)
                c1, c2 = st.columns([1, 2])
                 
                with c1:
                    st.image(item["image_path"], use_container_width=True)
                 
                with c2:
                    st.subheader(item["category"])
                    st.write(f"📍 **Ort:** {item['location']}")
                    st.write(f"🏷️ **Details:** {item['tags'] if item['tags'] else 'Keine'}")
                     
                    if item["status"] == "verfügbar":
                        st.markdown('<p class="status-verfuegbar">Status: Verfügbar</p>', unsafe_allow_html=True)
                        if st.button("Das ist meins!", key=f"claim_{item['id']}"):
                            st.session_state[f"conf_{item['id']}"] = True
                         
                        if st.session_state.get(f"conf_{item['id']}", False):
                            st.error("Gegenstand als 'Abgeholt' markieren? Er wird nach 48h gelöscht.")
                            if st.button("Ja, bestätigen", key=f"yes_{item['id']}"):
                                item["status"] = "abgeholt"
                                item["claimed_at"] = datetime.now().isoformat()
                                save_db(items)
                                st.session_state[f"conf_{item['id']}"] = False
                                st.rerun()
                    else:
                        st.markdown('<p class="status-abgeholt">Status: Abgeholt / Reserviert</p>', unsafe_allow_html=True)
                        # Countdown
                        claimed_dt = datetime.fromisoformat(item["claimed_at"])
                        diff = (claimed_dt + timedelta(hours=48)) - datetime.now()
                        hours = int(diff.total_seconds() // 3600)
                        if hours > 0:
                            st.write(f"⏳ Endgültige Löschung in ca. {hours} Stunden.")
                            st.progress(max(0.0, min(1.0, diff.total_seconds() / (48*3600))))
                         
                        if st.button("Rückgängig (Wieder verfügbar machen)", key=f"undo_{item['id']}"):
                            item["status"] = "verfügbar"
                            item["claimed_at"] = None
                            save_db(items)
                            st.rerun()
                 
                st.markdown('</div>', unsafe_allow_html=True)

# Cleanup am Ende des Runs
cleanup_expired_items()
