import streamlit as st
import os

# --- CLOUD & KOMPATIBILITÄTS-EINSTELLUNGEN ---
# 1. Erzwingt CPU-Nutzung (spart Ressourcen in der Cloud)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2. WICHTIG: Aktiviert Kompatibilität für .h5 Modelle in neuen TF-Versionen
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import tensorflow as tf

# --- KONFIGURATION ---
st.set_page_config(page_title="Digitale Fundkiste", page_icon="📦", layout="wide")

UPLOAD_DIR = "uploads"
DB_FILE = "database.json"
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

# --- CSS STYLING ---
st.markdown("""
<style>
    .fund-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #f0f2f6;
        color: #1f1f1f;
    }
    .status-verfuegbar { color: #2e7d32; font-weight: bold; }
    .status-abgeholt { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HILFSFUNKTIONEN ---

def init_system():
    """Erstellt Ordner und Datenbank, falls nicht vorhanden."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4)
    
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
    """Löscht Einträge nach Ablauf der 48h-Frist."""
    db = load_db()
    now = datetime.now()
    updated_db = []
    changed = False
    
    for item in db:
        if item.get("status") == "abgeholt" and item.get("claimed_at"):
            claim_time = datetime.fromisoformat(item["claimed_at"])
            if now > claim_time + timedelta(hours=48):
                if os.path.exists(item["image_path"]):
                    os.remove(item["image_path"])
                changed = True
                continue
        updated_db.append(item)
            
    if changed:
        save_db(updated_db)

def process_and_save_image(image_file):
    """Komprimiert das Bild auf Full HD Breite."""
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    
    if img.width > 1920:
        ratio = 1920 / img.width
        new_height = int(img.height * ratio)
        img = img.resize((1920, new_height), Image.Resampling.LANCZOS)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    img.save(filepath, "JPEG", quality=85)
    return filepath

@st.cache_resource
def load_ai_model():
    """Lädt das Modell sicher."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, "r") as f:
            class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"Modell-Ladefehler: {e}")
        return None, None

def classify_image(image_path, model, class_names):
    """KI Klassifizierung."""
    if not model: return "Unbekannt", 0.0
    
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype(np.float32)
    normalized_img = (img_array / 127.5) - 1
    data = np.expand_dims(normalized_img, axis=0)

    prediction = model.predict(data, verbose=0)
    idx = np.argmax(prediction)
    return class_names[idx], prediction[0][idx]

# --- APP START ---
init_system()
model, labels = load_ai_model()

st.title("📦 Digitale Fundkiste")

tab1, tab2 = st.tabs(["📸 Fund melden", "🔍 Fundkiste durchsuchen"])

# --- TAB 1: MELDEN ---
with tab1:
    st.header("Neuen Fund erfassen")
    source = st.radio("Quelle:", ["Kamera", "Upload"], horizontal=True)
    img_input = st.camera_input("Foto") if source == "Kamera" else st.file_uploader("Datei", type=["jpg", "png"])
    
    if img_input:
        path = process_and_save_image(img_input)
        label, conf = classify_image(path, model, labels)
        
        st.success(f"Erkannt: **{label}** ({conf:.0%})")
        
        with st.form("add_form"):
            cat = st.text_input("Kategorie", value=label)
            loc = st.text_input("Fundort", placeholder="Wo wurde es gefunden?")
            tags = st.text_input("Schlagworte (mit Komma)")
            
            if st.form_submit_button("Speichern"):
                db = load_db()
                db.append({
                    "id": str(uuid.uuid4()), "category": cat, "location": loc,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "image_path": path, "status": "verfügbar",
                    "found_at": datetime.now().isoformat(), "claimed_at": None
                })
                save_db(db)
                st.success("In Fundkiste gespeichert!")
                st.rerun()

# --- TAB 2: DURCHSUCHEN ---
with tab2:
    db = load_db()
    if not db:
        st.info("Keine Gegenstände vorhanden.")
    else:
        cols = st.columns(3)
        for i, item in enumerate(reversed(db)):
            with cols[i % 3]:
                st.markdown('<div class="fund-card">', unsafe_allow_html=True)
                st.image(item["image_path"])
                st.subheader(item["category"])
                st.write(f"📍 {item['location']}")
                
                if item["status"] == "verfügbar":
                    st.markdown('<p class="status-verfuegbar">Status: Verfügbar</p>', unsafe_allow_html=True)
                    if st.button("Das ist meins!", key=f"btn_{item['id']}"):
                        st.session_state[f"confirm_{item['id']}"] = True
                    
                    if st.session_state.get(f"confirm_{item['id']}", False):
                        if st.button("Bestätigen", key=f"confirm_btn_{item['id']}", type="primary"):
                            for d_item in db:
                                if d_item["id"] == item["id"]:
                                    d_item["status"] = "abgeholt"
                                    d_item["claimed_at"] = datetime.now().isoformat()
                            save_db(db)
                            st.rerun()
                else:
                    st.markdown('<p class="status-abgeholt">Status: Abgeholt</p>', unsafe_allow_html=True)
                    claimed_time = datetime.fromisoformat(item["claimed_at"])
                    remaining = (claimed_time + timedelta(hours=48)) - datetime.now()
                    
                    if remaining.total_seconds() > 0:
                        st.write(f"⏳ Löschung in: {int(remaining.total_seconds()//3600)}h")
                        progress = max(0.0, remaining.total_seconds() / (48*3600))
                        st.progress(progress)
                        if st.button("Rückgängig", key=f"undo_{item['id']}"):
                            for d_item in db:
                                if d_item["id"] == item["id"]:
                                    d_item["status"] = "verfügbar"
                                    d_item["claimed_at"] = None
                            save_db(db)
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
