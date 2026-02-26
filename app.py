import streamlit as st
import os
import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import shutil

# Verhindert die Nutzung der lokalen GPU (erzwingt CPU-Nutzung für Cloud-Kompatibilität)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.keras.models as models

# --- KONFIGURATION ---
st.set_page_config(page_title="Digitale Fundkiste", page_icon="📦", layout="wide")

UPLOAD_DIR = "uploads"
DB_FILE = "database.json"
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

# --- CSS STYLING ---
# Modernes App-Design mit Karten-Layout
st.markdown("""
<style>
    .fund-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #f0f2f6;
    }
    .status-verfuegbar { color: #2e7d32; font-weight: bold; }
    .status-abgeholt { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HILFSFUNKTIONEN ---

def init_system():
    """Erstellt Ordner, Dummy-Daten und bereinigt alte Einträge beim Start."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    if not os.path.exists(DB_FILE):
        # Erstelle Dummy-Datenbank
        dummy_data = []
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=4)
            
    cleanup_expired_items()

def load_db():
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def cleanup_expired_items():
    """Löscht Einträge und Bilder, deren 48h-Frist abgelaufen ist."""
    db = load_db()
    now = datetime.now()
    active_items = []
    
    for item in db:
        keep = True
        if item.get("status") == "abgeholt" and item.get("claimed_at"):
            claim_time = datetime.fromisoformat(item["claimed_at"])
            if now > claim_time + timedelta(hours=48):
                # Frist abgelaufen -> Löschen
                keep = False
                try:
                    if os.path.exists(item["image_path"]):
                        os.remove(item["image_path"])
                except Exception as e:
                    st.error(f"Fehler beim Löschen des Bildes: {e}")
        
        if keep:
            active_items.append(item)
            
    if len(active_items) != len(db):
        save_db(active_items)

def process_and_save_image(image_file):
    """Speichert Bild in Full HD (max 1920px Breite) um Platz zu sparen."""
    img = Image.open(image_file)
    
    # Exif-Rotation korrigieren (falls vom Smartphone hochgeladen)
    img = ImageOps.exif_transpose(img)
    
    if img.width > 1920:
        ratio = 1920 / img.width
        new_height = int(img.height * ratio)
        img = img.resize((1920, new_height), Image.Resampling.LANCZOS)
    
    # RGB Modus erzwingen (falls PNG mit Transparenz)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    img.save(filepath, "JPEG", quality=85)
    return filepath

@st.cache_resource
def load_ai_model():
    """Lädt das Teachable Machine Modell."""
    try:
        model = models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, "r") as f:
            class_names = f.readlines()
        return model, class_names
    except Exception as e:
        return None, None

def classify_image(image_path, model, class_names):
    """Klassifiziert das Bild mit dem Teachable Machine Modell."""
    if not model:
        return "Unbekannt", 0.0
        
    # Bildvorbereitung für Teachable Machine (224x224)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip().split(" ", 1)[-1] # Format "0 Jacke" -> "Jacke"
    confidence_score = prediction[0][index]
    
    return class_name, float(confidence_score)

# --- SYSTEM INITIALISIEREN ---
init_system()
model, class_names = load_ai_model()

# --- UI ---
st.title("📦 Digitale Fundkiste")
st.write("Verlorene Gegenstände intelligent erfassen und wiederfinden.")

tab1, tab2 = st.tabs(["📸 Fund melden", "🔍 Fundkiste durchsuchen"])

# ==========================================
# TAB 1: FUND MELDEN
# ==========================================
with tab1:
    st.header("Neuen Fund melden")
    
    input_method = st.radio("Wie möchtest du das Bild hinzufügen?", ["Kamera", "Datei-Upload"], horizontal=True)
    
    img_file = None
    if input_method == "Kamera":
        img_file = st.camera_input("Mache ein Foto vom gefundenen Gegenstand")
    else:
        img_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
        
    if img_file is not None:
        st.image(img_file, caption="Vorschau", use_container_width=True)
        
        with st.spinner("KI analysiert das Bild (in der Cloud)..."):
            # Temporär speichern für Analyse
            temp_path = process_and_save_image(img_file)
            predicted_class, confidence = classify_image(temp_path, model, class_names)
            
        st.success(f"KI-Erkennung: **{predicted_class}** (Sicherheit: {confidence:.0%})")
        
        with st.form("fund_form"):
            st.write("Details zum Fund")
            kategorie = st.text_input("Kategorie", value=predicted_class)
            tags = st.text_input("Zusätzliche Schlagworte (kommagetrennt)", placeholder="z.B. blau, Adidas, Größe M")
            ort = st.text_input("Fundort", placeholder="z.B. Mensa, Bibliothek")
            
            submit = st.form_submit_button("Fund in die Kiste legen", type="primary")
            
            if submit:
                # In Datenbank speichern
                db = load_db()
                new_item = {
                    "id": str(uuid.uuid4()),
                    "category": kategorie,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "location": ort,
                    "image_path": temp_path,
                    "status": "verfügbar",
                    "found_at": datetime.now().isoformat(),
                    "claimed_at": None
                }
                db.append(new_item)
                save_db(db)
                st.success("Erfolgreich hinzugefügt! Du findest es nun in der Fundkiste.")
                st.balloons()

# ==========================================
# TAB 2: FUNDKISTE DURCHSUCHEN
# ==========================================
with tab2:
    st.header("Aktuelle Fundstücke")
    
    db = load_db()
    if not db:
        st.info("Die Fundkiste ist momentan leer.")
    
    # Grid Layout für die Galerie erstellen (3 Spalten)
    cols = st.columns(3)
    
    for index, item in enumerate(reversed(db)): # Neueste zuerst
        col = cols[index % 3]
        
        with col:
            # HTML Container für den Karten-Look
            st.markdown(f'<div class="fund-card">', unsafe_allow_html=True)
            
            st.image(item["image_path"], use_container_width=True)
            st.subheader(item["category"])
            st.write(f"📍 **Ort:** {item['location']}")
            st.write(f"🏷️ **Tags:** {', '.join(item['tags']) if item['tags'] else '-'}")
            
            # Datumsformatierung mit Pandas (als Helfer)
            found_date = pd.to_datetime(item["found_at"]).strftime("%d.%m.%Y %H:%M")
            st.caption(f"Gefunden am: {found_date}")
            
            if item["status"] == "verfügbar":
                st.markdown('<p class="status-verfuegbar">Status: Verfügbar</p>', unsafe_allow_html=True)
                
                # Bestätigungslogik via Session State
                claim_key = f"claim_btn_{item['id']}"
                if st.button("Das ist meins!", key=claim_key, use_container_width=True):
                    st.session_state[f"confirm_{item['id']}"] = True
                    st.rerun()
                    
                if st.session_state.get(f"confirm_{item['id']}", False):
                    st.warning("Bist du sicher? Der Gegenstand wird in 48 Stunden endgültig gelöscht.")
                    c1, c2 = st.columns(2)
                    if c1.button("Ja, abholen", key=f"yes_{item['id']}", type="primary"):
                        # Status aktualisieren
                        for db_item in db:
                            if db_item["id"] == item["id"]:
                                db_item["status"] = "abgeholt"
                                db_item["claimed_at"] = datetime.now().isoformat()
                        save_db(db)
                        st.session_state[f"confirm_{item['id']}"] = False
                        st.rerun()
                    if c2.button("Abbrechen", key=f"no_{item['id']}"):
                        st.session_state[f"confirm_{item['id']}"] = False
                        st.rerun()

            elif item["status"] == "abgeholt":
                st.markdown('<p class="status-abgeholt">Status: Abgeholt</p>', unsafe_allow_html=True)
                
                # 48-Stunden Countdown Logik
                claim_time = datetime.fromisoformat(item["claimed_at"])
                now = datetime.now()
                time_elapsed = now - claim_time
                time_remaining = timedelta(hours=48) - time_elapsed
                
                if time_remaining.total_seconds() > 0:
                    hours_left = int(time_remaining.total_seconds() // 3600)
                    mins_left = int((time_remaining.total_seconds() % 3600) // 60)
                    
                    st.write(f"⏳ **Wird gelöscht in:** {hours_left}h {mins_left}m")
                    # Progress bar (0.0 bis 1.0)
                    progress = 1.0 - (time_elapsed.total_seconds() / (48 * 3600))
                    st.progress(max(0.0, min(1.0, progress)))
                    
                    # Wiederherstellungs-Button
                    if st.button("Löschen stoppen / Rückgängig", key=f"undo_{item['id']}", help="Setzt den Status wieder auf 'verfügbar'"):
                        for db_item in db:
                            if db_item["id"] == item["id"]:
                                db_item["status"] = "verfügbar"
                                db_item["claimed_at"] = None
                        save_db(db)
                        st.rerun()
                else:
                    st.error("Dieser Eintrag wird beim nächsten Seitenaufbau endgültig gelöscht.")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Automatisches Cleanup bei jedem Reload, um abgelaufene Items zu löschen
cleanup_expired_items()
