import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime

# --- DIRECTORY & DATABASE INITIALIZATION ---
IMAGE_DIR = "criminal_images"
CSV_PATH = "criminals.csv"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Initialize Session State for Live Surveillance Logs
if "logs" not in st.session_state:
    st.session_state.logs = []

# Generic Face Silhouette/Placeholder (Base64) to prevent crashes for fictional characters
DEFAULT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# --- 🚀 APKI UPLOADED PHOTOS KA BASE64 ENCODING 🚀 ---
B64_YOUR_PHOTO_1 = "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"

B64_YOUR_PHOTO_2 = "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.AAnNf3zf8ABBbV21z/AII+fAmdn8wxeH5Id2/dnZdToO3B46dv1r+BmP8A1g/Gv7wf+De7w5deE/8AAgfteO6+3WMZkiVbSIsJA3IcCv4Gv+C602nzf8FdP2gV0nSrfRdNt/GE9vbaZbnMdlGkaIIRwOEVQuABjFf3faiNTheS3k8wxyOvlhBlQvAr+EL/AIOJPCjeD/8Ags/+0RbG1ubRbjxbPcpHONrEMiNvx6NuLAdgaqXwomPxaHxdRRRUGgUUUUAFFFFABRRQQAAnNf3zf8ABBbV21z/AII+fAmdn8wxeH5Id2/dnZdToO3B46dv1r+BmP81g/Gv7wf+De7w5deE/8Agjz8DbG7UpL/AGAsuCpT780smMGl9pfP8gPteO6+3WMZkiVbSIsJA3IcCv4Gv+C602nzf8FdP2gV0nSrfRdNt/GE9vbaZbnMdlGkaIIRwOEVQuABjFf3faiNTheS3k8wxyOvlhBlQvAr+EL/AIOJPCjeD/8Ags/+0RbG1ubRbjxbPcpHONrEMiNvx6NuLAdgaqXwomPxaHxdRRRUGgUUUUAFFFFQoovpqgP31sf+D13WIgGuPgzfXNyS4LL4giWNlyNhx9lOCOcnB6cCvxj/b3/atTdtuLg79Rvbi9unbLZaZ2eQgnsWYnp1ArxyihaRsFtbhRRRQAUUUUAf//Z"

# --- 🎬 FULL MIXED DATABASE (OLD HEROES + YOUR PHOTOS) ---
PRELOADED_CRIMINALS = {
    "CRM-101": {
        "name": "GABBAR SINGH", 
        "crime": "Dacoity, Extortion, and Terrorizing Ramgarh Villagers.",
        "b64": DEFAULT_B64
    },
    "CRM-102": {
        "name": "MOGAMBO", 
        "crime": "Illegal Weapon Manufacturing & Attempting Global Terrorism.",
        "b64": DEFAULT_B64
    },
    "CRM-103": {
        "name": "DON (VIJAY)", 
        "crime": "International Drug Smuggling. Wanted by Police of 11 Countries.",
        "b64": DEFAULT_B64
    },
    "CRM-104": {
        "name": "WALTER WHITE (HEISENBERG)", 
        "crime": "Methamphetamine Manufacturing & Cartel Operations in Albuquerque.",
        "b64": DEFAULT_B64
    },
    "CRM-105": {
        "name": "THE PROFESSOR", 
        "crime": "Mastermind behind the Royal Mint of Spain Multi-Billion Heist.",
        "b64": DEFAULT_B64
    },
    "CRM-106": {
        "name": "CHIEF SUSPECT ALPHA (PHOTO 1)", 
        "crime": "High-Value Cyber Security Breach and Neural Network Infiltration.",
        "b64": B64_YOUR_PHOTO_1
    },
    "CRM-107": {
        "name": "CHIEF SUSPECT BETA (PHOTO 2)", 
        "crime": "Advanced AI System Manipulation and Algorithmic Surveillance Evasion.",
        "b64": B64_YOUR_PHOTO_2
    }
}

# Automatically seed CSV file with complete data if it doesn't exist
if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
    df_init = pd.DataFrame([{"ID": k, "Name": v["name"], "Crime": v["crime"]} for k, v in PRELOADED_CRIMINALS.items()])
    df_init.to_csv(CSV_PATH, index=False)

# Recreate all photo assets in folder automatically on startup
for c_id, data in PRELOADED_CRIMINALS.items():
    img_path = os.path.join(IMAGE_DIR, f"{c_id}.jpg")
    if not os.path.exists(img_path):
        try:
            img_bytes = base64.b64decode(data["b64"])
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        except:
            pass

# OpenCV Face Cascade Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- ADVANCED FACE MATCHING WITH SCORE SYSTEM ---
def match_face(target_face_gray):
    if not os.path.exists(CSV_PATH):
        return None, 0
    
    df = pd.read_csv(CSV_PATH)
    best_score = 0
    matched_id = None
    target_face_gray = cv2.resize(target_face_gray, (150, 150))
    
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{row['ID']}.jpg")
        if os.path.exists(img_path):
            db_img = cv2.imread(img_path)
            if db_img is None: continue
            db_gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
            
            db_faces = face_cascade.detectMultiScale(db_gray, 1.3, 5)
            if len(db_faces) > 0:
                (x, y, w, h) = db_faces[0]
                db_face_cropped = cv2.resize(db_gray[y:y+h, x:x+w], (150, 150))
            else:
                db_face_cropped = cv2.resize(db_gray, (150, 150))
                
            result = cv2.matchTemplate(target_face_gray, db_face_cropped, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score and max_val > 0.40:
                best_score = max_val
                matched_id = row['ID']
                
    return matched_id, best_score

# --- CYBERPUNK HUD TARGET RETICLE DRAWING ---
def draw_target_reticle(img, x, y, w, h, color):
    length = int(w * 0.25)
    thick = 3
    # Top-Left Corner
    cv2.line(img, (x, y), (x + length, y), color, thick)
    cv2.line(img, (x, y), (x, y + length), color, thick)
    # Top-Right Corner
    cv2.line(img, (x + w, y), (x + w - length, y), color, thick)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thick)
    # Bottom-Left Corner
    cv2.line(img, (x, y + h), (x + length, y + h), color, thick)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thick)
    # Bottom-Right Corner
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thick)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thick)

# --- UI APP INTERFACE ---
st.set_page_config(page_title="AI Criminal Terminal", page_icon="🛡️", layout="wide")

# Styling to convert standard Streamlit into a High-Tech Dashboard
st.markdown("""
    <style>
    .main-title { font-size:40px !important; font-weight: 800; color: #E74C3C; text-align: center; letter-spacing: 1px; }
    .subtitle { font-size:18px !important; text-align: center; color: #7F8C8D; margin-bottom: 25px; }
    .terminal-card { background-color: #1E222B; border-left: 5px solid #E74C3C; padding: 15px; border-radius: 4px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ ADVANCED CRIMINAL FACE IDENTIFICATION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Biometric Surveillance Core Engine | Fictional & Real Threat Intel Vault</div>', unsafe_allow_html=True)

# Sidebar System Analytics
df_db = pd.read_csv(CSV_PATH)
st.sidebar.title("📊 SYSTEM LOG PANEL")
st.sidebar.markdown("---")
st.sidebar.metric(label="Total Registered Dossiers", value=len(df_db))
st.sidebar.success("⚡ CORE BIOMETRIC ENGINE: ACTIVE")

# Live Feed Log System
st.sidebar.markdown("### 📜 LIVE RADAR LOGS")
if st.session_state.logs:
    for log in reversed(st.session_state.logs[-5:]):
        st.sidebar.caption(log)
else:
    st.sidebar.caption("Awaiting camera frame capture...")

# Main Tabs Setup
tab1, tab2, tab3 = st.tabs(["🎥 Live CCTV Tracking", "📁 Intelligence Photo Scanner", "⚙️ Central Identity Vault"])

# --- TAB 1: LIVE CCTV TRACKING ---
with tab1:
    st.subheader("Real-Time Camera Intelligence Stream")
    camera_file = st.camera_input("Activate Terminal Camera Feed")
    
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            matched_id, score = match_face(cropped_face)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            if matched_id:
                details = df_db[df_db["ID"] == matched_id].iloc[0]
                name = details["Name"].upper()
                crime_details = details["Crime"]
                color = (231, 76, 60)  # Red Alert Brackets
                
                log_msg = f"[{timestamp}] 🚨 MATCH: {name} ({int(score*100)}% Match)"
                if log_msg not in st.session_state.logs: st.session_state.logs.append(log_msg)
                
                st.error(f"🚨 THREAT DETECTED: {name} | Security Match Confidence: {int(score*100)}%")
                st.warning(f"ℹ️ Intelligence Charges: {crime_details}")
            else:
                name = "UNKNOWN CITIZEN"
                color = (46, 204, 113)  # Green Secure Brackets
                log_msg = f"[{timestamp}] 🟢 Scan: Unknown face clear."
                if log_msg not in st.session_state.logs: st.session_state.logs.append(log_msg)
            
            # Apply Cyberpunk Brackets and Identity Text overlay
            draw_target_reticle(rgb_frame, x, y, w, h, color)
            cv2.putText(rgb_frame, name, (x, y - 12), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
            
        st.image(rgb_frame, caption="Processed Video Intercept Buffer", use_container_width=True)

# --- TAB 2: INTELLIGENCE PHOTO SCANNER ---
with tab2:
    st.subheader("Forensic Static Image Verification")
    uploaded_file = st.file_uploader("Upload Forensic Scene Photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        
        if len(faces) == 0:
            st.info("No facial structure captured in the uploaded asset frame.")
            
        for (x, y, w, h) in faces:
            cropped = gray_img[y:y+h, x:x+w]
            matched_id, score = match_face(cropped)
            
            if matched_id:
                details = df_db[df_db["ID"] == matched_id].iloc[0]
                st.markdown(f"""
                <div class="terminal-card">
                    <h3 style='color:#E74C3C;margin:0;'>🎯 DOSSIER RECORD FOUND [{int(score*100)}% Match]</h3>
                    <p style='margin:5px 0 0 0;'><b>Name:</b> {details['Name']}<br><b>Dossier ID:</b> {matched_id}<br><b>Active Record:</b> {details['Crime']}</p>
                </div>
                """, unsafe_allow_html=True)
                color = (231, 76, 60)
            else:
                st.success("✅ Frame Clearance: Structure scanned, zero database correlation.")
                color = (46, 204, 113)
                
            draw_target_reticle(rgb_image, x, y, w, h, color)
            
        st.image(rgb_image, caption="Evidence Analysis Viewer", width=550)

# --- TAB 3: CENTRAL REGISTRY DATABASE ---
with tab3:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📥 Register New Suspect Profile")
        with st.form("add_criminal_form_v3", clear_on_submit=True):
            new_id = st.text_input("Unique ID (e.g., CRM-108)")
            new_name = st.text_input("Full Legal Name")
            new_crime = st.text_area("Criminal Record / Charges")
            uploaded_photo = st.file_uploader("Upload reference photo", type=["jpg", "jpeg"])
            submit_btn = st.form_submit_button("Commit Record to Vault")
            
            if submit_btn and new_id and new_name and new_crime and uploaded_photo:
                df_current = pd.read_csv(CSV_PATH)
                if str(new_id) in df_current["ID"].astype(str).values:
                    st.error("Dossier ID already exists in central logs.")
                else:
                    img_path = os.path.join(IMAGE_DIR, f"{new_id}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(uploaded_photo.getbuffer())
                    new_data = pd.DataFrame([{"ID": new_id, "Name": new_name, "Crime": new_crime}])
                    pd.concat([df_current, new_data], ignore_index=True).to_csv(CSV_PATH, index=False)
                    st.success(f"Dossier for {new_name} added successfully!")
                    st.rerun()

    with col2:
        st.subheader("🗑️ Active Database Registry Logs")
        df_display = pd.read_csv(CSV_PATH)
        st.dataframe(df_display, use_container_width=True, height=260)
        delete_id = st.selectbox("Select Target ID to Delete", [""] + list(df_display["ID"].values))
        if st.button("Delete Selected Profile", type="primary") and delete_id:
            df_display = df_display[df_display["ID"] != delete_id]
            df_display.to_csv(CSV_PATH, index=False)
            img_path = os.path.join(IMAGE_DIR, f"{delete_id}.jpg")
            if os.path.exists(img_path): os.remove(img_path)
            st.success(f"Profile {delete_id} successfully wiped.")
            st.rerun()
