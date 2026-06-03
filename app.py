import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os

# --- DIRECTORY & DATABASE INITIALIZATION ---
IMAGE_DIR = "criminal_images"
CSV_PATH = "criminals.csv"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
    df = pd.DataFrame(columns=["ID", "Name", "Crime"])
    df.to_csv(CSV_PATH, index=False)

# OpenCV Built-in Face Detector Load
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- LIGHTWEIGHT FACE MATCHING LOGIC ---
def match_face(target_face_gray):
    if not os.path.exists(CSV_PATH):
        return None
    
    df = pd.read_csv(CSV_PATH)
    best_score = 0
    matched_id = None
    target_face_gray = cv2.resize(target_face_gray, (150, 150))
    
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{row['ID']}.jpg")
        if os.path.exists(img_path):
            db_img = cv2.imread(img_path)
            db_gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
            db_faces = face_cascade.detectMultiScale(db_gray, 1.3, 5)
            
            if len(db_faces) > 0:
                (x, y, w, h) = db_faces[0]
                db_face_cropped = cv2.resize(db_gray[y:y+h, x:x+w], (150, 150))
                
                result = cv2.matchTemplate(target_face_gray, db_face_cropped, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score and max_val > 0.45:
                    best_score = max_val
                    matched_id = row['ID']
                    
    return matched_id

# --- UI APP CONFIGURATION ---
st.set_page_config(page_title="AI Criminal Face Recognition", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; font-weight: bold; color: #E74C3C; text-align: center; margin-bottom: 0px; }
    .subtitle { font-size:18px !important; text-align: center; color: #7F8C8D; margin-bottom: 30px; }
    </style>
""", unsafe_index=True)

st.markdown('<div class="main-title">🛡️ AI-POWERED CRIMINAL FACE IDENTIFICATION SYSTEM</div>', unsafe_index=True)
st.markdown('<div class="subtitle">Advanced Facial Recognition & Surveillance Dashboard | Final Year Project</div>', unsafe_index=True)

# --- SIDEBAR METRICS ---
df_db = pd.read_csv(CSV_PATH)
st.sidebar.title("📊 System Analytics")
st.sidebar.metric(label="Total Registered Criminals", value=len(df_db))
st.sidebar.metric(label="System Status", value="ACTIVE", delta="Secure")
st.sidebar.divider()

tab1, tab2, tab3 = st.tabs(["🎥 Live CCTV Tracking", "📁 Intelligence Photo Scanner", "⚙️ Central Registry Database"])

# ----------------- TAB 1: LIVE CCTV TRACKING -----------------
with tab1:
    st.subheader("Real-Time Camera Intelligence Stream")
    camera_file = st.camera_input("Activate Terminal Camera Feed")
    
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        criminal_detected = False
        
        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            matched_id = match_face(cropped_face)
            
            if matched_id:
                details = df_db[df_db["ID"] == matched_id].iloc[0]
                name = details["Name"].upper()
                crime_details = details["Crime"]
                color = (255, 0, 0)
                criminal_detected = True
                st.error(f"🚨 THREAT DETECTED: {name} (ID: {matched_id}) is wanted for '{crime_details}'!")
            else:
                name = "UNKNOWN"
                color = (0, 255, 0)
            
            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(rgb_frame, (x, y - 35), (x+w, y), color, cv2.FILLED)
            cv2.putText(rgb_frame, name, (x + 6, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
        st.image(rgb_frame, caption="Processed Intelligence Feed", use_container_width=True)
        
        if not criminal_detected and len(faces) > 0:
            st.success("✅ Scanning Complete: All individuals cleared.")

# ----------------- TAB 2: INTELLIGENCE PHOTO SCANNER -----------------
with tab2:
    st.subheader("Forensic Image Verification")
    uploaded_file = st.file_uploader("Upload Scene or Suspect Photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        match_found = False
        
        for (x, y, w, h) in faces:
            cropped = gray_img[y:y+h, x:x+w]
            matched_id = match_face(cropped)
            
            if matched_id:
                details = df_db[df_db["ID"] == matched_id].iloc[0]
                name = details["Name"].upper()
                match_found = True
                color = (255, 0, 0)
                st.sidebar.error(f"🎯 MATCH FOUND:\n\nName: {name}\nID: {matched_id}\nCharges: {details['Crime']}")
            else:
                name = "CLEARED"
                color = (0, 255, 0)
                
            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), color, 3)
            cv2.putText(rgb_image, name, (x + 6, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
        st.image(rgb_image, caption="Analysis Report Viewer", width=600)

# ----------------- TAB 3: CENTRAL REGISTRY DATABASE -----------------
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Register New Suspect Profile")
        with st.form("add_criminal_form", clear_on_submit=True):
            new_id = st.text_input("Unique Criminal ID (e.g., CRM-101)")
            new_name = st.text_input("Full Legal Name")
            new_crime = st.text_area("Criminal Record / Description")
            uploaded_photo = st.file_uploader("Upload Profile Photo", type=["jpg", "jpeg"])
            submit_btn = st.form_submit_button("Commit Record to Database")
            
            if submit_btn:
                if new_id and new_name and new_crime and uploaded_photo:
                    df_current = pd.read_csv(CSV_PATH)
                    if str(new_id) in df_current["ID"].astype(str).values:
                        st.error("Operation Failed: Criminal ID already exists.")
                    else:
                        img_path = os.path.join(IMAGE_DIR, f"{new_id}.jpg")
                        with open(img_path, "wb") as f:
                            f.write(uploaded_photo.getbuffer())
                        
                        new_data = pd.DataFrame([{"ID": new_id, "Name": new_name, "Crime": new_crime}])
                        df_updated = pd.concat([df_current, new_data], ignore_index=True)
                        df_updated.to_csv(CSV_PATH, index=False)
                        st.success(f"Success: Saved {new_name}")
                        st.rerun()
                else:
                    st.warning("Please fill all fields.")

    with col2:
        st.subheader("🗑️ Active Database Logs")
        df_display = pd.read_csv(CSV_PATH)
        st.dataframe(df_display, use_container_width=True, height=250)
        
        delete_id = st.selectbox("Select Target ID to Delete", [""] + list(df_display["ID"].values))
        
        if st.button("Delete Selected Record", type="primary"):
            if delete_id:
                df_display = df_display[df_display["ID"] != delete_id]
                df_display.to_csv(CSV_PATH, index=False)
                img_path = os.path.join(IMAGE_DIR, f"{delete_id}.jpg")
                if os.path.exists(img_path):
                    os.remove(img_path)
                st.success(f"Record {delete_id} deleted.")
                st.rerun()
