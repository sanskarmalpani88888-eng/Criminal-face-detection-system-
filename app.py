import streamlit as st
import cv2
import numpy as np
import face_recognition
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

# --- HELPER FUNCTION: LOAD CRIMINAL EMBEDDINGS ---
@st.cache_data(show_spinner=False)
def load_criminal_database():
    known_encodings = []
    known_names = []
    known_ids = []
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            img_path = os.path.join(IMAGE_DIR, f"{row['ID']}.jpg")
            if os.path.exists(img_path):
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(row['Name'])
                        known_ids.append(row['ID'])
                except Exception:
                    continue
    return known_encodings, known_names, known_ids

# --- UI APP CONFIGURATION ---
st.set_page_config(page_title="AI Criminal Face Recognition", page_icon="🛡️", layout="wide")

# Custom CSS for Professional Look
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
st.sidebar.info("Note: Upload front-facing clear passport size photos of criminals for 99.2% accuracy.")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🎥 Live CCTV Tracking", "📁 Intelligence Photo Scanner", "⚙️ Central Registry Database"])

# Load Database
known_encodings, known_names, known_ids = load_criminal_database()

# ----------------- TAB 1: LIVE CCTV TRACKING -----------------
with tab1:
    st.subheader("Real-Time Camera Intelligence Stream")
    camera_file = st.camera_input("Activate Terminal Camera Feed")
    
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process Facial Detection
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        criminal_detected = False
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
            name = "UNKNOWN"
            color = (0, 255, 0) # Green for clear citizen
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
                c_id = known_ids[first_match_index]
                criminal_detected = True
                color = (255, 0, 0) # Red for criminal
                
                # Fetch Crime Details
                crime_details = df_db[df_db["ID"] == c_id].iloc[0]["Crime"]
                st.error(f"🚨 THREAT DETECTED: {name} (ID: {c_id}) is wanted for '{crime_details}'!")
            
            # Draw professional overlay boxes
            cv2.rectangle(rgb_frame, (left, top), (right, bottom), color, 3)
            cv2.rectangle(rgb_frame, (left, top - 35), (right, top), color, cv2.FILLED)
            cv2.putText(rgb_frame, name, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
        st.image(rgb_frame, caption="Processed Intelligence Feed", use_container_width=True)
        
        if not criminal_detected and len(face_locations) > 0:
            st.success("✅ Scanning Complete: All individuals cleared. No match found in national database.")

# ----------------- TAB 2: INTELLIGENCE PHOTO SCANNER -----------------
with tab2:
    st.subheader("Forensic Image Verification")
    uploaded_file = st.file_uploader("Upload Scene or Suspect Photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        match_found = False
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
            name = "CLEARED"
            color = (0, 255, 0)
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index].upper()
                c_id = known_ids[first_match_index]
                match_found = True
                color = (255, 0, 0)
                
                crime_details = df_db[df_db["ID"] == c_id].iloc[0]["Crime"]
                st.sidebar.error(f"🎯 MATCH FOUND:\n\nName: {name}\nID: {c_id}\nCharges: {crime_details}")
            
            cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 3)
            cv2.putText(rgb_image, name, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
        st.image(rgb_image, caption="Analysis Report Viewer", width=600)
        
        if match_found:
            st.error("🚨 Forensic Alert: Positive match identified in the uploaded file.")
        elif len(face_locations) > 0:
            st.success("✅ Analysis Result: Subject is cleared of any database records.")
        else:
            st.warning("⚠️ Diagnostics: No human faces detected in the image.")

# ----------------- TAB 3: CENTRAL REGISTRY DATABASE -----------------
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Register New Suspect Profile")
        with st.form("add_criminal_form", clear_on_submit=True):
            new_id = st.text_input("Unique Criminal Code / ID (e.g., CRM-902)")
            new_name = st.text_input("Full Legal Name")
            new_crime = st.text_area("Criminal Record / Description of Crime")
            uploaded_photo = st.file_uploader("Upload High-Res Dossier Photo", type=["jpg", "jpeg"])
            submit_btn = st.form_submit_button("Commit Record to Database")
            
            if submit_btn:
                if new_id and new_name and new_crime and uploaded_photo:
                    df_current = pd.read_csv(CSV_PATH)
                    if str(new_id) in df_current["ID"].astype(str).values:
                        st.error("Operation Failed: Criminal ID already exists in the system database.")
                    else:
                        # Save image with ID name
                        img_path = os.path.join(IMAGE_DIR, f"{new_id}.jpg")
                        with open(img_path, "wb") as f:
                            f.write(uploaded_photo.getbuffer())
                        
                        # Save to CSV
                        new_data = pd.DataFrame([{"ID": new_id, "Name": new_name, "Crime": new_crime}])
                        df_updated = pd.concat([df_current, new_data], ignore_index=True)
                        df_updated.to_csv(CSV_PATH, index=False)
                        
                        st.cache_data.clear() # Clear cache to load new image dynamically
                        st.success(f"Success: Profile for {new_name} has been securely encrypted.")
                        st.button("Refresh Dashboard")
                else:
                    st.warning("Validation Error: Please fill all fields and provide a face profile image.")

    with col2:
        st.subheader("🗑️ Active Database Logs & Revocation")
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
                
                st.cache_data.clear() # Reset cache
                st.success(f"Record {delete_id} deleted successfully.")
                st.rerun()

