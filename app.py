import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import re

# =================================================================
# CABLE WIRING STANDARDS
# =================================================================
T568A = ["White/Green", "Green", "White/Orange", "Blue", "White/Blue", "Orange", "White/Brown", "Brown"]
T568B = ["White/Orange", "Orange", "White/Green", "Blue", "White/Blue", "Green", "White/Brown", "Brown"]

# =================================================================
# SESSION STATE INITIALIZATION
# =================================================================
if 'current_end' not in st.session_state:
    st.session_state.current_end = 'A'
if 'end_a' not in st.session_state:
    st.session_state.end_a = None
if 'end_b' not in st.session_state:
    st.session_state.end_b = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None

# =================================================================
# OPENCV COLOR & CONTOUR LOGIC
# =================================================================
def classify_wire(wire_roi):
    if wire_roi.size == 0:
        return "Unknown"

    lab = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    wire_roi = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2HSV)
    
    masks = {
        "Orange": cv2.inRange(hsv, np.array([5, 80, 60]), np.array([25, 255, 255])),
        "Green": cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])),
        "Blue": cv2.inRange(hsv, np.array([90, 50, 40]), np.array([130, 255, 255])),
        "Brown": cv2.inRange(hsv, np.array([2, 40, 20]), np.array([20, 255, 180]))
    }

    total_pixels = max(hsv.shape[0] * hsv.shape[1], 1)
    scores = {name: cv2.countNonZero(mask) / total_pixels for name, mask in masks.items()}
    best_colour = max(scores, key=scores.get)

    if scores[best_colour] < 0.20:
        return "Unknown"

    white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 70, 255]))
    if (cv2.countNonZero(white_mask) / total_pixels) > 0.12:
        return f"White/{best_colour}"

    return best_colour

def detect_wire_sequence_contours(frame):
    height, width = frame.shape[:2]
    roi_y1, roi_y2 = int(height * 0.10), int(height * 0.85)
    roi_x1, roi_x2 = int(width * 0.05), int(width * 0.95)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi.size == 0:
        return [], frame

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    wire_mask = cv2.bitwise_or(cv2.inRange(hsv[:, :, 1], 30, 255), cv2.inRange(hsv[:, :, 2], 150, 255))
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(wire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wire_blobs = []
    min_area = (roi.shape[0] * roi.shape[1]) * 0.0008

    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            if h > 15 and w < int(roi.shape[1] * 0.30):
                wire_blobs.append((x, y, w, h))

    wire_blobs = sorted(wire_blobs, key=lambda b: b[0])
    display_frame = frame.copy()
    sequence = []

    for (x, y, w, h) in wire_blobs:
        abs_x, abs_y = roi_x1 + x, roi_y1 + y
        color = classify_wire(frame[abs_y:abs_y+h, abs_x:abs_x+w])
        if color != "Unknown":
            sequence.append(color)
            cv2.rectangle(display_frame, (abs_x, abs_y), (abs_x + w, abs_y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, str(len(sequence)), (abs_x, abs_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(sequence) == 8:
            break

    return sequence, display_frame

def reset_scanner():
    st.session_state.current_end = 'A'
    st.session_state.end_a = None
    st.session_state.end_b = None
    st.session_state.captured_image = None

# =================================================================
# STREAMLIT UI
# =================================================================
st.set_page_config(page_title="Cable Smart Wiring Scanner", layout="wide")
st.title("🔌 Cable Smart Wiring Scanner")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("END A")
    st.write(st.session_state.end_a if st.session_state.end_a else "Not scanned")
with col2:
    st.subheader("END B")
    st.write(st.session_state.end_b if st.session_state.end_b else "Not scanned")
with col3:
    st.subheader("CABLE RESULT")
    if st.session_state.end_a and st.session_state.end_b:
        if st.session_state.end_a == st.session_state.end_b:
            st.success("STRAIGHT-THROUGH (VALID)")
        else:
            st.warning("CROSSOVER CABLE")
    else:
        st.write("---")

st.button("↻ RESET SCANNER", on_click=reset_scanner)
st.markdown("---")

if st.session_state.end_a and st.session_state.end_b:
    st.info("Cable analysis completed.")
    st.stop()

st.subheader(f"Scanning: END {st.session_state.current_end}")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("#### Live Scanner (Touch 'n Go Style)")
    
    # HTML5 Component with Camera Stream, Viewfinder Box, and Real-time Zoom Slider
    camera_html = """
    <div style="text-align: center; background: #111; padding: 10px; border-radius: 10px; width: 380px;">
        <div style="position: relative; width: 360px; height: 270px; overflow: hidden; margin: auto; border: 2px solid #333;">
            <video id="webcam" autoplay playsinline style="width: 100%; height: 100%; object-fit: cover; transform-origin: center;"></video>
            <!-- Target Scanner Box -->
            <div style="position: absolute; top: 15%; left: 10%; width: 80%; height: 70%; border: 2px dashed #00ff00; box-sizing: border-box; pointer-events: none;">
                <span style="color: #00ff00; font-size: 11px; background: rgba(0,0,0,0.6); padding: 2px 4px; position: absolute; top: 2px; left: 2px;">ALIGN CABLE HERE</span>
            </div>
        </div>
        <div style="margin-top: 10px; color: white;">
            <label for="zoomRange">🔍 Live Viewfinder Zoom: </label>
            <input type="range" id="zoomRange" min="1" max="4" step="0.1" value="1" style="width: 150px;" oninput="applyZoom(this.value)">
            <span id="zoomVal">1.0x</span>
        </div>
        <button onclick="captureFrame()" style="margin-top: 10px; width: 100%; padding: 8px; background: #0084ff; color: white; border: none; border-radius: 5px; font-weight: bold; cursor: pointer;">📸 SCAN NOW</button>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>

    <script>
        const video = document.getElementById('webcam');
        const zoomRange = document.getElementById('zoomRange');
        const zoomVal = document.getElementById('zoomVal');
        let track = null;

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } })
            .then(stream => {
                video.srcObject = stream;
                track = stream.getVideoTracks()[0];
            })
            .catch(err => console.error("Camera Error: ", err));

        function applyZoom(val) {
            zoomVal.innerText = val + 'x';
            // Hardware optical zoom fallback to CSS scale transform
            if (track && 'zoom' in track.getCapabilities()) {
                track.applyConstraints({ advanced: [{ zoom: parseFloat(val) }] });
            } else {
                video.style.transform = `scale(${val})`;
            }
        }

        function captureFrame() {
            const canvas = document.getElementById('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg');
            
            // Pass captured frame back to Streamlit URL query string
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: dataUrl}, '*');
        }
    </script>
    """
    
    # Render component
    st.components.v1.html(camera_html, height=380)

    # Alternative Image Upload
    uploaded_file = st.file_uploader("Or upload image directly", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.session_state.captured_image = Image.open(uploaded_file)

with right_col:
    st.markdown("#### Scan Analysis Results")
    if st.session_state.captured_image:
        frame_bgr = cv2.cvtColor(np.array(st.session_state.captured_image), cv2.COLOR_RGB2BGR)
        sequence, annotated_frame = detect_wire_sequence_contours(frame_bgr)
        
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Processed Viewfinder Capture", width=380)

        for i, color in enumerate(sequence):
            st.write(f"**Pin {i+1}:** {color}")

        standard = "T568A" if sequence == T568A else ("T568B" if sequence == T568B else None)
        if standard:
            st.success(f"Detected Standard: **{standard}**")
            if st.button(f"Save Result as END {st.session_state.current_end}"):
                if st.session_state.current_end == 'A':
                    st.session_state.end_a = standard
                    st.session_state.current_end = 'B'
                else:
                    st.session_state.end_b = standard
                st.session_state.captured_image = None
                st.rerun()
        else:
            st.error("Sequence does not match T568A or T568B standards.")
