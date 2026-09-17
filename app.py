import streamlit as st
import cv2
import numpy as np
from PIL import Image

# =================================================================
# CABLE WIRING STANDARDS
# =================================================================
T568A = [
    "White/Green", "Green", "White/Orange", "Blue", 
    "White/Blue", "Orange", "White/Brown", "Brown"
]

T568B = [
    "White/Orange", "Orange", "White/Green", "Blue", 
    "White/Blue", "Green", "White/Brown", "Brown"
]

# =================================================================
# SESSION STATE INITIALIZATION
# =================================================================
if 'current_end' not in st.session_state:
    st.session_state.current_end = 'A'
if 'end_a' not in st.session_state:
    st.session_state.end_a = None
if 'end_b' not in st.session_state:
    st.session_state.end_b = None

# =================================================================
# CORE OPENCV LOGIC
# =================================================================
def classify_wire(wire_roi):
    if wire_roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2HSV)
    masks = {
        "Orange": cv2.inRange(hsv, np.array([5, 80, 60]), np.array([25, 255, 255])),
        "Green": cv2.inRange(hsv, np.array([35, 50, 40]), np.array([90, 255, 255])),
        "Blue": cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255])),
        "Brown": cv2.inRange(hsv, np.array([5, 40, 20]), np.array([30, 255, 180]))
    }

    scores = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]

    for name, mask in masks.items():
        scores[name] = cv2.countNonZero(mask) / total_pixels

    best_colour = max(scores, key=scores.get)
    best_score = scores[best_colour]

    if best_score < 0.35:
        return "Unknown"

    hue_channel = hsv[:, :, 0]
    sat_channel = hsv[:, :, 1]
    coloured_pixels = hue_channel[sat_channel > 40]

    if coloured_pixels.size < (total_pixels * 0.35):
        return "Unknown"

    if float(np.std(coloured_pixels)) > 18:
        return "Unknown"

    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))
    white_score = cv2.countNonZero(white_mask) / total_pixels

    if white_score > 0.10:
        if best_colour == "Green": return "White/Green"
        elif best_colour == "Orange": return "White/Orange"
        elif best_colour == "Brown": return "White/Brown"
        elif best_colour == "Blue": return "White/Blue"

    return best_colour

def detect_wire_sequence(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    height, width = roi.shape[:2]
    wire_width = width / 8
    sequence = []

    for i in range(8):
        sx = int(i * wire_width)
        ex = int((i + 1) * wire_width)
        margin_x = int((ex - sx) * 0.20)
        
        sample_x1 = sx + margin_x
        sample_x2 = ex - margin_x
        sample_y1 = int(height * 0.20)
        sample_y2 = int(height * 0.85)

        wire_roi = roi[sample_y1:sample_y2, sample_x1:sample_x2]
        sequence.append(classify_wire(wire_roi))

    return sequence

def identify_standard(sequence):
    if len(sequence) != 8:
        return None
    if sequence == T568A:
        return "T568A"
    if sequence == T568B:
        return "T568B"
    return None

def reset_scanner():
    st.session_state.current_end = 'A'
    st.session_state.end_a = None
    st.session_state.end_b = None

# =================================================================
# STREAMLIT UI
# =================================================================
st.set_page_config(page_title="Cable Smart Wiring Scanner", layout="wide")

st.title("🔌 Cable Smart Wiring Scanner")
st.markdown("Automatic Network Cable Detection System")

# Results Dashboard
st.markdown("---")
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
            st.success("STRAIGHT-THROUGH\n\n✓ VALID")
        else:
            st.warning("CROSSOVER\n\n⚠ CHECK WIRING")
    else:
        st.write("---")

st.button("↻ RESET SCANNER", on_click=reset_scanner)
st.markdown("---")

# Stop if both ends are scanned
if st.session_state.end_a and st.session_state.end_b:
    st.info("Cable analysis completed. Reset the scanner to test a new cable.")
    st.stop()

# Scanning Area
st.subheader(f"Scanning: END {st.session_state.current_end}")
st.write("Lay the 8 bare cable wires flat and side-by-side. Keep them straight and use good lighting.")

# Feature: Choose between live camera or image upload
input_method = st.radio("Select Input Method:", ["📷 Camera", "📁 Upload Image"], horizontal=True)

img_file_buffer = None
if input_method == "📷 Camera":
    img_file_buffer = st.camera_input("Take a picture of the wires")
else:
    img_file_buffer = st.file_uploader("Insert a picture to scan the cable", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    # Process the image
    image = Image.open(img_file_buffer)
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = frame_bgr.shape[:2]

    # Calculate target box dimensions (70% width, 65% height)
    box_width = int(width * 0.70)
    box_height = int(height * 0.65)
    x1 = int((width - box_width) / 2)
    y1 = int((height - box_height) / 2)
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Draw guide box for visual feedback
    display_frame = frame_bgr.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(display_frame, "DETECTION AREA", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Make sure the bare wires fill the green detection area left-to-right.")

    # Detect the standard
    sequence = detect_wire_sequence(frame_bgr, x1, y1, x2, y2)
    standard = identify_standard(sequence)

    st.markdown("### Detection Live Results")
    if sequence:
        seq_display = "\n".join([f"- **Pin {i+1}:** {color}" for i, color in enumerate(sequence)])
        st.markdown(seq_display)
    
    if "Unknown" in sequence:
        st.error("Some wires could not be identified. Improve lighting and ensure all 8 wires are inside the green box.")
    elif standard:
        st.success(f"Detected Standard: **{standard}**")
        
        if st.button(f"Save as END {st.session_state.current_end}"):
            if st.session_state.current_end == 'A':
                st.session_state.end_a = standard
                st.session_state.current_end = 'B'
            else:
                st.session_state.end_b = standard
            st.rerun()
    else:
        st.error("The detected sequence does not match T568A or T568B.")
