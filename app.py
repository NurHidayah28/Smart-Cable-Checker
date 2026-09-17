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
# ENHANCED OPENCV & COLOR DETECTION LOGIC
# =================================================================
def preprocess_image(roi):
    """Normalize lighting using CLAHE on the L channel in LAB color space."""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def classify_wire(wire_roi):
    if wire_roi.size == 0:
        return "Unknown"

    wire_roi = preprocess_image(wire_roi)
    hsv = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2HSV)
    
    masks = {
        "Orange": cv2.inRange(hsv, np.array([5, 80, 60]), np.array([25, 255, 255])),
        "Green": cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])),
        "Blue": cv2.inRange(hsv, np.array([90, 50, 40]), np.array([130, 255, 255])),
        "Brown": cv2.inRange(hsv, np.array([2, 40, 20]), np.array([20, 255, 180]))
    }

    scores = {}
    total_pixels = max(hsv.shape[0] * hsv.shape[1], 1)

    for name, mask in masks.items():
        scores[name] = cv2.countNonZero(mask) / total_pixels

    best_colour = max(scores, key=scores.get)
    best_score = scores[best_colour]

    if best_score < 0.20:
        return "Unknown"

    # Stripe Detection (white content threshold)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 70, 255]))
    white_score = cv2.countNonZero(white_mask) / total_pixels

    if white_score > 0.12:
        if best_colour == "Green": return "White/Green"
        elif best_colour == "Orange": return "White/Orange"
        elif best_colour == "Brown": return "White/Brown"
        elif best_colour == "Blue": return "White/Blue"

    return best_colour

def detect_wire_sequence_contours(frame):
    """Detects individual wire strands via contours and sorts them left-to-right."""
    height, width = frame.shape[:2]
    
    # Target upper section where wire tips spread out
    roi_y1, roi_y2 = int(height * 0.15), int(height * 0.60)
    roi_x1, roi_x2 = int(width * 0.10), int(width * 0.90)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi.size == 0:
        return [], frame

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Mask colorful wire components vs neutral background
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    wire_mask = cv2.bitwise_or(
        cv2.inRange(sat, 30, 255),
        cv2.inRange(val, 150, 255)
    )

    # Clean noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(wire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wire_blobs = []
    min_area = (roi.shape[0] * roi.shape[1]) * 0.001

    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out non-wire shapes
            if h > 15 and w < int(roi.shape[1] * 0.25):
                wire_blobs.append((x, y, w, h))

    # Sort wires from left to right based on X coordinate
    wire_blobs = sorted(wire_blobs, key=lambda b: b[0])

    display_frame = frame.copy()
    sequence = []

    for (x, y, w, h) in wire_blobs:
        abs_x = roi_x1 + x
        abs_y = roi_y1 + y
        wire_roi = frame[abs_y:abs_y+h, abs_x:abs_x+w]
        
        color = classify_wire(wire_roi)
        if color != "Unknown":
            sequence.append(color)
            # Draw bounding box over recognized wire
            cv2.rectangle(display_frame, (abs_x, abs_y), (abs_x + w, abs_y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, str(len(sequence)), (abs_x, abs_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(sequence) == 8:
            break

    return sequence, display_frame

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

# Results Header
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

if st.session_state.end_a and st.session_state.end_b:
    st.info("Cable analysis completed. Reset the scanner to test a new cable.")
    st.stop()

st.subheader(f"Scanning: END {st.session_state.current_end}")

input_method = st.radio("Select Input Method:", ["📷 Camera", "📁 Upload Image"], horizontal=True)

img_file_buffer = None
if input_method == "📷 Camera":
    img_file_buffer = st.camera_input("Take a picture of the wires")
else:
    img_file_buffer = st.file_uploader("Insert a picture to scan the cable", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process image with dynamic contour scanning
    sequence, annotated_frame = detect_wire_sequence_contours(frame_bgr)
    standard = identify_standard(sequence)

    # Split display into 2 compact columns to limit picture size
    img_col, info_col = st.columns([1, 1])

    with img_col:
        # Fixed width limits image scale on screen
        st.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
            caption="Detected Wires (Left to Right)", 
            width=400
        )

    with info_col:
        st.markdown("### Detection Live Results")
        if sequence:
            for i, color in enumerate(sequence):
                st.write(f"**Pin {i+1}:** {color}")
        else:
            st.warning("No clear wire strands detected.")

        if len(sequence) < 8:
            st.warning(f"Detected {len(sequence)}/8 wires. Ensure wires are well-lit and separated.")
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
            st.error("Wire sequence detected, but order does not match T568A or T568B standard.")
