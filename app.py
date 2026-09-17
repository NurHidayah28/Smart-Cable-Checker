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
# IMPROVED OPENCV LOGIC
# =================================================================
def classify_wire(wire_roi):
    if wire_roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2HSV)
    
    # Adjusted HSV ranges for realistic lighting/shadows
    masks = {
        "Orange": cv2.inRange(hsv, np.array([5, 100, 80]), np.array([25, 255, 255])),
        "Green": cv2.inRange(hsv, np.array([35, 60, 40]), np.array([85, 255, 255])),
        "Blue": cv2.inRange(hsv, np.array([90, 70, 40]), np.array([130, 255, 255])),
        "Brown": cv2.inRange(hsv, np.array([5, 40, 20]), np.array([20, 255, 180]))
    }

    scores = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]

    for name, mask in masks.items():
        scores[name] = cv2.countNonZero(mask) / max(total_pixels, 1)

    best_colour = max(scores, key=scores.get)
    best_score = scores[best_colour]

    # Check for white stripe (low saturation + high brightness)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 60, 255]))
    white_score = cv2.countNonZero(white_mask) / max(total_pixels, 1)

    if white_score > 0.15:
        if best_colour in ["Green", "Orange", "Brown", "Blue"]:
            return f"White/{best_colour}"

    if best_score > 0.20:
        return best_colour

    return "Unknown"

def detect_wire_sequence_contours(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return [], []

    # Noise reduction
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Detect non-background wire regions (saturated colors + white)
    color_mask = cv2.inRange(hsv, np.array([0, 30, 40]), np.array([180, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))
    wire_mask = cv2.bitwise_or(color_mask, white_mask)

    # Find distinct wire contours
    contours, _ = cv2.findContours(wire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wire_blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:  # Filter out tiny noise
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Ensure contour resembles a vertical wire segment
            if bh > 15:
                wire_sample = roi[by:by+bh, bx:bx+bw]
                color = classify_wire(wire_sample)
                wire_blobs.append((bx, color, (bx, by, bw, bh)))

    # Sort wires from Left to Right based on X position
    wire_blobs.sort(key=lambda b: b[0])

    sequence = [blob[1] for blob in wire_blobs[:8]]
    boxes = [(x1 + b[2][0], y1 + b[2][1], b[2][2], b[2][3]) for b in wire_blobs[:8]]

    return sequence, boxes

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

if st.session_state.end_a and st.session_state.end_b:
    st.info("Cable analysis completed. Reset the scanner to test a new cable.")
    st.stop()

# Scanning Area
st.subheader(f"Scanning: END {st.session_state.current_end}")
st.write("Lay all 8 bare wires flat and straight inside the green detection box.")

input_method = st.radio("Select Input Method:", ["📷 Camera", "📁 Upload Image"], horizontal=True)

img_file_buffer = None
if input_method == "📷 Camera":
    img_file_buffer = st.camera_input("Take a picture of the wires")
else:
    img_file_buffer = st.file_uploader("Insert a picture to scan the cable", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = frame_bgr.shape[:2]

    # Focused bounding area for wire detection
    box_width = int(width * 0.70)
    box_height = int(height * 0.50)
    x1 = int((width - box_width) / 2)
    y1 = int(height * 0.15)
    x2 = x1 + box_width
    y2 = y1 + box_height

    display_frame = frame_bgr.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display_frame, "DETECTION AREA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Perform contour wire detection
    sequence, boxes = detect_wire_sequence_contours(frame_bgr, x1, y1, x2, y2)

    # Draw detected wire contours on image
    for bx, by, bw, bh in boxes:
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    # Rescale image preview display using Streamlit columns
    left_pad, center_col, right_pad = st.columns([1, 2, 1])
    with center_col:
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Wire Detection Preview", width=450)

    standard = identify_standard(sequence)

    st.markdown("### Live Wire Detection")
    if sequence:
        seq_display = "\n".join([f"- **Pin {i+1}:** {color}" for i, color in enumerate(sequence)])
        st.markdown(seq_display)
    else:
        st.warning("No wires detected in ROI. Reposition the cable closer to the green box.")

    if len(sequence) < 8 or "Unknown" in sequence:
        st.error(f"Detected {len(sequence)}/8 clear wires. Improve lighting, straighten wires, and align them inside the green box.")
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
        st.error("The detected sequence does not match T568A or T568B standards.")
