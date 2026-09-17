import streamlit as st
import cv2
import numpy as np
from PIL import Image


# =================================================================
# PAGE CONFIG
# =================================================================

st.set_page_config(
    page_title="Cable Smart Wiring Assistant",
    page_icon="🔌",
    layout="wide"
)


# =================================================================
# RETRO TERMINAL THEME (kept from the original Tkinter version)
# =================================================================

BG = "#0a0a0a"
BORDER = "#e0a94f"
ACCENT_CYAN = "#5fd0ff"
TEXT = "#e8e8e8"
DIM_TEXT = "#8a8a8a"
GREEN = "#3ddc84"
ORANGE = "#f0883e"
RED = "#ff5f56"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
        font-family: 'Courier New', monospace;
    }}
    .panel {{
        border: 1px dashed {BORDER};
        border-radius: 4px;
        padding: 14px;
        background-color: #000000;
        margin-bottom: 12px;
    }}
    .panel-title {{
        color: {ACCENT_CYAN};
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =================================================================
# CABLE WIRING STANDARDS
# =================================================================

T568A = [
    "White/Green",
    "Green",
    "White/Orange",
    "Blue",
    "White/Blue",
    "Orange",
    "White/Brown",
    "Brown"
]

T568B = [
    "White/Orange",
    "Orange",
    "White/Green",
    "Blue",
    "White/Blue",
    "Green",
    "White/Brown",
    "Brown"
]


# =================================================================
# SESSION STATE (replaces the Tkinter instance variables)
# =================================================================

if "current_end" not in st.session_state:
    st.session_state.current_end = "A"

if "end_a_sequence" not in st.session_state:
    st.session_state.end_a_sequence = None

if "end_b_sequence" not in st.session_state:
    st.session_state.end_b_sequence = None

if "end_a_standard" not in st.session_state:
    st.session_state.end_a_standard = None

if "end_b_standard" not in st.session_state:
    st.session_state.end_b_standard = None


# =================================================================
# CLASSIFY ONE WIRE (unchanged detection logic from the original)
# =================================================================

def classify_wire(wire_roi):

    if wire_roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2HSV)

    # -----------------------------------------------------
    # COLOUR MASKS
    # -----------------------------------------------------

    masks = {}

    masks["Orange"] = cv2.inRange(
        hsv, np.array([5, 80, 60]), np.array([25, 255, 255])
    )

    masks["Green"] = cv2.inRange(
        hsv, np.array([35, 50, 40]), np.array([90, 255, 255])
    )

    masks["Blue"] = cv2.inRange(
        hsv, np.array([90, 50, 40]), np.array([135, 255, 255])
    )

    masks["Brown"] = cv2.inRange(
        hsv, np.array([5, 40, 20]), np.array([30, 255, 180])
    )

    # -----------------------------------------------------
    # FIND BASE COLOUR
    # -----------------------------------------------------

    scores = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]

    for name, mask in masks.items():
        scores[name] = cv2.countNonZero(mask) / total_pixels

    best_colour = max(scores, key=scores.get)
    best_score = scores[best_colour]

    # A real wire fills almost the whole sampled patch with one
    # colour. Background clutter usually only partially matches
    # a colour mask, so require strong coverage before trusting it.

    if best_score < 0.35:
        return "Unknown"

    # -----------------------------------------------------
    # UNIFORMITY CHECK
    # -----------------------------------------------------
    # A real wire is one flat, evenly-lit colour, so its hue
    # barely varies across the sample. Backgrounds are textured
    # and mix several hues even if part of them matched a mask.

    hue_channel = hsv[:, :, 0]
    sat_channel = hsv[:, :, 1]

    coloured_pixels = hue_channel[sat_channel > 40]

    if coloured_pixels.size < (total_pixels * 0.35):
        return "Unknown"

    if float(np.std(coloured_pixels)) > 18:
        return "Unknown"

    # -----------------------------------------------------
    # WHITE STRIPE DETECTION
    # -----------------------------------------------------

    white_mask = cv2.inRange(
        hsv, np.array([0, 0, 150]), np.array([180, 80, 255])
    )

    white_score = cv2.countNonZero(white_mask) / total_pixels

    if white_score > 0.10:

        if best_colour == "Green":
            return "White/Green"
        elif best_colour == "Orange":
            return "White/Orange"
        elif best_colour == "Brown":
            return "White/Brown"
        elif best_colour == "Blue":
            return "White/Blue"

    return best_colour


# =================================================================
# DETECT 8 WIRE POSITIONS (unchanged logic, applied to a still photo)
# =================================================================

def detect_wire_sequence(frame, x1, y1, x2, y2):

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return []

    height, width = roi.shape[:2]

    # Divide the guide box into 8 vertical sections — lay the 8
    # bare wires flat and side by side (no RJ45 head) left to right.

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


# =================================================================
# GUIDE BOX + ANALYSIS ON A CAPTURED PHOTO
# =================================================================

def analyse_photo(pil_image):

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]

    box_width = int(width * 0.70)
    box_height = int(height * 0.65)

    x1 = int((width - box_width) / 2)
    y1 = int((height - box_height) / 2)
    x2 = x1 + box_width
    y2 = y1 + box_height

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    sequence = detect_wire_sequence(frame, x1, y1, x2, y2)

    return sequence, display_frame


# =================================================================
# IDENTIFY T568A / T568B
# =================================================================

def identify_standard(sequence):

    if len(sequence) != 8:
        return None

    if sequence == T568A:
        return "T568A"

    if sequence == T568B:
        return "T568B"

    return None


# =================================================================
# RESET
# =================================================================

def reset_scan():
    st.session_state.current_end = "A"
    st.session_state.end_a_sequence = None
    st.session_state.end_b_sequence = None
    st.session_state.end_a_standard = None
    st.session_state.end_b_standard = None


# =================================================================
# HEADER
# =================================================================

st.markdown(
    f"<h2 style='color:{ACCENT_CYAN};'>🔌 CABLE SMART WIRING ASSISTANT</h2>"
    f"<p style='color:{TEXT};'>Automatic Network Cable Detection System</p>",
    unsafe_allow_html=True
)

st.divider()

col_camera, col_scanner = st.columns([2, 1])

# =================================================================
# CAMERA + SCAN COLUMN
# =================================================================

with col_camera:

    st.markdown(
        f"<div class='panel-title'>SCANNING: END {st.session_state.current_end}</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Lay the 8 cable wires flat and side by side inside the "
        "green guide box. Do not crimp an RJ45 head on — bare, "
        "untwisted wires only. Keep the wires straight and use "
        "good lighting for best detection."
    )

    photo = st.camera_input(
        f"Capture End {st.session_state.current_end}",
        key=f"camera_{st.session_state.current_end}"
    )

    sequence = None

    if photo is not None:

        pil_image = Image.open(photo)
        sequence, display_frame = analyse_photo(pil_image)

        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        st.image(display_rgb, caption="Guide box used for detection", use_container_width=True)

        # Live sequence preview

        seq_text = "\n".join(
            f"Pin {i + 1}: {colour}" for i, colour in enumerate(sequence)
        )
        st.code(seq_text if sequence else "---", language=None)

        live_standard = identify_standard(sequence)

        if live_standard:
            st.success(f"Detected Standard: {live_standard}")
        elif "Unknown" in sequence:
            st.warning(
                "Some wires could not be identified. Improve lighting "
                "and make sure all 8 bare wires are laid flat inside "
                "the green box."
            )
        else:
            st.error("The detected sequence does not match T568A or T568B.")

    confirm_label = f"🔍 Confirm Scan — End {st.session_state.current_end}"

    if st.button(confirm_label, disabled=(photo is None)):

        if sequence is None or len(sequence) != 8:
            st.warning("The system could not detect 8 wires.")

        elif "Unknown" in sequence:
            st.warning("Some wires could not be identified. Try again.")

        else:
            standard = identify_standard(sequence)

            if standard is None:
                st.error("The detected sequence does not match T568A or T568B.")

            elif st.session_state.current_end == "A":
                st.session_state.end_a_sequence = sequence
                st.session_state.end_a_standard = standard
                st.session_state.current_end = "B"
                st.rerun()

            else:
                st.session_state.end_b_sequence = sequence
                st.session_state.end_b_standard = standard
                st.rerun()

    if st.button("↻ RESET"):
        reset_scan()
        st.rerun()


# =================================================================
# RESULTS COLUMN
# =================================================================

with col_scanner:

    st.markdown("<div class='panel-title'>RESULTS</div>", unsafe_allow_html=True)

    # END A

    if st.session_state.end_a_sequence:
        st.markdown(
            f"<div class='panel'><b>END A</b><br>{st.session_state.end_a_standard}<br>"
            f"✓ Scanned</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='panel'><b>END A</b><br>Not scanned</div>",
            unsafe_allow_html=True
        )

    # END B

    if st.session_state.end_b_sequence:
        st.markdown(
            f"<div class='panel'><b>END B</b><br>{st.session_state.end_b_standard}<br>"
            f"✓ Scanned</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='panel'><b>END B</b><br>Not scanned</div>",
            unsafe_allow_html=True
        )

    # FINAL RESULT

    if st.session_state.end_a_sequence and st.session_state.end_b_sequence:

        if st.session_state.end_a_standard == st.session_state.end_b_standard:

            st.markdown(
                f"<div class='panel' style='border-color:{GREEN};color:{GREEN};'>"
                f"<b>CABLE RESULT</b><br>STRAIGHT-THROUGH<br>✓ VALID</div>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"<div class='panel' style='border-color:{ORANGE};color:{ORANGE};'>"
                f"<b>CABLE RESULT</b><br>CROSSOVER<br>⚠ CHECK WIRING</div>",
                unsafe_allow_html=True
            )

        st.write(f"End A: {st.session_state.end_a_standard}")
        st.write(f"End B: {st.session_state.end_b_standard}")

    else:
        st.markdown(
            "<div class='panel'><b>CABLE RESULT</b><br>---</div>",
            unsafe_allow_html=True
        )