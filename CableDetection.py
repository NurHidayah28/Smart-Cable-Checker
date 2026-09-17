import cv2
import numpy as np


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
# CLASSIFY ONE WIRE (unchanged detection logic from the original)
# =================================================================

def classify_wire(wire_roi):

    if wire_roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(wire_roi, cv2.COLOR_BGR2HSV)

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
# DETECT 8 WIRE POSITIONS
# =================================================================

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


# =================================================================
# GUIDE BOX COORDINATES FOR A GIVEN FRAME SIZE
# =================================================================

def guide_box(width, height):

    box_width = int(width * 0.70)
    box_height = int(height * 0.65)

    x1 = int((width - box_width) / 2)
    y1 = int((height - box_height) / 2)
    x2 = x1 + box_width
    y2 = y1 + box_height

    return x1, y1, x2, y2


# =================================================================
# DIGITAL ZOOM
# =================================================================
#
# Browsers/webcams rarely expose true optical zoom through
# Streamlit, so this crops the centre of the frame and scales it
# back up — the same trick a phone's "1x/2x" digital zoom uses.
# =================================================================

def apply_digital_zoom(frame, zoom_factor):

    if zoom_factor <= 1.0:
        return frame

    height, width = frame.shape[:2]

    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2

    cropped = frame[y1:y1 + crop_height, x1:x1 + crop_width]

    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


# =================================================================
# FOCUS MODE (SHARPENING)
# =================================================================
#
# Real hardware autofocus isn't controllable from the browser for
# most webcams, so "Focus Mode" applies an unsharp-mask filter —
# this sharpens edges between wires, which measurably helps the
# colour/uniformity checks above on slightly soft or low-quality
# camera frames.
# =================================================================

def apply_focus_enhancement(frame):

    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    return sharpened


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
