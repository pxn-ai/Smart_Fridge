import cv2
import pytesseract
import re
import platform
import os
from datetime import datetime

# --- PORTABILITY CONFIGURATION ---
if platform.system() == "Windows":
    # Using your specific path: D:\Programmes\tesseract.exe
    windows_tesseract_path = r'D:\Programmes\tesseract.exe'
    if os.path.exists(windows_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = windows_tesseract_path
    else:
        print(f"Warning: Tesseract not found at {windows_tesseract_path}")

def extract_date_flexible(text):
    """
    ENTC Improvement: Loosened Regex to find dates even if 
    separators are missing or weird (dots, spaces, slashes).
    """
    # Looks for patterns like 12/05/2026, 12.05.2026, or 12 05 2026
    pattern = r'(\d{1,2})[\s\/\-\.]+(\d{1,2})[\s\/\-\.]+(\d{2,4})'
    match = re.search(pattern, text)
    if match:
        day, month, year = match.groups()
        # Fix short years (e.g., '26' to '2026')
        if len(year) == 2:
            year = "20" + year
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    return None

cap = cv2.VideoCapture(0)

print("--- Expiry Detector Ready ---")
print("Press 'S' to scan | Press 'Q' to exit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # UI Overlay
    h, w, _ = frame.shape
    cv2.rectangle(frame, (w//4, h//3), (3*w//4, 2*h//3), (0, 255, 0), 2)
    cv2.imshow("Scanner - Press S to Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # --- Pre-processing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # We use a slight blur to remove 'noise' before OCR
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # --- OCR Extraction ---
        # config='--psm 11' tells Tesseract to find text anywhere in the image
        raw_text = pytesseract.image_to_string(processed, config='--psm 11')
        
        print("\n" + "="*30)
        print("RAW DATA SEEN BY COMPUTER:")
        print(f"'{raw_text.strip()}'") 
        print("="*30)

        date_str = extract_date_flexible(raw_text)
        
        # --- Calculation ---
        if date_str:
            try:
                exp_date = datetime.strptime(date_str, "%d/%m/%Y")
                days_left = (exp_date - datetime.now()).days
                
                print(f"RESULT: Found Expiry Date -> {date_str}")
                if days_left > 0:
                    print(f"STATUS: Valid. {days_left} days remaining.")
                else:
                    print(f"STATUS: EXPIRED by {abs(days_left)} days!")
            except Exception as e:
                print(f"Found {date_str} but couldn't calculate. Check format.")
        else:
            print("RETRY: No date-like numbers found. Try better light or focus.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()