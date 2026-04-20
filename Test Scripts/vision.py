import cv2
import mss
import numpy as np
import time

# Use your exact monitor settings from the main script
MONITOR = {"top": 40, "left": 25, "width": 640, "height": 480}

def debug_speed_capture():
    sct = mss.mss()
    print("--- Speed ROI Debugger Active ---")
    print("Press 'q' to quit.")

    while True:
        # Grab screen
        sct_img = sct.grab(MONITOR)
        raw = np.array(sct_img)[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # 1. Your current Speed ROI logic
        # gray[y1:y2, x1:x2]
        speed_roi = gray[448:473, 540:620]
        
        # 2. Your current threshold logic
        _, thresh = cv2.threshold(speed_roi, 250, 255, cv2.THRESH_BINARY)
        
        # 3. Your current speed calculation
        speed = float(np.sum(thresh) // 1020)

        # --- Visualizations ---
        # Upscale for visibility (it's a tiny crop)
        display_roi = cv2.resize(speed_roi, (400, 150), interpolation=cv2.INTER_NEAREST)
        display_thresh = cv2.resize(thresh, (400, 150), interpolation=cv2.INTER_NEAREST)

        # Draw a red rectangle on the full view to show WHERE the crop is
        preview_full = raw.copy()
        cv2.rectangle(preview_full, (540, 448), (620, 473), (0, 0, 255), 2)

        # Show windows
        cv2.imshow("1. ROI (What AI sees)", display_roi)
        cv2.imshow("2. Threshold (High Contrast)", display_thresh)
        cv2.imshow("3. Full View (Red Box = Crop Location)", preview_full)

        print(f"Calculated Speed Value: {speed}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_speed_capture()