import os
import cv2
import numpy as np
import re

class FrameSaver:
    def __init__(self, save_dir="./frames", annotate=True):
        self.save_dir = save_dir
        self.annotate = annotate

        # Create main save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Create a new subdirectory like exp1, exp2, ...
        self.exp_dir = self._create_new_exp_dir()
        os.system(f"chmod -R 777 {self.exp_dir}")

    def _create_new_exp_dir(self):
        # List all existing exp* folders
        existing = [d for d in os.listdir(self.save_dir) if re.match(r'exp\d+', d)]
        numbers = [int(re.findall(r'\d+', d)[0]) for d in existing] if existing else []
        next_num = max(numbers) + 1 if numbers else 1
        new_exp_dir = os.path.join(self.save_dir, f"exp{next_num}")
        os.makedirs(new_exp_dir, exist_ok=True)
        return new_exp_dir

    def save_frame(self, frame, episode, frame_count, state=None, action=None):
        annotated_frame = frame.copy()

        if self.annotate and state is not None and action is not None:
            text = f"State: {[round(val, 2) for val in state]}, Action: {action}"
            annotated_frame = cv2.putText(
                annotated_frame,
                text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        frame_filename = f"episode_{episode + 1}_frame_{frame_count}.png"
        frame_path = os.path.join(self.exp_dir, frame_filename)
        cv2.imwrite(frame_path, annotated_frame)
