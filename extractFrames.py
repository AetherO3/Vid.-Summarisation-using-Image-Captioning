import cv2
import os
from scenedetect import open_video, SceneManager, ContentDetector

def extract_frames_hybrid(vidPath, outputPath, fallback_every_n=30):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    cap = cv2.VideoCapture(vidPath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    used_scene_detection = False

    if total_frames > 300:
        try:
            video = open_video(vidPath)
            sceneManager = SceneManager()
            sceneManager.add_detector(ContentDetector(threshold=15))
            sceneManager.detect_scenes(video)
            scenes = sceneManager.get_scene_list()

            if len(scenes) > 0:
                for start_time, end_time in scenes:
                    mid_frame = int((start_time.get_frames() + end_time.get_frames()) // 2)
                    if mid_frame < 0 or mid_frame >= total_frames:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                    success, frame = cap.read()
                    if success:
                        cv2.imwrite(os.path.join(outputPath, f"frame_{mid_frame}.jpg"), frame)
                used_scene_detection = True
        except:
            pass

    if not used_scene_detection:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_count in range(total_frames):
            success, frame = cap.read()
            if not success:
                break
            if frame_count % fallback_every_n == 0:
                cv2.imwrite(os.path.join(outputPath, f"frame_{frame_count}.jpg"), frame)

    cap.release()

vid = 'example.mp4' # --- vid name ---
newDir = 'Frames'
extract_frames_hybrid(vid, newDir)