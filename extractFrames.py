import cv2
import os
from scenedetect import open_video, SceneManager, AdaptiveDetector

def extract_frames_hybrid(vidPath, outputPath, fallback_every_n=30):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    cap = cv2.VideoCapture(vidPath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 300:  
        try:
            video = open_video(vidPath)
            sceneManager = SceneManager()
            sceneManager.add_detector(AdaptiveDetector())
            sceneManager.detect_scenes(video)
            scenes = sceneManager.get_scene_list()
            print(f"Scenes detected: {len(scenes)}")
            
            if len(scenes) > 0:
                for i, (start_time, end_time) in enumerate(scenes):
                    frame_number = int((start_time.get_frames() + end_time.get_frames()) // 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    success, frame = cap.read()
                    if success:
                        cv2.imwrite(os.path.join(outputPath, f"scene_{i}.jpg"), frame)
                cap.release()
                return
        except Exception as e:
            print(f"Scene detection failed: {e}")
    
    print("Using fallback frame extraction")
    for frame_count in range(total_frames):
        success, frame = cap.read()
        if not success:
            break
        if frame_count % fallback_every_n == 0:
            cv2.imwrite(os.path.join(outputPath, f"frame_{frame_count}.jpg"), frame)
    cap.release()

vid = 'example.mp4'
newDir = 'Frames'
extract_frames_hybrid(vid, newDir)
