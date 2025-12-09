# # import cv2
# # import os

# # def extractFrames(vidPath, outputPath):
# #     if not os.path.exists(outputPath):
# #         os.makedirs(outputPath)
# #         print(f'Created the directory{outputPath}')

# #     video = cv2.VideoCapture(vidPath)

# #     if not video.isOpened():
# #         print("Could not open video!!")
# #         return
    
# #     frameCount = 0
# #     success = True

# #     while(success):
# #         success, image = video.read()
# #         if not success:
# #             break

# #         frameCount += 1
# #         if frameCount % 10 == 0:
# #             frameFile = os.path.join(outputPath, f"frame{frameCount}.jpg")
# #             cv2.imwrite(frameFile, image)

# #         if frameCount % 100 == 0:
# #             print(f"processed {frameCount} frames")
    
# #     video.release()

# # videofile = 'example.mp4'
# # outputDir = 'Video Frames'

# # extractFrames(videofile, outputDir)


# from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector
# from scenedetect.scene_manager import save_images
# import os

# def extractframes(vidPath, outputPath):
#     video = open_video(vidPath)
#     sceneManager = SceneManager()

#     sceneManager.add_detector(ContentDetector(threshold=10))

#     sceneManager.detect_scenes(video)
#     scenes = sceneManager.get_scene_list()

#     if not os.path.exists(outputPath):
#         os.makedirs(outputPath)

#     print(f"Scenes detected: {len(scenes)}")

#     if len(scenes) > 0:
#         save_images(
#             scene_list=scenes,
#             video=video,
#             num_images=1,
#             output_dir=outputPath,
#             image_extension="jpg"
#         )
#     else:
#         video.seek(0)
#         frame = video.read()
#         frame.save(os.path.join(outputPath, "frame_0.jpg"))

# vid = 'example.mp4'
# newDir = 'Frames'
# extractframes(vid, newDir)



import cv2
import os
from scenedetect import open_video, SceneManager, AdaptiveDetector

def extract_frames_hybrid(vidPath, outputPath, fallback_every_n=30):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    cap = cv2.VideoCapture(vidPath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use scene detection for long or complex videos
    if total_frames > 300:  # threshold, ~10 sec at 30fps
        try:
            video = open_video(vidPath)
            sceneManager = SceneManager()
            sceneManager.add_detector(AdaptiveDetector())
            sceneManager.detect_scenes(video)
            scenes = sceneManager.get_scene_list()
            print(f"Scenes detected: {len(scenes)}")
            
            if len(scenes) > 0:
                # Extract 1 frame per scene
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
    
    # Fallback for simple videos or if no scenes detected
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
