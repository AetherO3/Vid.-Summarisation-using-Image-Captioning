import cv2
import os

def extractFrames(vidPath, outputPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
        print(f'Created the directory{outputPath}')

    video = cv2.VideoCapture(vidPath)

    if not video.isOpened():
        print("Could not open video!!")
        return
    
    frameCount = 0
    success = True

    while(success):
        success, image = video.read()
        if not success:
            break

        frameCount += 1
        if frameCount % 10 == 0:
            frameFile = os.path.join(outputPath, f"frame{frameCount}.jpg")
            cv2.imwrite(frameFile, image)

        if frameCount % 100 == 0:
            print(f"processed {frameCount} frames")
    
    video.release()

videofile = 'example.mp4'
outputDir = 'Video Frames'

extractFrames(videofile, outputDir)
