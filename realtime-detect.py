import cv2
import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

model = torch.hub.load('.', 'custom', path='weights/best.pt', source='local')
model.conf = 0.1
model.iou = 0.45
model.multi_label = True
model.agnostic = True

video_mode = True

# result = model('test.jfif')
# result.show()

if video_mode:
    cap = cv2.VideoCapture('data/videos/Honda-motor.mp4')
    # Check i\f camera opened successfully
    if cap.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    count_frame = 0
    color = (0, 255, 0)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        dim = frame.shape
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
        if count_frame%1 == 0:
            if ret == True:
                # Display the resulting frameq
                result = model(frame, size=640)
                # print(result.pandas().xyxy[0])
                for id,box in enumerate(result.xyxy[0]):
                    # if box[5] == 0:
                    xB = int(box[2])
                    xA = int(box[0])
                    yB = int(box[3])
                    yA = int(box[1])
                    vehicle_class = result.pandas().xyxy[0].iloc[[id]]['name'].item()
                    print(vehicle_class)
                    if vehicle_class == 'xe-khach':
                        print(vehicle_class)
                        color = (0, 255, 0)
                    elif vehicle_class == 'xe-con':
                        color = (255, 0, 0)
                    elif vehicle_class == 'xe-tai':
                        color = (0, 0, 255)
                    cv2.rectangle(frame, (xA, yA), (xB, yB), color, 2)
                    color = (0, 255, 0)
            frame = cv2.resize(frame, (dim[1], dim[0]), interpolation = cv2.INTER_AREA)
        cv2.imshow('video', frame)
        count_frame += 1
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # Break the loop
        # else:
        #     break
    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()
