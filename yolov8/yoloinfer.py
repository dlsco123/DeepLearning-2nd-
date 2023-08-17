from ultralytics import YOLO

model = YOLO('C:/Users/user/Desktop/2nd/DeepLearning/yolov8/runs/detect/train4/weights/best.pt')
results = model.predict(source='C:/Users/user/Desktop/2nd/DeepLearning/yolov8/fish.jpg', show=False, save=True)

for result in results:
    boxes = result.boxes
    print(boxes)

print(boxes.xywh)
print(boxes.cls)