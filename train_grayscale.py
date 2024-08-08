from ultralytics import YOLO

model = YOLO('./ultralytics/cfg/models/v10/yolov10n.yaml')
model.train(data="/home/allanbatista/Workspaces/newtail/newtail-nia-smart-ads/notebooks/training/Dataset/widerface.yaml",
            epochs=1, batch=8)