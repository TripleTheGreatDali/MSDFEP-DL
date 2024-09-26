import torch
from ultralytics import YOLO
import os

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = YOLO()
# model.to(device)
# # Use the model
# model.to(device)
# model.hyp = {'giou': 3.54, 'cls': 1.0, 'cls_pw': 1.0, 'obj': 64.3, 'obj_pw': 1.0,
#              'iou_t': 0.213, 'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005}
# model.train(data=os.path.abspath("./Final_Dataset/data.yaml"), epochs=100, batch=64)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(os.path.abspath("yolov8n.pt"))
    model.to(device)
    model.hyp = {'giou': 3.54, 'cls': 1.0, 'cls_pw': 1.0, 'obj': 64.3, 'obj_pw': 1.0,
                 'iou_t': 0.213, 'lr0': 0.001, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0001}
    # 运行训练
    model.train(data=os.path.abspath("Final_Dataset/data.yaml"), epochs=146, batch=16,device=device)

if __name__ == '__main__':
    main()