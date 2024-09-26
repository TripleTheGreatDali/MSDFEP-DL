import base64
import io
import json
import sqlite3

import PIL
import cv2

from PIL import Image
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os

from .Model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normal_abnormal_VGG_model = VGG([2, 2, 4, 4, 4], 2).to(device)
observational_surgical_VGG_model = VGG([2, 2, 4, 4, 4], 2).to(device)
normal_abnormal_VGG_model.load_state_dict(torch.load("./model_0726/VGG_0_or_not_0_best.pth"))
observational_surgical_VGG_model.load_state_dict(torch.load("./model_0726/VGG_1_or_2_best.pth"))

YOLO_path = "./model_0726/YOLO_2024_07_best.pt"
yolo_model = YOLO(YOLO_path)

transunet = Transunet()
transunet_path = "./model_0726/TransUnet_gray_best_2024_07.pth"
transunet.load_state_dict(torch.load(transunet_path))

# ========================================================VGG Function========================================================
# 导入图片
VGG_img_size = 224

def VGG_transTensor(img):
    transform = transforms.Compose([
        transforms.Resize((VGG_img_size, VGG_img_size)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.size()
    img_tensor = img_tensor.to(device)
    return img_tensor


# ========================================================YOLO Function========================================================
def new_coordinate(img, x_min, y_min, x_max, y_max):
    # 计算原始裁剪区域的宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    # 确定新的裁剪尺寸，取宽度和高度的较大值
    size = min(max(width, height) + 20,600)  # 添加额外的边界

    # 计算中心点
    center_x = x_min + width // 2
    center_y = y_min + height // 2

    # 计算新的裁剪坐标，保证不越界
    new_x_min = max(center_x - size // 2, 0)
    new_y_min = max(center_y - size // 2, 0)
    new_x_max = min(new_x_min + size, img.width)
    new_y_max = min(new_y_min + size, img.height)

    # 如果计算的新坐标导致越界，需要调整最小值以确保尺寸一致
    if new_x_max - new_x_min < size:
        new_x_min = max(new_x_max - size, 0)
        new_x_max = new_x_min + size
    if new_y_max - new_y_min < size:
        new_y_min = max(new_y_max - size, 0)
        new_y_max = new_y_min + size

    # 确保新坐标在图像范围内
    new_x_max = min(new_x_max, img.width)
    new_y_max = min(new_y_max, img.height)
    new_x_min = max(new_x_max - size, 0)
    new_y_min = max(new_y_max - size, 0)

    return new_x_min, new_y_min, new_x_max, new_y_max


def crop_square(img, x_min, y_min, x_max, y_max):
    # 根据YOLO的目标位置 裁剪图像
    new_x_min, new_y_min, new_x_max, new_y_max = new_coordinate(img, x_min, y_min, x_max, y_max)
    # 裁剪并返回图像
    return img.crop((new_x_min, new_y_min, new_x_max, new_y_max))


def YOLO_Detecting_diseased_areas(org_img, conf=0.4):
    org_img_np = np.array(org_img)
    # YOLO 通过灰度图识别病变位置
    res = yolo_model.predict(org_img.convert('L'), conf=conf)
    boxes = res[0].boxes
    frame = res[0].plot()[:, :, ::-1]
    conf = res[0].boxes.conf.cpu().numpy()
    detections = res[0].boxes.xyxy.cpu().numpy()
    for row, confidence in zip(detections, conf):
        x1, y1, x2, y2, score = int(row[0]), int(row[1]), int(row[2]), int(row[3]), confidence
        cv2.rectangle(org_img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(org_img_np, f'confidence {confidence:.4f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)
    return org_img_np, boxes


# ========================================================TransUnet Function========================================================
Unet_img_size = 128
transform = transforms.Compose([
    transforms.Resize((Unet_img_size, Unet_img_size)),
    transforms.ToTensor(),
])


def Transunet_Split_diseased_areas(boxes, org_img):
    segmented_image_list = []
    for box in boxes.xyxy.cpu().numpy():
        x_min, y_min, x_max, y_max = map(int, box)
        # print(x_min, y_min, x_max, y_max)
        # frame = res[0].plot()[:, :, ::-1]
        # new_x_min, new_y_min, new_x_max, new_y_max = new_coordinate(org_img, x_min, y_min, x_max, y_max)
        cropped_img = org_img.crop((x_min, y_min, x_max, y_max))  # 裁剪原图像
        mask_img = Transunet_Split_img(cropped_img)
        segmented_image_list.append([mask_img, x_min, y_min, x_max, y_max])

    return segmented_image_list


# 输入
# 根据YOLO标签分割后的区域
# return：
# 该区域对应的mask，
# type：np；
# 维度：（w，h，1）
def Transunet_Split_img(cropped_img):
    img_tensor = transform(cropped_img.convert('L'))
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.size()
    img_tensor = img_tensor.to('cuda')

    y_pred = transunet(img_tensor)
    output_image = torch.argmax(y_pred[0], axis=0).squeeze(0).cpu().detach().numpy()

    output_np = np.array(output_image)
    # # 对预测结果应用形态学操作
    # output_np = output_np.astype(np.uint8)  # 确保是 uint8 类型
    # # 定义腐蚀和膨胀的核
    # kernel = np.ones((3, 3), np.uint8)
    # # 腐蚀操作
    # erosion = output_np
    # for _ in range(15):
    #     erosion = cv2.erode(erosion, kernel, iterations=1)
    # # 膨胀操作
    # output_np = erosion
    # for _ in range(15):
    #     output_np = cv2.dilate(output_np, kernel, iterations=1)

    # print("output_np:",output_np.shape)
    # 将 numpy 数组转换为 PIL 图像
    output_pil = Image.fromarray(output_np.astype('uint8'))
    # 调整模型输出到原始图像的大小
    output_pil = output_pil.resize(cropped_img.size, Image.NEAREST)
    mask = np.array(output_pil).reshape((cropped_img.size[1], cropped_img.size[0], 1))

    return mask  # (shape,shape,1)


# 输入参数：
#     图像 cv RGB
#

# return：
# 如果没病:  acc(准确率), 0，原图像，全黑mask
# 如果观察:  acc(准确率), 1，带标注的图像，分割的部位
# 如果手术:  acc(准确率), 2，带标注的图像，分割的部位

def identify_system(org_img, conf):
    acc_abnormal = []  # 识别100次，看结果是否正常
    acc_surgical = []  # 识别100次，看结果是观察还是手术

    # 打开图片
    img_tensor = VGG_transTensor(org_img)
    print(img_tensor.size())
    # 识别100次 将结果记录在acc_abnormal中
    for i in range(100):
        outputs = normal_abnormal_VGG_model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        acc_abnormal.append(predicted.cpu().numpy()[0])
    if sum(acc_abnormal) > 60:
        # 识别100次 将结果记录在acc_surgical中
        for i in range(100):
            outputs = observational_surgical_VGG_model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            acc_surgical.append(predicted.cpu().numpy()[0])
        if sum(acc_surgical) > 50:
            acc = sum(acc_surgical)
            resize_img, boxes = YOLO_Detecting_diseased_areas(org_img, conf)
            segmented_image = Transunet_Split_diseased_areas(boxes, org_img)
            return acc, 2, resize_img, segmented_image
        else:
            acc = 100 - sum(acc_surgical)
            resize_img, boxes = YOLO_Detecting_diseased_areas(org_img, conf)
            segmented_image = Transunet_Split_diseased_areas(boxes, org_img)
            return acc, 1, resize_img, segmented_image
    else:
        segmented_image = Image.new('RGB', org_img.size)
        acc = 100 - sum(acc_abnormal)

        return acc, 0, np.array(org_img), segmented_image


def cope_masks(org_img, masks):
    org_img_np = np.array(org_img)
    # 初始化总掩码图像和结果图像
    combined_array = np.zeros((org_img.size[1], org_img.size[0], 1), dtype=np.uint8)
    # combined_array = None
    for image_array, new_x_min, new_y_min, new_x_max, new_y_max in masks:
        # 创建总掩码图像，初始为全黑
        mask = np.zeros((org_img.size[1], org_img.size[0], 1), dtype=np.uint8)
        mask[new_y_min:new_y_max, new_x_min:new_x_max] = image_array
        # if combined_array is None:
        #     combined_array = mask * org_img
        combined_array[mask != 0] = mask[mask != 0]

    # 确保结果在有效范围内
    combined_array = np.clip(combined_array, 0, 255)
    # 红色图层 病变区域
    red_layer = np.zeros_like(org_img_np, dtype=np.uint8)
    red_layer[:, :, 0] = 255  # 红色通道
    # 转为灰度图
    gray_img = cv2.cvtColor(org_img_np, cv2.COLOR_RGB2GRAY)
    gray_img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # 设置红色图层的透明度
    alpha = 0.3
    combined_img = cv2.addWeighted(gray_img_3ch, 1 - alpha, combined_array * red_layer, alpha, 0)
    return combined_img


def model_recognize(source_img, confidence):
    # print("开始识别")
    # 将模型转移到定义的设备上
    acc, class_num, img, masks = identify_system(source_img, confidence)

    list = ["Normal", "Observation Recommended", "Surgery Recommended"]
    result_text = f"There is a {acc}% chance that it is {list[class_num]}"
    print(result_text)

    # ====================标注识别的图片========================

    img_pil = Image.fromarray(img)
    # 将PIL图像保存到字节流中，而不是文件
    frame_buf = io.BytesIO()
    img_pil.save(frame_buf, format='JPEG')

    # 将图片编码为Base64字符串
    img_base64 = base64.b64encode(frame_buf.getvalue()).decode('utf-8')
    img_url = "data:image/jpeg;base64," + img_base64

    # ====================标注分割的图片========================
    if class_num != 0:
        result_img = cope_masks(source_img, masks)
        result_img = Image.fromarray(result_img)
    else:
        result_img = masks

    result_img_buf = io.BytesIO()
    result_img.save(result_img_buf, format='JPEG')

    # 将图片编码为Base64字符串
    result_img_base64 = base64.b64encode(result_img_buf.getvalue()).decode('utf-8')
    result_img_url = "data:image/jpeg;base64," + result_img_base64

    return result_text, img_url, result_img_url


if __name__ == "__main__":
    model_path = '*******'
    img_path = '01.jpg'
    confidence = 0.4
    model_recognize(img_path, model_path, confidence)