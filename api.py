'''
Author: fuzhenghao
Date: 2024-05-15 16:15:08
LastEditTime: 2024-05-15 20:46:52
LastEditors: fuzhenghao
Description: 
FilePath: \classal\api.py
'''
from fastapi import FastAPI
from ultralytics import YOLO
import numpy as np
import Config
import detect_tools as tools
import cv2 

app = FastAPI()

model = YOLO('best.pt', task='detect')
model(np.zeros((48, 48, 3)))  # 预先加载推理模型


def get_resize_size(show_width, show_height, img):
    _img = img.copy()
    img_height, img_width, depth = _img.shape
    ratio = img_width / img_height
    if ratio >= show_width / show_height:
        img_width = show_width
        img_height = int(img_width / ratio)
    else:
        img_height = show_height
        img_width = int(img_height * ratio)
    return img_width, img_height

@app.get("/detect")
async def detect(path: str):
    org_path = path
    org_img = tools.img_cvread(org_path)
    results = model(org_img)[0]
    location_list = results.boxes.xyxy.tolist()
    r_locationr_list = [list(map(int, e)) for e in location_list]
    cls_list = results.boxes.cls.tolist()
    cls_list = [int(i) for i in cls_list]
    conf_list = results.boxes.conf.tolist()
    conf_list = ['%.2f %%' % (each * 100) for each in conf_list]
    total_nums = len(location_list)
    cls_percents = []
    for i in range(6):
        res = cls_list.count(i) / total_nums
        cls_percents.append(res)
    now_img = results.plot()
    # draw_img = now_img
    # 获取缩放后的图片尺寸
    img_width, img_height = get_resize_size(770, 480, now_img)
    # resize_cvimg = cv2.resize(now_img, (img_width, img_height))
    # pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
    # resize_cvimg = cv2.resize(now_img, (img_width, img_height))
    #pix_img = cv2.resize(now_img, (img_width, img_height))
    # 设置路径显示kill

    # 目标数目 return
    target_nums = len(cls_list)

    # 设置目标选择下拉框
    choose_list = ['全部']
    target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(cls_list)]
    # object_list = sorted(set(self.cls_list))
    # for each in object_list:
    #     choose_list.append(Config.CH_names[each])
    choose_list = choose_list + target_names
    return {'location_list': r_locationr_list, 'conf_list': conf_list, 'cls_list':cls_list, 'cls_percents': cls_percents, 'choose_list': choose_list, 'target_nums': target_nums}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7020)


