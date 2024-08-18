import os
import cv2
from draw_utils import draw_keypoints
import matplotlib.pyplot as plt
import pandas as pd

def read_excel():
    file_name = "data.xlsx"
    # 读取Excel文件
    df = pd.read_excel(file_name)
    # 将列分为三组
    name = df.iloc[:, :1]  # 第一组：第1列
    point = df.iloc[:, 1:9]  # 第二组：第2到第9列
    score = df.iloc[:, -1]  # 第三组：第10列
    return name, point, score

def predict_single_person(info):
    img_path = "0"
    for i, each in enumerate(info[0]):
        img_path = os.path.join(img_path, each)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plot_img = draw_keypoints(img, info[1], info[2], thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("000.jpg")


if __name__ == '__main__':
    chart = read_excel()
    predict_single_person()
