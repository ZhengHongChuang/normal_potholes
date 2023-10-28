
import os
import cv2  
import numpy as np

def calculate_channel_mean_and_variance(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    red_channel_sum = 0
    green_channel_sum = 0
    blue_channel_sum = 0
    red_channel_sum_squared = 0
    green_channel_sum_squared = 0
    blue_channel_sum_squared = 0
    num_images = len(image_files)

    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image = image / 255.0  # 将像素值标准化到 [0, 1]

        # 分别计算每个通道的和
        red_channel_sum += np.sum(image[:, :, 0])
        green_channel_sum += np.sum(image[:, :, 1])
        blue_channel_sum += np.sum(image[:, :, 2])

        red_channel_sum_squared += np.sum(np.square(image[:, :, 0]))
        green_channel_sum_squared += np.sum(np.square(image[:, :, 1]))
        blue_channel_sum_squared += np.sum(np.square(image[:, :, 2]))

    # 计算每个通道的均值
    red_mean = red_channel_sum / (num_images * image.shape[0] * image.shape[1])
    green_mean = green_channel_sum / (num_images * image.shape[0] * image.shape[1])
    blue_mean = blue_channel_sum / (num_images * image.shape[0] * image.shape[1])

    # 计算每个通道的方差
    red_variance = (red_channel_sum_squared / (num_images * image.shape[0] * image.shape[1])) - np.square(red_mean)
    green_variance = (green_channel_sum_squared / (num_images * image.shape[0] * image.shape[1])) - np.square(green_mean)
    blue_variance = (blue_channel_sum_squared / (num_images * image.shape[0] * image.shape[1])) - np.square(blue_mean)

    return red_mean, green_mean, blue_mean, red_variance, green_variance, blue_variance

if __name__ == "__main__":
    image_folder = "datasets\\data"
    red_mean, green_mean, blue_mean, red_variance, green_variance, blue_variance = calculate_channel_mean_and_variance(image_folder)
    print("红色通道均值：", red_mean)
    print("绿色通道均值：", green_mean)
    print("蓝色通道均值：", blue_mean)
    print("红色通道方差：", red_variance)
    print("绿色通道方差：", green_variance)
    print("蓝色通道方差：", blue_variance)
