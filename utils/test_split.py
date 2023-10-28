
import os
import shutil

source_folder = "datasets\\data"
# 获取源文件夹中的所有文件
files = os.listdir(source_folder)
total = 0
for file in files:
    if file.startswith('normal'):
        print(0)
    elif file.startswith('potholes'):
        total += 1
        print(1)
print(total)