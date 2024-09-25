# import os

# ds_store_path = '/Users/kjh/kjh_Project/AI/MP3/.DS_Store'
# if os.path.exists(ds_store_path):
#     os.remove(ds_store_path)
#     print('.DS_Store 파일이 삭제되었습니다.')
# else:
#     print('.DS_Store 파일이 존재하지 않습니다.')

import os
from PIL import Image

def is_image_file(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # 이미지 파일인지 확인
        return True
    except Exception:
        return False

# 이미지 파일 체크
image_dir = '/content/drive/MyDrive/cat_dog'
for subdir, _, files in os.walk(image_dir):
    for file in files:
        if not is_image_file(os.path.join(subdir, file)):
            print(f"손상된 이미지 파일: {os.path.join(subdir, file)}")
