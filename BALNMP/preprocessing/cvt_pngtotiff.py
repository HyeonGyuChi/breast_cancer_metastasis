import os
from PIL import Image
from tqdm import tqdm

def main():
    src_dir = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs'
    dst_dir = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_tiff'

    file_list = os.listdir(src_dir)
    file_list_img = [file for file in file_list if file.endswith(".png")]

    os.makedirs(dst_dir, exist_ok=True)

    print('[+] cvt png to tiff ...')

    for f_name in tqdm(file_list):
        src_path = os.path.join(src_dir, f_name)
        dst_path = os.path.join(dst_dir, os.path.splitext(f_name)[0] + '.tiff')
        
        image = Image.open(src_path)
        image.save(dst_path)

    print('[-] cvt png to tiff ...')

if __name__ == "__main__":
    main()