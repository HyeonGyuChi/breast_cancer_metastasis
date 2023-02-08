experiment_type=$1 # fold
num_classes=2

log_dir="./logs_demo"
model_path="/home/hyeongyuc/code/breast_cancer_metastasis/BALNMP/code/logs_demo/1/checkpoint/best.pth"

python -u inference.py \
    --train_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/inference.json \
    --val_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/inference.json \
    --test_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/inference.json \
    --clinical_data_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/clinical_data_new/clinical_inference_data.xlsx \
    --data_dir_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/test_imgs_PyHIST \
    --num_classes ${num_classes} \
    --clinical_data_size 23 \
    --backbone vgg16_bn \
    --model_path ${model_path} \
    --log_dir_path ${log_dir} \
    --log_name ${experiment_type}