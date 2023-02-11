# 230208-a-1
log_d="logs_challenge_wandb"
experiment_type=1 # fold
num_classes=2

if [ ! -d ${log_d}/${experiment_type} ]; then
    mkdir -p ${log_d}/${experiment_type}
fi

python -u train.py \
    --train_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/train-fold-${experiment_type}.json \
    --val_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/valid-fold-${experiment_type}.json \
    --test_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/test-fold-${experiment_type}.json \
    --data_dir_path "/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_PyHIST" \
    --num_classes ${num_classes} \
    --backbone resnet50 \
    --log_dir_path ./${log_d} \
    --log_name ${experiment_type} | tee ${log_d}/${experiment_type}/output.log 2>&1


: <<'END'
# 230207-a-1
log_d="logs_challenge"
experiment_type=1 # fold
num_classes=2

python -u train.py \
    --train_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/train-fold-${experiment_type}.json \
    --val_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/valid-fold-${experiment_type}.json \
    --test_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/test-fold-${experiment_type}.json \
    --data_dir_path "/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_PyHIST" \
    --num_classes ${num_classes} \
    --backbone vgg16_bn \
    --log_dir_path ./${log_d} \
    --log_name ${experiment_type} | tee ${log_d}/${experiment_type}/output.log 2>&1

# 230206-a-1
log_d="logs_demo"
experiment_type=1 # fold
num_classes=2

python -u train.py \
    --train_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/train-fold-${experiment_type}.json \
    --val_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/valid-fold-${experiment_type}.json \
    --test_json_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json/test-fold-${experiment_type}.json \
    --clinical_data_path /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/clinical_data/clinical_train_data.xlsx \
    --data_dir_path "/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_PyHIST" \
    --num_classes ${num_classes} \
    --clinical_data_size 23 \
    --backbone vgg16_bn \
    --log_dir_path ./${log_d} \
    --log_name ${experiment_type} | tee ${log_d}/${experiment_type}/output.log 2>&1
END