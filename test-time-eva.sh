DATASET="pacs"     # cifar10_c cifar100_c imagenet_c domainnet126 officehome imagenet_vit imagenet_convnet imagenet_efn coloredMNIST waterbirds
METHOD="deyo"
echo DATASET: $DATASET
echo METHOD: $METHOD
GPU_id=0
CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-deyo.py --cfg cfg/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}" 