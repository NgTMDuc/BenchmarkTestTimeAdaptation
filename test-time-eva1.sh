DATASET="imagenet_c"     # cifar10_c cifar100_c imagenet_c domainnet126 officehome imagenet_vit imagenet_convnet imagenet_efn coloredMNIST waterbirds
METHOD="proposal"
echo DATASET: $DATASET
echo METHOD: $METHOD

GPU_id=0

# METHOD="source"        # source norm_test memo eata cotta tent t3a norm_alpha lame adacontrast sar
CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg cfg/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}" --PROPOSAL_LAYER 4

# METHOD="tent"
# CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg cfg/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}"


# METHOD="sar"
# CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg cfg/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}"


# METHOD="eata"
# CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg cfg/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}"

# METHOD="deyo"
# CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg cfg/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}"