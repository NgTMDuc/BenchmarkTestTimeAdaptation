### config
# export CUDA_LAUNCH_BLOCKING=1

DATASET="cifar10_c" # cifar10_c cifar100_c imagenet_c domainnet126 officehome imagenet_convnet waterbirds coloredmnist
METHOD="proposal"          # source norm_test memo eata cotta tent t3a norm_alpha lame adacontrast norm_alpha64
MODEL_CONTINUAL='Fully' # Continual Fully
GPUS=(2) #available gpus
NUM_GPUS=${#GPUS[@]}
NUM_MAX_JOB=$((NUM_GPUS))
# NUM_MAX_JOB=1
i=0
#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=$NUM_MAX_JOB #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & NOTE; Evaluation: Target domains  ######
  ###############################################################

  #Evaluate memo
  if [ "$METHOD" == "memo" ]; then
    if [ "$DATASET" == "cifar10_c" ] || [ "$DATASET" == "cifar100_c" ]; then
      lrs=(0.01 0.001 0.002 0.005 0.0005)
      bn_alphas=(0.1 0.2 0.5 0.9)
    elif [ "$DATASET" == "imagenet_c" ] || [ "$DATASET" == "domainnet126" ] || [ "$DATASET" == "officehome" ]; then
      lrs=(0.001 0.0001 0.0002 0.0005 0.00025 0.00005)
      bn_alphas=(0.1 0.2 0.5 0.9)
    fi
    for lr in ${lrs[*]}; do
      for bn_alpha in ${bn_alphas[*]}; do
        wait_n
        i=$((i + 1))
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
          --OPTIM_LR "$lr" --BN_ALPHA "$bn_alpha" &
      done
    done

  #Evaluate norm_alpha, norm_alpha64
  elif [ "$METHOD" == "norm_alpha" ] || [ "$METHOD" == "norm_alpha64" ]; then
    bn_alphas=(0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95)
    for bn_alpha in ${bn_alphas[*]}; do
      wait_n
      i=$((i + 1))
      CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
        --BN_ALPHA "$bn_alpha" &
    done

  #Evaluate tent
  elif [ "$METHOD" == "tent" ] || [ "$METHOD" == "tentE10" ]; then
    if [ "$DATASET" == "cifar10_c" ] || [ "$DATASET" == "cifar100_c" ]; then
      lrs=(0.0001 0.0002 0.00025 0.0005 0.001 0.002 0.005 0.01)
    elif [ "$DATASET" == "imagenet_c" ] || [ "$DATASET" == "domainnet126" ] || [ "$DATASET" == "officehome" ] || [ "$DATASET" == "imagenet_vit" ]; then
      lrs=(0.00005 0.0001 0.00025 0.0005 0.001 0.002 0.005 0.01)
    elif [ "$DATASET" == "imagenet_convnet" ]; then
      lrs=(0.000001 0.00002 0.00001 0.00005 0.0001 0.0002 0.0005 0.001)
    elif [ "$DATASET" == "imagenet_efn" ]; then
      lrs=(0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
    elif [ "$DATASET" == "coloredmnist" ]; then
      lrs=(0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5)
    fi
    for lr in ${lrs[*]}; do
      wait_n
      i=$((i + 1))
      if [ "$MODEL_CONTINUAL" == "Continual" ]; then
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}_continual" \
          --OPTIM_LR "$lr" --MODEL_CONTINUAL "$MODEL_CONTINUAL" &
      else
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
          --OPTIM_LR "$lr" --MODEL_CONTINUAL "$MODEL_CONTINUAL" &
      fi
    done

  #Evaluate cotta
  elif [ "$METHOD" == "cotta" ] || [ "$METHOD" == "cottaE10" ]; then
    if [ "$DATASET" == "cifar10_c" ] || [ "$DATASET" == "cifar100_c" ]; then
      lrs=(0.0001 0.0002 0.00025 0.0005 0.001 0.002 0.005 0.01)
      if [ "$METHOD" == "cottaE10" ]; then
        lrs=(0.0001 0.00025 0.0005 0.001)
      fi
      rsts=(0.005 0.01 0.02)
      if [ "$MODEL_CONTINUAL" == "Continual" ]; then
        lrs=(0.0005 0.001 0.005)
      fi
      if [ "$DATASET" == "cifar10_c" ]; then
        aps=(0.8 0.92 0.95)
      elif [ "$DATASET" == "cifar100_c" ]; then
        aps=(0.5 0.72 0.9)
      fi
    elif [ "$DATASET" == "imagenet_c" ] || [ "$DATASET" == "domainnet126" ] || [ "$DATASET" == "officehome" ] || [ "$DATASET" == "imagenet_vit" ] || [ "$DATASET" == "imagenet_convnet" ] || [ "$DATASET" == "imagenet_efn" ]; then
      lrs=(0.001 0.002 0.0025 0.005 0.01 0.02 0.05 0.1)
      if [ "$DATASET" == "imagenet_vit" ]; then
        lrs=(0.0001 0.0005 0.001 0.002 0.005 0.01 0.02 0.05)
      elif [ "$DATASET" == "imagenet_convnet" ]; then
        lrs=(0.00001 0.0001  0.001 0.005 0.001)
      elif [ "$DATASET" == "imagenet_efn" ]; then
#        lrs=(0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
        lrs=(0.2 0.5 1 2 5)
      fi
      if [ "$METHOD" == "cottaE10" ]; then
        lrs=(0.001 0.0025 0.005 0.01)
      fi
      rsts=(0.0005 0.001 0.002)
#      rsts=(0.002)
      aps=(0.05 0.1 0.2)
      if [ "$DATASET" == "officehome" ] || [ "$DATASET" == "domainnet126" ]; then
        rsts=(0.001 0.005 0.01 0.02)
        aps=(0.1 0.2 0.5)
      fi
      if [ "$MODEL_CONTINUAL" == "Continual" ]; then
        lrs=(0.005 0.01 0.05)
      fi
    fi
    for lr in ${lrs[*]}; do
      for rst in ${rsts[*]}; do
        for ap in ${aps[*]}; do
          i=$((i + 1))
          wait_n
          if [ "$MODEL_CONTINUAL" == "Continual" ]; then
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}_continual" \
              --OPTIM_LR "$lr" --COTTA_RST "$rst" --COTTA_AP "$ap" --MODEL_CONTINUAL "$MODEL_CONTINUAL" &
          else
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --COTTA_RST "$rst" --COTTA_AP "$ap" --MODEL_CONTINUAL "$MODEL_CONTINUAL" &
          fi
#          sleep 600
        done
      done
    done

  #Evaluate eata
  elif [ "$METHOD" == "eata" ] || [ "$METHOD" == "eataE10" ]; then
    dms=(0.05 0.1 0.2 0.4)
    fisher_alphas=(1 10 100 500 2000 )
    em_coes=(0.4)
    if [ "$DATASET" == "cifar10_c" ] || [ "$DATASET" == "cifar100_c" ]; then
      lrs=(0.00025 0.0005 0.001 0.002 0.005 0.01)
      if [ "$DATASET" == "cifar10_c" ]; then
        dms=(0.2 0.4 0.6 0.8)
      fi
    elif [ "$DATASET" == "imagenet_c" ] || [ "$DATASET" == "domainnet126" ] || [ "$DATASET" == "officehome" ] || [ "$DATASET" == "imagenet_vit" ]; then
      lrs=(0.0001 0.0002 0.00025 0.0005 0.001 0.002)
    elif [ "$DATASET" == "imagenet_convnet" ]; then
      lrs=(0.00001 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01)
    elif [ "$DATASET" == "imagenet_efn" ]; then
      lrs=(0.00005 0.0001 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2)
      em_coes=(0.1 0.2 0.4 0.6)
    elif [ "$DATASET" == "coloredmnist" ] || [ "$DATASET" == "waterbirds" ];  then
      lrs=(0.00025 0.0005 0.0025 0.005 0.01 0.025 0.05 0.1 0.25)
      # em_cor
    fi
    for lr in ${lrs[*]}; do
      for dm in ${dms[*]}; do
        for fisher_alpha in ${fisher_alphas[*]}; do
          for em_coe in ${em_coes[*]}; do
            i=$((i + 1))
            wait_n
            if [ "$MODEL_CONTINUAL" == "Continual" ]; then
              CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}_continual" \
                --OPTIM_LR "$lr" --EATA_DM "$dm" --EATA_FISHER_ALPHA "$fisher_alpha" --MODEL_CONTINUAL "$MODEL_CONTINUAL" --EATA_E_MARGIN_COE "$em_coe"&
            else
              CUDA_VISIBLE_DEVICES="$GPUS" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
                --OPTIM_LR "$lr" --EATA_DM "$dm" --EATA_FISHER_ALPHA "$fisher_alpha" --MODEL_CONTINUAL "$MODEL_CONTINUAL" --EATA_E_MARGIN_COE "$em_coe"&
            fi
          done
        done
      done
    done
  #Evaluate t3a
  elif [ "$METHOD" == "t3a" ] || [ "$METHOD" == "t3aE10" ]; then
    filter_ks=( -1 1 5 10 20 50 75 100)
    if [ "$METHOD" == "t3aE10" ] || [ "$MODEL_CONTINUAL" == "Continual" ]; then
      filter_ks=(1 5 10 20 50 75 100)
    fi

    for filter_k in ${filter_ks[*]}; do
      i=$((i + 1))
      wait_n
      if [ "$MODEL_CONTINUAL" == "Continual" ]; then
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}_continual" \
          --T3A_FILTER_K "$filter_k" --MODEL_CONTINUAL "$MODEL_CONTINUAL" &
      else
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
          --T3A_FILTER_K "$filter_k" --MODEL_CONTINUAL "$MODEL_CONTINUAL" &
      fi
    done
  #Evaluate lame
  elif [ "$METHOD" == "lame" ]; then
    affs=( 'kNN' 'rbf' 'linear' )
    KNNs=(1 3 5)
    for aff in ${affs[*]}; do
      for KNN in ${KNNs[*]}; do
        i=$((i + 1))
      wait_n
      CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
        --LAME_AFFINITY "$aff" --LAME_KNN "$KNN" &
      done
    done
  #Evaluate sar
  elif [ "$METHOD" == "sar" ] || [ "$METHOD" == "sarE10" ]; then
    rsts=(0.05 0.1 0.2 0.3 0.5)
    em_coes=(0.4)
    if [ "$DATASET" == "cifar10_c" ] || [ "$DATASET" == "cifar100_c" ]; then
      lrs=(0.0001 0.0002 0.00025 0.0005 0.001 0.002 0.005 0.01)
    elif [ "$DATASET" == "imagenet_c" ] || [ "$DATASET" == "domainnet126" ] || [ "$DATASET" == "officehome" ] || [ "$DATASET" == "imagenet_vit" ]; then
      lrs=(0.00005 0.0001 0.00025 0.0005 0.001 0.002 0.005 0.01)
    elif [ "$DATASET" == "imagenet_convnet" ]; then
      lrs=(0.000002 0.000005 0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005)
    elif [ "$DATASET" == "imagenet_efn" ]; then
      lrs=(0.00005 0.0001 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2)
      em_coes=(0.8)
    fi
    for lr in ${lrs[*]}; do
      for rst in ${rsts[*]}; do
        for em_coe in ${em_coes[*]}; do
          i=$((i + 1))
          wait_n
          if [ "$MODEL_CONTINUAL" == "Continual" ]; then
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}_continual" \
              --OPTIM_LR "$lr" --SAR_RESET_CONSTANT "$rst" --MODEL_CONTINUAL "$MODEL_CONTINUAL" --SAR_E_MARGIN_COE "$em_coe" &
          else
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --SAR_RESET_CONSTANT "$rst" --MODEL_CONTINUAL "$MODEL_CONTINUAL" --SAR_E_MARGIN_COE "$em_coe" &
          fi
        done
      done
    done
  #Evaluate adacontrast
  elif [ "$METHOD" == "adacontrast" ]; then
    lrs=(0.00001 0.00002 0.00005 0.0001 0.0005 0.001)
    num_neighbors=(5 10 15)
    if [ "$DATASET" == "cifar10_c" ] || [ "$DATASET" == "cifar100_c" ]; then
      queue_sizes=(2000 5000 10000)
    elif [ "$DATASET" == "imagenet_c" ] || [ "$DATASET" == "domainnet126" ] || [ "$DATASET" == "imagenet_convnet" ]; then
      queue_sizes=(5000 10000 15000)
    elif [ "$DATASET" == "officehome" ]; then
      queue_sizes=(2000 5000)
    elif [ "$DATASET" == "imagenet_vit" ]; then
      num_neighbors=(5)
      queue_sizes=(15000)
    fi
    for lr in ${lrs[*]}; do
      for num_neighbor in ${num_neighbors[*]}; do
        for queue_size in ${queue_sizes[*]}; do
          i=$((i + 1))
          wait_n
          CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
            --OPTIM_LR "$lr" --ADACONTRAST_NUM_NEIGHBORS "$num_neighbor" --ADACONTRAST_QUEUE_SIZE "$queue_size" &
        done
      done
    done

  #Evaluate deyo
  elif [ "$METHOD" == "deyo" ]; then
    if [ "$DATASET" == "coloredmnist" ]; then
              filter_ent=0
    else
        filter_ent=1
    fi
    lrs=(0.00025 0.00125)
    margins=(0.3 0.4 0.5 1)
    margin_e0s=(0 0.2 0.4 1)
    thrs=(-1 0 0.2 0.4)
    for lr in ${lrs[*]}; do
      for margin in ${margins[*]}; do
        for margin_e0 in ${margin_e0s[*]}; do
          for thr in ${thrs[*]}; do
            i=$((i + 1))
            wait_n
            CUDA_VISIBLE_DEVICES="$GPUS" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --DEYO_MARGIN "$margin" --DEYO_MARGIN_E0 "$margin_e0" --DEYO_PLPD_THRESHOLD "$thr" --DEYO_FILTER_ENT "$filter_ent" &
          done
        done
      done
    done

  elif [ "$METHOD" == "proposal" ]; then
    lrs=(0.00025 0.00125)
    margins=(0.3 0.4 0.5 1)
    margin_e0s=(0 0.2 0.4 1)
    thrs=(-1 0 0.2 0.4)
    new_margins=(0.5 0.6 0.7 0.8 0.9)
    new_margins_e0=(0.5 0.6 0.7 0.8 0.9)
    alphas=(0.5 0.6 0.7)
    filter_ent=1
    layers=(1 2 3)
    for lr in ${lrs[*]}; do
      for margin in ${margins[*]}; do
        for margin_e0 in ${margin_e0s[*]}; do
          for thr in ${thrs[*]}; do
            for new_margin in ${new_margins[*]}; do
              for new_margin_e0 in ${new_margins_e0[*]}; do
                for alpha in ${alphas[*]}; do
                  for layer in ${layers[*]}; do
                    i=$((i + 1))
                    wait_n
                    CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "test-time-validation/${DATASET}/${METHOD}" \
                    --OPTIM_LR "$lr" --DEYO_MARGIN "$margin" --DEYO_MARGIN_E0 "$margin_e0" --DEYO_PLPD_THRESHOLD "$thr" --DEYO_FILTER_ENT "$filter_ent"  --PROPOSAL_NEW_MARGIN "$new_margin"  --PROPOSAL_NEW_MARGIN_E0 "$new_margin_e0"  --PROPOSAL_ALPHA "$alpha" --PROPOSAL_LAYER "$layer"&
                done
              done
            done
          done
        done
      done
    done
  done
  fi

}

test_time_adaptation
