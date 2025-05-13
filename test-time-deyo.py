import logging
import os
import time
import pandas as pd 
import numpy as np
import random
from src.methods import *
from src.models.load_model import load_model
from src.utils import get_accuracy, get_args, evaluate_model, get_accuracy_deyo
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def evaluate(cfg):
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, 
                            checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN, 
                            cfg = cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # base_model = nn.DataParallel(base_model)
    print(device)
    base_model.to(device)

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model, param_names = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "t3a":
        model, param_names = setup_t3a(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
        model, param_names = setup_test_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model, param_names = setup_alpha_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "memo":
        model, param_names = setup_memo(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "tent":
        model, param_names = setup_tent(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model, param_names = setup_cotta(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "lame":
        model, param_names = setup_lame(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "adacontrast":
        model, param_names = setup_adacontrast(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "eata":
        model, param_names = setup_eata(base_model, num_classes, cfg)
    elif cfg.MODEL.ADAPTATION == "sar":
        model = setup_sar(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "deyo":
        model, param_names = setup_deyo(base_model, cfg, num_classes)
    elif cfg.MODEL.ADAPTATION == "proposal":
        model, param_names = setup_nu(base_model, cfg, num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126", "officehome", "pacs"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.SOURCE_DOMAIN)
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = dom_names_all
    
    # setup the severities for the gradual setting

    severities = cfg.CORRUPTION.SEVERITY

    accs = []
    biased = False
    if cfg.CORRUPTION.DATASET in {"coloredMNIST", "waterbirds"}:
        biased = True
    # start evaluation
    folder_save = f"save_deyo/{cfg.CORRUPTION.DATASET}/{cfg.CORRUPTION.SOURCE_DOMAIN}"
    for severity in severities:
        for i_dom, domain_name in enumerate(dom_names_loop):
        # for severity in severities:
            folder = os.path.join(folder_save, domain_name, str(severity))
            if not os.path.exists(folder):
                os.makedirs(folder)
            # try:
            #     model.reset()
            #     logger.info("resetting model")
            # except:
            #     logger.warning("not resetting model")

        # for severity in severities:
            testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                                cfg.TEST.BATCH_SIZE,
                                                split='all', domain=domain_name, level=severity,
                                                adaptation=cfg.MODEL.ADAPTATION,
                                                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                num_aug=cfg.TEST.N_AUGMENTATIONS)

            for epoch in range(cfg.TEST.EPOCH):
                if not biased:
                    acc = get_accuracy_deyo(
                        model, data_loader=test_loader, folder = folder)
                else:
                    acc, LL, LS, SL, SS = evaluate_model(model, data_loader=test_loader, folder = folder)
                if cfg.TEST.EPOCH > 1:
                    print(f"epoch: {epoch}, acc: {acc:.2%}")
                    # logger.info(f"epoch: {epoch}, acc: {acc:.2%}")


            accs.append(acc)
            if not biased:
                logger.info(
                    f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={len(testset)}]: {acc:.2%}")
            else:
                logger.info(f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={len(testset)}]: {acc:.2%}")
                logger.info(f"LL = {LL:.2%}, LS = {LS:.2%}, SL = {SL:.2%}, SS = {SS:.2%}, avg = {(LL + SL + LS + SS)/4:.2%}")

        logger.info(f"mean accuracy: {np.mean(accs):.2%}")
    return accs


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'online_evaluation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    logger.info(cfg)
    # random.seed(cfg.RNG_SEED)
    start_time = time.time()
    accs = []
    for domain in cfg.CORRUPTION.SOURCE_DOMAINS:
        logger.info("#" * 50 + f'evaluating domain {domain}' + "#" * 50)
        cfg.CORRUPTION.SOURCE_DOMAIN = domain
        acc = evaluate(cfg)
        accs.extend(acc)

    logger.info("#" * 50 + 'final result' + "#" * 50)
    logger.info(f"total mean accuracy: {np.mean(accs):.2%}")

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
