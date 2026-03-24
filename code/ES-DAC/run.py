import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from config import get_config_regression, get_config_tune
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


SUPPORTED_MODELS = ['ES_DAC']
SUPPORTED_DATASETS = ['MOSI', 'MOSEI', 'SIMS']

logger = logging.getLogger('MSA')

from datetime import datetime      
now = datetime.now()
format = "%Y%m%d_%H%M%S"
format_1 = "%y%m%d_%H%M%S"
formatted_now = now.strftime(format)
formatted_now_1 = now.strftime(format_1)
epoch_num = 100
ResultName = ""


def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}-{formatted_now}.log"
    logger = logging.getLogger('MSA') 
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def MSA_run(
    model_name: str, 
    dataset_name: str, 
    config_file: str = None,
    config: dict = None, 
    seeds: list = [], 
    is_tune: bool = False,
    tune_times: int = 50, 
    custom_feature: str = None, 
    feature_T: str = None, 
    feature_A: str = None, 
    feature_V: str = None, 
    feature_A_LLD: str = None,
    gpu_ids: list = [0],
    num_workers: int = 1, 
    verbose_level: int = 1,
    model_save_dir: str = Path().home() / "MSA" / "saved_models",
    res_save_dir: str = Path().home() / "MSA" / "results",
    log_dir: str = Path().home() / "MSA" / "logs",
):
    # Initialization
    model_name = model_name.lower()
    MODEL_NAME = model_name.upper()
    dataset_name = dataset_name.lower()
    
    if config_file is not None:
        config_file = Path(config_file)
    else: # use default config files
        if is_tune:
            config_file = Path(__file__).parent / "config" / "config_tune.json"
        else:
            config_file = Path(__file__).parent / "config" / "config_regression.json"
    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    if model_save_dir is None: # use default model save dir
        model_save_dir = Path.home() / "MSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir is None: # use default result save dir
        res_save_dir = Path.home() / "MSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir is None: # use default log save dir
        log_dir = Path.home() / "MSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    logger.info("======================================== Program Start ========================================")
    
    if is_tune: # run tune
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        initial_args['custom_feature'] = custom_feature
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_A_LLD'] = feature_A_LLD
        initial_args['feature_V'] = feature_V

        if str(initial_args['device']).startswith('cuda'):
            torch.cuda.set_device(initial_args['device'])

        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = [] # save used params
        # csv_file = res_save_dir / f"{MODEL_NAME}_{dataset_name}{result_name}.csv"
        csv_file = res_save_dir / f"{MODEL_NAME}_{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            args.update(new_args)
            if config:
                if config.get('model_name'):
                    assert(config['model_name'] == args['model_name'])
                args.update(config)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            # check if this param has been run
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            # actual running
            setup_seed(seeds[0])
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)
            # save result to csv file
            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                value = result[col]
                res.append(value)
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")

            
    else: # run normal
        args = get_config_regression(model_name, dataset_name, config_file)
        args['result_name'] = ResultName
        args['model_name'] = model_name
        args['model_save_dir'] = Path(model_save_dir) / f"{args['dataset_name']}"
        args['model_save_path'] = Path(model_save_dir) / f"{args['dataset_name']}" / f"{args['model_name']}{args.result_name}_{args['dataset_name']}.pth"
        Path(args['model_save_path']).parent.mkdir(parents=True, exist_ok=True)
        args['device'] = assign_gpu(gpu_ids)
        args['train_mode'] = 'regression'
        args['custom_feature'] = custom_feature
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        args['feature_A_LLD'] = feature_A_LLD
        args['epochs'] = epoch_num

        if config: # override some arguments
            if config.get('model_name'):
                assert(config['model_name'] == args['model_name'])
            args.update(config)

        if str(args['device']).startswith('cuda'):
            torch.cuda.set_device(args['device'])

        # logger.info("Running with args:")
        # logger.info(args)
        logger.info(f"Model Name: {args.model_name}")
        logger.info(f"Dataset Name: {args.dataset_name}")
        logger.info(f"Task Type: {args.train_mode}")

        logger.info(f"Seeds: {seeds}")
        
        res_save_dir = Path(res_save_dir) / f"{model_name}"
        res_save_dir.mkdir(parents=True, exist_ok=True)

        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-'*30}")

            # actual running
            epoch_results = _run(args, num_workers, is_tune)
            # logger.info(f"Result for seed {seed}: {result}")

        # save result to csv
        csv_file = res_save_dir / f"{MODEL_NAME}_{dataset_name}{args.result_name}.csv"
        if csv_file.is_file():
            csv_file_1 = csv_file
            csv_file = res_save_dir / f"{MODEL_NAME}_{dataset_name}{args.result_name}_{formatted_now_1}.csv"
            res_save_dir.parent.mkdir(parents=True, exist_ok=True)
            print("\033[31m" + f" {csv_file_1} 已存在, 已另存为{csv_file}, 请到 {res_save_dir} 检查！." + "\033[0m")
        # else:
        #     df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = []
        # criterions = list(epoch_results[0]['train'].keys())
        criterions = list(epoch_results['train'][0].keys())

        
        columns = []
        for c in criterions:
            columns.append(c)
            # columns.extend([f"{c}"])
        df = pd.DataFrame(columns=columns)

        rows = []
        num_epochs = len(epoch_results['train'])
        for epoch in range(num_epochs):
            row = {'epoch': epoch + 1}
            # for phase in ['train', 'valid', 'test']:
            for phase in ['valid']:
                for metric, value in epoch_results[phase][epoch].items():
                    if metric != 'Loss':
                        row[f"{phase}_{metric}"] = value
            rows.append(row)
            row['Train_Loss'] = epoch_results['train'][epoch]['Loss']
            row['Val_Loss'] = epoch_results['valid'][epoch]['Loss']

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
                

        # for c in criterions:
        #     values = [r[c] for r in model_results]
        #     mean = round(np.mean(values), 8)
        #     # std = round(np.std(values), 8)
        #     # res.append((mean, std))
        #     res.append((mean))

        # df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=True):
    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args['device'])

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # TODO: use multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    trainer = ATIO().getTrain(args)
    # do train
    # epoch_results = trainer.do_train(model, dataloader)
    # epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
    epoch_results = trainer.do_train(model, dataloader)

    
    # load trained model & do test
    assert Path(args['model_save_path']).exists()
    model.load_state_dict(torch.load(args['model_save_path']))
    model.to(args['device'])
    if from_sena:
        final_results = {}
        # final_results['train'] = trainer.do_test(model, dataloader['train'], mode="TRAIN", return_sample_results=True)
        # final_results['valid'] = trainer.do_test(model, dataloader['valid'], mode="VALID", return_sample_results=True)
        # final_results['test'] = trainer.do_test(model, dataloader['test'], mode="TEST", return_sample_results=True)
    elif is_tune:
        # use valid set to tune hyper parameters
        # results = trainer.do_test(model, dataloader['valid'], mode="VALID")
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        # delete saved model
        Path(args['model_save_path']).unlink(missing_ok=True)
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    if str(args['device']).startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()


    # return {"epoch_results": epoch_results, 'final_results': final_results} if from_sena else results
    return epoch_results if from_sena else results



def MSA_test(
    config: dict | str,
    weights_path: str,
    feature_path: str, 
    # seeds: list = [], 
    gpu_id: int = 0, 
):
    """Test MSA models on a single sample.

    Load weights and configs of a saved model, input pre-extracted
    features of a video, then get sentiment prediction results.

    Args:
        model_name: Name of MSA model.
        config: Config dict or path to config file. 
        weights_path: Pkl file path of saved model weights.
        feature_path: Pkl file path of pre-extracted features.
        gpu_id: Specify which gpu to use. Use cpu if value < 0.
    """
    if type(config) == str or type(config) == Path:
        config = Path(config)
        with open(config, 'r') as f:
            args = json.load(f)
    elif type(config) == dict or type(config) == edict:
        args = config
    else:
        raise ValueError(f"'config' should be string or dict, not {type(config)}")
    args['train_mode'] = 'regression' # backward compatibility.

    if gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    args['device'] = device
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    # args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1]]
    args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1], feature['audio_LLD'].shape[1]]
    # args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0]]
    args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0], feature['audio_LLD'].shape[0]]
    model = AMIO(args)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.get('use_bert', None):
            if type(text := feature['text_bert']) == np.ndarray:
                text = torch.from_numpy(text).float()
        else:
            if type(text := feature['text']) == np.ndarray:
                text = torch.from_numpy(text).float()
        if type(audio := feature['audio']) == np.ndarray:
            audio = torch.from_numpy(audio).float()
        if type(vision := feature['vision']) == np.ndarray:
            vision = torch.from_numpy(vision).float()
        text = text.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        vision = vision.unsqueeze(0).to(device)
        if args.get('need_normalized', None):
            audio = torch.mean(audio, dim=1, keepdims=True)
            vision = torch.mean(vision, dim=1, keepdims=True)
        # TODO: write a do_single_test function for each model in trains
        if args['model_name'] == 'self_mm' or args['model_name'] == 'mmim':
            output = model(text, (audio, torch.tensor(audio.shape[1]).unsqueeze(0)), (vision, torch.tensor(vision.shape[1]).unsqueeze(0)))
        elif args['model_name'] == 'tfr_net':
            input_mask = torch.tensor(feature['text_bert'][1]).unsqueeze(0).to(device)
            output, _ = model((text, text, None), (audio, audio, input_mask, None), (vision, vision, input_mask, None))
        else:
            output = model(text, audio, vision)
        if type(output) == dict:
            output = output['M']
    return output.cpu().detach().numpy()[0][0]
        

def train(dataset_name='mosi', feature_A_LLD_path=None):
    with open('config/config_regression.json', 'r') as f:
        config = json.load(f)

    if feature_A_LLD_path:
        config['feature_A_LLD'] = feature_A_LLD_path 

    MSA_run(
        model_name='ES_DAC',
        dataset_name=dataset_name,
        config=config,
        seeds=[1111],
        is_tune=False,
        model_save_dir="./saved_models",
        res_save_dir="./results",
        log_dir="./logs",
        # gpu_ids=[-1],
        feature_A_LLD=feature_A_LLD_path,
        # num_workers=0,
    )

def test(dataset_name='mosi'):
    with open('config/config_regression.json', 'r') as f:
        config = json.load(f)

    MSA_run(
        model_name='ES_DAC',
        dataset_name=dataset_name,
        config=config,
        seeds=[1111],
        is_tune=False,
        model_save_dir="./saved_models",
        res_save_dir="./results",
        log_dir="./logs",
        # gpu_ids=[-1],
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset_name', type=str, default='mosi')
    parser.add_argument('--feature_A_LLD_path', type=str, default='D:/DL/MSA/paper_code/dataset/MOSI/Processed/audio_LowLevelDescriptors.pkl')
    args = parser.parse_args()

    if args.mode == 'train':
        train(dataset_name=args.dataset_name, feature_A_LLD_path=args.feature_A_LLD_path)
    else:
        test(dataset_name=args.dataset_name)