import os
import sys
import argparse
import logging
import datetime
import random
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter


import numpy as np

from csi.config import CfgNode
from csi.utils.file_io import PathManager

def default_argument_parser(epilog=None):

    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
            Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")

    return parser


def _try_get_key(cfg, *keys, default=None):
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default

def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def default_setup(cfg, args):

    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if output_dir:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        output_dir = output_dir + date_time
        val_dir = os.path.join(output_dir, "val")
        test_dir = os.path.join(output_dir, "test")
        PathManager.mkdirs(output_dir)
        PathManager.mkdirs(val_dir)
        PathManager.mkdirs(test_dir)

    logger = gen_log(output_dir)
    writer = SummaryWriter(log_dir=output_dir)


    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )
    
    if output_dir:
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    seed = _try_get_key(cfg, "SEED", "train.seed", default=3407)
    seed_everything(seed, deterministic=cfg.DETERMINISTIC)

    return logger, writer, output_dir


def seed_everything(
    seed = 3407,
    deterministic = False, 
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False