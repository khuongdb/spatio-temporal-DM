import argparse
import logging

import yaml


def parse_args():
    """
    parser: argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_level', type=str, default='INFO', metavar='L',
                        help='log level from : ["INFO", "DEBUG", "WARNING", "ERROR"]')

    parser.add_argument('--config', type=str, default='projects/dummy_project/config/sadm.yaml', metavar='C',
                        help='path to configuration yaml file')

    return parser.parse_args() 


def main():
    args = parse_args()

    if args.log_level == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.log_level == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == 'WARNING':
        logging.basicConfig(level=logging.WARNING)
    elif args.log_level == 'ERROR':
        logging.basicConfig(level=logging.ERROR)

    # cfgs = None

    logging.info(
        '------------------------------- LONGITUDINAL DIFFUSION MODEL  -------------------------------')

    try:
        with open(args.config, 'r') as cfgs_f:
            cfgs = yaml.load(cfgs_f, Loader=yaml.FullLoader)
            logging.info(f"Success: Loaded configuration file at: {args.config}")
    except Exception as e:
        logging.error(f"ERROR: Could not load config file: {e}")
        exit()


    exp = cfgs.get("experiment", None)
    if exp["task"] == "train":
        print("Start training")
        train_args = cfgs["train_ds"]
        print(train_args)
    elif exp["task"] == "test":
        print("Start inference")
    else: 
        print("Task need to be train or test.")
        exit()


if __name__=="__main__":
    main()