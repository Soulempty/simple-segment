import time
import argparse
from apis import Segment 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/wafer.yaml')
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--class_names', type=str, nargs="+", default=None,)
    parser.add_argument('--lr', type=float, default=None) 
    parser.add_argument('--base_size', type=int, nargs=2, default=None)  
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    return args
# example for train api call.

def train():   
    args = parse_args()
    segment = Segment(args.config_path)
    
    # set param by demanding with api set_func,such as bellow.
    segment.set_lr(args.lr)
    segment.set_datapath(args.data_path)
    segment.set_epoch(args.epochs)
    segment.set_workdir(args.work_dir)
    segment.set_classnames(args.class_names)
    segment.set_batchsize(args.batch_size)
    segment.set_basesize(args.base_size)
    segment.init_config()

    segment.train()

if __name__ == "__main__":
    train()
