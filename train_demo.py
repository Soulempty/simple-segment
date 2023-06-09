import time
from apis import Segment 

class DynamicDrawing:
    def __init__(self,iters,base_lr):
        super().__init__()
        self.iters = iters
        self.base_lr = base_lr

        self.left_time = 0
        self.iter_list = []
        self.train_mean_iou = []
        self.test_mean_iou = []
        self.train_loss = []
        self.lr = []

    def add_data(self,data):
        self.iter_list.append(data['iter'])
        self.test_mean_iou.append(data['test_mean_iou'])
        self.train_loss.append(data['loss'])
        self.lr.append(data['lr'])
        self.left_time = data['left_time']
    def draw(self):
        print("iter: ",self.iter_list,"loss: ",self.train_loss,"miou: ",self.test_mean_iou,"lr: ",self.lr)

# example for train api call.
def train(config_path='config.yaml'):   
    segment = Segment(config_path)
    iters = segment.iters
    lr = segment.lr
    epochs = segment.epochs
    draw = DynamicDrawing(iters,lr)

    # set param by demanding with api set_func,such as bellow.
    # segment.set_lr(0.002)
    # segment.set_datapath('dataset/wafer')
    # segment.set_epoch(160)
    # segment.set_workdir('work_dir')

    segment.train(draw.add_data)
    time.sleep(300)
    print(draw.draw())

if __name__ == "__main__":
    train()
    