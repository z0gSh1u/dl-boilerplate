'''
    Training code for your model.
'''

# ### If you need to import from other folder ###
import os.path as path
import sys

dirname__ = path.dirname(path.abspath(__file__))
sys.path.append(path.join(dirname__, '..'))
# ### ###

# Hyper Parameters.
LR = 1e-3
EPOCH = 500
BATCH_SIZE = 32
VAL_EVERY = 10
OUTPUT_PATH = r''

TrainOpts = {
    # Dataset.
    'train_dir': r'',
    'val_dir': r'',

    # Data specified paramters.
    'crop_size': 256,

    # Data loading.
    'num_workers': 16,
}

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import optim, lr_scheduler
from useful.meter import AverageMeter

from model import NETWORK_NAME
from dataset import YOUR_DATASET

tb = SummaryWriter(os.path.join(OUTPUT_PATH, './tensorboard'))


# Customize a loss function here.
class MY_LOSS_FUNCTION(nn.Module):
    def __init__(self):
        super(MY_LOSS_FUNCTION, self).__init__()

    def forward(self, x, y):
        # Define how to forward your loss.
        return x - y


noop = lambda: None  # noop

if __name__ == '__main__':
    model = NETWORK_NAME()
    model = model.cuda()

    # Adam optimizer
    criterion = MY_LOSS_FUNCTION()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    # Define how to decay your learning rate.
    scheduler = lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.9)

    # Load the dataset.
    Dataset_ = YOUR_DATASET  # Your dataset interface.
    train_set = Dataset_()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=TrainOpts['num_workers'])
    val_set = Dataset_()
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    # ###### Train ######
    for epoch in range(EPOCH):
        model.train()
        epoch_loss = AverageMeter()

        with tqdm(total=(len(train_set) - len(train_set) % BATCH_SIZE), ncols=80, desc='[Training]') as t:
            t.set_description('Epoch: {}/{}'.format(epoch + 1, EPOCH))

            for data in train_loader:
                lr, hr = data
                lr = lr.cuda()
                hr = hr.cuda()

                # Pass through the network.
                predict = model(lr).cuda()
                loss = criterion(predict, hr)

                # Update loss record.
                epoch_loss.update(torch.mean(loss))

                # Optimize.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg))
                t.update(len(lr))

            tb.add_scalar('Train Loss', epoch_loss.avg, global_step=epoch)

            # Decay learning rate.
            scheduler.step()

            # Save checkpoint.
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, './ckp/', 'epoch_{}.pth'.format(epoch)))

            # ###### Validation ######
            if epoch == 0 or (epoch + 1) % VAL_EVERY == 0:
                model.eval()
                model.valing = True

                epoch_what_metric = AverageMeter()

                for data in tqdm(val_loader, ncols=80, desc='[Validation]'):
                    lr, hr, cb_lr, cr_lr, filename = data
                    lr = lr.cuda()
                    hr = hr.cuda()

                    # Pass through the network with gradients frozen.
                    with torch.no_grad():
                        predict = model(lr).detach().cpu().clamp(0, 1)

                    # squeeze BatchSize
                    predict = predict.squeeze()
                    hr = hr.squeeze()

                    # Update record.
                    epoch_what_metric.update(noop(predict, hr.cpu()))

                    # Save validation result.
                    predict_np = np.array(predict.numpy() * 255, dtype=np.uint8)
                    predict_np.save(os.path.join(OUTPUT_PATH, './val_inference/', '{}.png'.format(filename[0])))

                # Print the metric.
                print('Val What: {:.2f}'.format(epoch_what_metric.avg))

                # Save to Tensorboard.
                tb.add_scalar('Val What', epoch_what_metric.avg, global_step=epoch)
