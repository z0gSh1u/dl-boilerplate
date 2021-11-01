'''
    Test manually with trained ckp.
'''

# ### If you need to import from other folder ###
import os.path as path
import sys

dirname__ = path.dirname(path.abspath(__file__))
sys.path.append(path.join(dirname__, '..'))
# ### ###

OUTPUT_PATH = r''
CKP_PATH = r''  # weights to use.

TestOpts = {
    # Fill it.
}

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from useful.meter import AverageMeter
from tqdm import tqdm
from PIL import Image
import numpy as np

from model import NETWORK_NAME
from dataset import YOUR_DATASET

tb = SummaryWriter(os.path.join(OUTPUT_PATH, './tensorboard'))

noop = lambda: None

if __name__ == '__main__':
    with torch.no_grad():
        # Load model.
        model = NETWORK_NAME()
        model = model.cuda()
        model.load_state_dict(torch.load(CKP_PATH))
        model.eval()

        # Load dataset.
        Dataset_ = YOUR_DATASET
        val_set = Dataset_()
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

        # ###### Testing ######
        epoch_somewhat = AverageMeter()

        for data in tqdm(val_loader, ncols=80, desc='[Testing]'):
            x, filename = data

            # Pass through the netwrok.
            with torch.no_grad():
                predict = model(x).detach().cpu().clamp(0, 1)

            # Squeeze BatchSize.
            predict = predict.squeeze()

            # Update record.
            epoch_somewhat.update(noop(predict))

            # Save test result.
            predict_np = np.array(predict.numpy() * 255, dtype=np.uint8)
            predict_np = Image.fromarray(np.array(predict_np.detach().cpu().numpy().squeeze() * 255, dtype=np.uint8))
            predict_np.save(os.path.join(OUTPUT_PATH, './test_inference/', '{}.png'.format(filename[0])))

        # Print the metric.
        print('Test What: {:.2f}'.format(epoch_somewhat.avg))

        # Save to Tensorboard.
        tb.add_scalar('Test What', epoch_somewhat.avg, global_step=0)
