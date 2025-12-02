import os
from loss.Saturation_Loss import GradientLoss
import numpy as np
from torchvision import transforms
from model import Shift_Net
import torch.optim as optim
import scipy.io as scio
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
import torchvision
import torch
import torch.nn as nn

train_data_path = 'dataset_XXX/'
root_high = train_data_path + 'high/'
root_low = train_data_path+'low/'

train_path = 'checkpoints/dataset_XXX/'
device = "cuda"

batch_size = 20
epochs = 360
lr = 1e-2

Train_Image_Number = len(os.listdir(train_data_path+'high/high'))
Iter_per_epoch = (Train_Image_Number % batch_size != 0)+Train_Image_Number//batch_size

transforms = transforms.Compose([
    transforms.CenterCrop(180),
    transforms.ToTensor(),
])

Data_high = torchvision.datasets.ImageFolder(root_high, transform=transforms)
dataloader_high = torch.utils.data.DataLoader(Data_high, batch_size, shuffle=False)

Data_low = torchvision.datasets.ImageFolder(root_low, transform=transforms)
dataloader_low = torch.utils.data.DataLoader(Data_low, batch_size, shuffle=False)

lowlight_enhancement = Shift_Net()
is_cuda = True
if is_cuda:
    lowlight_enhancement = lowlight_enhancement.cuda()

optimizer = optim.Adam(lowlight_enhancement.parameters(), lr = lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160, 200, 240, 280, 320], gamma=0.1)
#-----------------------------------------------------------------------------------------------------------------------
loss_l1 = nn.SmoothL1Loss(reduction='mean')
loss_ED = GradientLoss(operator='Sobel', channel_mean=True, r1=1.1, r2=1, r3=0.04)
loss_train=[]
loss_l1_train=[]
loss_ED_train=[]
loss_ssim_train=[]
lr_list=[]

for iteration in range(epochs):

    lowlight_enhancement.train()

    data_iter_high = iter(dataloader_high)
    data_iter_low = iter(dataloader_low)

    for step in range(Iter_per_epoch):
        data_high, _ = next(data_iter_high)
        data_low, _ = next(data_iter_low)

        if is_cuda:
            data_high = data_high.cuda()
            data_low = data_low.cuda()

        optimizer.zero_grad()
        mid1, mid2, out = lowlight_enhancement(data_low)
        loss_l1_val = loss_l1(out, data_high)
        loss_ED_val = loss_ED(mid1, mid2, out, data_high)
        loss_ssim_val = 1-ms_ssim(out, data_high, data_range=1.0, size_average=True)
        loss = 0.025 * loss_ssim_val + (1 - 0.025) * loss_l1_val + loss_ED_val
        loss.backward()
        optimizer.step()
        loss_total = loss.item()
        print('Epoch/step: %d/%d, loss: %.7f, lr: %f' % (
        iteration + 1, step + 1, loss_total, optimizer.state_dict()['param_groups'][0]['lr']))

        loss_train.append(loss.item())
        loss_l1_train.append(loss_l1_val.item())
        loss_ED_train.append(loss_ED_val.item())
        loss_ssim_train.append(loss_ssim_val.item())
    scheduler.step()

    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    torch.save({'weight': lowlight_enhancement.state_dict(), 'epoch': epochs},
               os.path.join(train_path, 'Net_weight.pkl'))

    scio.savemat(os.path.join(train_path, 'TrainData.mat'),
                 {'Loss': np.array(loss_train),
                  'loss_l1_train': np.array(loss_l1_train),
                  'loss_ED_train': np.array(loss_ED_train),
                  'loss_ssim_train': np.array(loss_ssim_train),
                  })
    scio.savemat(os.path.join(train_path, 'TrainData_plot_loss.mat'),
                 {'loss_train': np.array(loss_train),
                  'loss_l1_train': np.array(loss_l1_train),
                  'loss_ED_train': np.array(loss_ED_train),
                  'loss_ssim_train': np.array(loss_ssim_train),
                  })

    def Average_loss(loss):
        return [sum(loss[i * Iter_per_epoch:(i + 1) * Iter_per_epoch]) / Iter_per_epoch for i in
                range(int(len(loss) / Iter_per_epoch))]

    plt.figure(figsize=[12, 8])
    plt.subplot(2, 3, 1), plt.plot(Average_loss(loss_train)), plt.title('Loss')
    plt.subplot(2, 3, 3), plt.plot(Average_loss(loss_l1_train)), plt.title('loss_L1')
    plt.subplot(2, 3, 4), plt.plot(Average_loss(loss_ED_train)), plt.title('loss_ED')
    plt.subplot(2, 3, 5), plt.plot(Average_loss(loss_ssim_train)), plt.title('loss_ssim')
    plt.tight_layout()
    plt.savefig(os.path.join(train_path, 'curve_per_epoch.jpg'))