import time
import torch
from model import Shift_Net



def Test_img(img_test1):
    Netshow = Shift_Net().cuda()
    Netshow.load_state_dict(torch.load(
        "train_result/XXXX/XXXX.pkl"
    )['weight'])

    Netshow.eval()

    img_test1 = img_test1.cuda()

    with torch.no_grad():
        a = time.time()
        mid1, mid2, out = Netshow(img_test1)
        b = time.time()
        print(b-a)
    return mid1, mid2, out