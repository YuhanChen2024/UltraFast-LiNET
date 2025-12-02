from PIL import Image
import os
from data_test import Test_img
import torchvision
from torchvision import transforms

test_data_path = 'test_paper/'

tfs_full = transforms.Compose([
        transforms.ToTensor()
    ])
Test_Image_Number=len(os.listdir(test_data_path))
print(Test_Image_Number)

for i in range(int(Test_Image_Number)):

    Test_low = Image.open(test_data_path+str(i+1)+'.jpg').convert('RGB')
    low = tfs_full(Test_low).unsqueeze(0)
    mid1, mid2, out = Test_img(low)

    torchvision.utils.save_image(out, '%d.jpg' % i, padding=0)