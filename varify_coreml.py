import coremltools as ct
from colorizers import *
import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with coreml.png suffixes')
opt = parser.parse_args()

# load coreml model
coreml_model = ct.models.MLModel('Colorizer.mlpackage')

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

tens_ab_rs = coreml_model.predict({'input1': tens_l_rs.numpy()})['var_518']
out_img_coreml = postprocess_tens(tens_l_orig, torch.from_numpy(tens_ab_rs))

# save output image
plt.imsave('%s_coreml.png'%opt.save_prefix, out_img_coreml)

# visualize
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(out_img_coreml)
plt.title('CoreML Output')
plt.axis('off')
plt.show()