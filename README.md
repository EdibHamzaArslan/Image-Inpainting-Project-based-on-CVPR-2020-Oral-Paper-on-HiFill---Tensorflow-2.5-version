# Image-Inpainting-Project-based-on-CVPR-2020-Oral-Paper-on-HiFill--Tensorflow-2.5-version


The original repo is [here](https://github.com/Atlas200dk/sample-imageinpainting-HiFill) with tensorflow 1x code. 
I turn tf 1x code to tf 2.5, I test it on google colab and it's working. 
My input image shape is (512, 512, 3) and the masked image shape is (512, 512, 1)

***
* cd Image-Inpainting-Project-based-on-CVPR-2020-Oral-Paper-on-HiFill---Tensorflow-2.5-version/
* python inpainting.py --images /content/images --masks /content/masks --model /content/hifill.pb --results results
***
