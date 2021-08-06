# Image-Inpainting-Project-based-on-CVPR-2020-Oral-Paper-on-HiFill--Tensorflow-2.5-version


The original repo is [here](https://github.com/Atlas200dk/sample-imageinpainting-HiFill) with tensorflow 1x code. 
I turn tf 1x code to tf 2.5, I test it on google colab and it's working. 
My input image shape is (512, 512, 3) and the masked image shape is (512, 512, 1)

***
* cd Image-Inpainting-Project-based-on-CVPR-2020-Oral-Paper-on-HiFill---Tensorflow-2.5-version/
* python inpainting.py --images /content/images --masks /content/masks --model /content/hifill.pb --results results
***
|     Input        |    Masked   |     Inpainted      |
:-----------------:|:------------|--------------------:
![](https://user-images.githubusercontent.com/35764362/128516771-da511ee3-acf1-4cce-bdd4-a98aa041aadc.jpg) | ![](https://user-images.githubusercontent.com/35764362/128516821-43085b88-2002-4d87-9713-dc66e6327f2e.jpg) | ![](https://user-images.githubusercontent.com/35764362/128516856-4e1bd999-914b-4fd0-82e0-226e888228a7.jpg)
![](https://user-images.githubusercontent.com/35764362/128516921-43d96199-ad15-4e0d-a9a7-57ea318282e2.jpg) | ![](https://user-images.githubusercontent.com/35764362/128516956-e81f6ba4-b7ee-4211-a75c-05903e2a6678.jpg) | ![](https://user-images.githubusercontent.com/35764362/128516977-ca82b1df-3724-41fb-a208-0c96f5ba0910.jpg)
![](https://user-images.githubusercontent.com/35764362/128517270-e5e15370-d842-4a95-8dae-da82beca48f3.jpg) | ![](https://user-images.githubusercontent.com/35764362/128517303-a85253a2-0f83-473d-b487-92ba7d11c2a5.jpg) | ![](https://user-images.githubusercontent.com/35764362/128517314-294af7ca-c0dd-4fbb-9dce-fd2ed0c8b6ca.jpg)





