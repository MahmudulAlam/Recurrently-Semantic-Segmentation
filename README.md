## Recurrently Semantic Segmentation
<p align="justify">
  In this paper, a recurrent method of semantic segmentation is proposed where convolutional LSTM is used to generate a mask of objects present in the image recurrently. The proposed method only generates a mask of objects that are only present in the image. Therefore, requiring less output tensor space. The method uses Unet architecture and convolutional LSTM in the bottleneck and is trained using the PASCAL VOC dataset.
</p>

## Requirements
[![Downloads](https://img.shields.io/badge/download-weights-fc2003.svg?style=popout-flat&logo=mega)](https://mega.nz/folder/6lE0TLKQ#9JDOk31P3HAuHVrJuimJRg)

- PyTorch==1.8.1
- Weights: [```Download the pre-trained weights```](https://mega.nz/folder/6lE0TLKQ#9JDOk31P3HAuHVrJuimJRg) file of the recurrent semantic segmentation model and put the ```weights``` folder in the working directory.

## Preview
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/146846843-1feef668-0af7-42c0-8e8d-bb2b4888ed81.gif" width="256">
</p>

## Network Architecture 
The proposed method is implemented using the following network architecture where encoder and decoder are used from U-net architecture and convolutional LSTM is used in the bottleneck. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/146846653-96bbdbff-77eb-44ea-8395-5be6d40d5eac.jpg">
</p>

## Results
The experimental results are reported in terms of accuracy and mean interaction over union (IOU) between ground truth and predicted mask. The mean accuracy among all the classes is ```30.02%``` and the mean IOU achieved among all the classes is ```43.77%```, and the maximum accuracy and IOU achieved are ```70%``` and ```71.54%```, subsequently.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/146848376-936c1f41-d140-49da-9fe5-d0a8b1e58843.jpg" width="700">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/146848385-7e0135cc-e261-436f-b668-f0a3c9b0c24c.jpg" width="700">
</p>
