### PyTorch Implementation of U-Net, R2U-Net
**U-Net: Convolutional Networks for Biomedical Image Segmentation**
https://arxiv.org/abs/1505.04597

**Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation**
https://arxiv.org/abs/1802.06955

# Results
You can see the stable version and finals results in Google Colab:
https://colab.research.google.com/drive/1C16K0BN5W6dATF1ubeibTl8vXgb6MDFj?usp=sharing

![image](https://github.com/user-attachments/assets/00ff0cd4-c5d7-4e81-a56a-827621354657)
![image](https://github.com/user-attachments/assets/f59deef9-c3aa-4cf7-8026-702a9ed6c900)


## U-Net
![image](https://github.com/HosseinEn/computer-vision-RBVS/assets/83599557/6b9876a3-5d41-41ae-a179-7f4a6a0f7656)

## R2U-Net
![image](https://github.com/HosseinEn/computer-vision-RBVS/assets/83599557/5bed296a-b7c7-4646-8965-1494896b1bc6)


## Evaluation
I tested the R2U-Net model with [CHASEDB1](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/) and [DRIVE](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) dataset. The dataset was split into three subsets: training set, validation set, and test set, which the proportion is 60%, 20% and 20% of the whole dataset, respectively.
