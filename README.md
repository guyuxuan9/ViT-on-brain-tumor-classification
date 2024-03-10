# ViT-on-brain-tumor-classification

This repo aims to classify the tumor using Vision Transformer (ViT)
- [Reference code](https://www.kaggle.com/code/ebrahimelgazar/vision-transformer-vit-keras-pretrained-models/notebook) from kaggle
- Original paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Dataset](https://www.kaggle.com/datasets/denizkavi1/brain-tumor) from kaggle
- 3 classes:
  - Meningioma  (708 slices)
  - Glioma (1426 slices)
  - Pituitary tumor (930 slices)
- MLP head is replaced with BatchNorm-->3*Dense Layers

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/48ef9a0c-72bf-4aa4-b9be-9ec3961f139a)

- Each image is reshaped to 224*224.
- Transfer learning is used on the pre-trained ViT B16 model. The patch size is 16*16. Hence, there are (224/16) * (224/16) = 14 * 14 = 196 patches. Each patch ($\mathbf{x}_p^i$) is passed to a linear projection layer (multiplied by a matrix $E$), and after which each patch becomes a 1D vector of dimension $D$. A token $[CLS]$ is pre-pend to the vectors to indicate this is a classification problem. This sequence of vector is then passed to the transformer encoder.


- $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$ is composed of $N$ patches, where ($P$, $P$) is the resolution of each image patch and $C$ is the # of channels

- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$, where $D$ is the latent vector size.

- The input to the transformer is a vector
  
$$
z_0 =
\begin{bmatrix}
x_{\text{class}}\\ 
x^1_{\text{p}}E\\ 
x^2_{\text{p}}E\\ 
\ldots\\
x^N_{\text{p}}E
\end{bmatrix} + E_{\text{pos}}, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D},  \quad z_0, E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}
$$

# Evaluate the performance
## Classification report
![image](https://github.com/guyuxuan9/ViT-on-brain-tumor-classification/assets/58468284/6fb62202-452f-4774-aef1-7e7d70fb092b)

## Confusion matrix
x axis: ground truth; y axis: predicted value
![image](https://github.com/guyuxuan9/ViT-on-brain-tumor-classification/assets/58468284/214040be-ff85-4228-a288-ce167c4ee2fc)

## 9 randomly selected samples
![image](https://github.com/guyuxuan9/ViT-on-brain-tumor-classification/assets/58468284/f83705be-94e2-4876-8c96-ddfb8ed435ec)

## 9 randomly selected output probabilities
![image](https://github.com/guyuxuan9/ViT-on-brain-tumor-classification/assets/58468284/a15af407-6e15-46fb-84f5-993aa30e0213)

## 9 randomly selected misclassified samples
![image](https://github.com/guyuxuan9/ViT-on-brain-tumor-classification/assets/58468284/775a4037-91bf-41cb-b664-1399a6d583b5)

