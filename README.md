# Kolmogorov-Arnold-Transformer (KAT)
_Note: This repository contains the official implementation and experimental code for the Bachelor's Thesis **"Kolmogorov-Arnold-Transformer: Investigation of Hybrid KAN-Transformer Architectures regarding Scaling, Efficiency, and Interpretability"** submitted to *Rhine-Waal University of Applied Sciences*._

## Abstract

Transformers have become the de-facto standard in Deep Learning but rely heavily on Multi-Layer Perceptrons (MLPs), which constitute the majority of non-embedding parameters. This work investigates whether **Kolmogorov-Arnold Networks (KANs)**—which use learnable activation functions on edges rather than fixed activations on nodes—can replace MLPs in Transformer architectures to improve parameter efficiency and interpretability. 

Based on the literature, five architecture-variants are evaluated over three model sizes ranging from total parameter counts of 15M-124M. The architectures include different basis functions for the Kolmogorov-Arnold Network (rational vs. b-spline) as well as a MLP with learnable b-spline activations as control-group to isolate the effect of the KAN-Architecture from the learnable activation function (i. e. b-splines). Tasks for benchmarking were text-classification ("AG News") and generative language modeling ("FineWeb").

While KANs achieve performance-parity on discriminative tasks such as text-classification, they underperform on language modeling compared with MLP-Transformers. The MLP with learnable b-spline activation showed performance comparable to the baseline (i. e. MLP with fixed GELU activations), which suggests that most of the difference in performance stems from the topological structure rather than from the learnable activations themselves. Notably, KANs with b-spline basis functions exhibit sparse representations of up to $ \approx 90$% (of feed-forward params.), while MLP reaches $\approx 40$% sparsity.
## Architectures

The repository implements a modular Transformer with swappable Feed-Forward Network (FFN) blocks. To ensure fair comparison, all models are parameter-matched ($\pm 1\%$).

| ID | Architecture | Description | Implementation Source                      |
| :--- | :--- | :--- |:-------------------------------------------|
| **A1** | **MLP (Baseline)** | Standard FFN with GELU activation. | Vaswani et al. (2017)                      |
| **A2** | **KAN (B-Spline)** | KAN layers with cubic B-Splines and Sum aggregation. | Based on [efficient-kan](https://github.com/Blealtan/efficient-kan)                 |
| **A3** | **KAN (Mean)** | KAN layers with cubic B-Splines and **Mean** aggregation for stability. | Altarabichi (2024)                         |
| **A4** | **GR-KAN** | Group-Rational KAN using rational functions (Pade approx). | Yang & Wang (2024)  |
| **A5** | **MLP + B-Spline** | Standard MLP topology but with learnable B-Spline activations. | Control    |


## Model Sizes
The architectures are evaluated in three model-sizes with identical configuration as the ViT "Kolmogorov-Arnold Transformer" from Yang & Wang (2024). Parameter counts differ due to larger embedding vocab of the gpt-2 tokenizer.

| Model | Params. (ca.) | $d_{model}$ | $n_{heads}$ | $n_{layers}$ | $d_{ff}$ | $d_{ff,KAN}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Tiny | 15M | 192 | 3 | 12 | 768 | 77 |
| Small | 41M | 384 | 6 | 12 | 1536 | 154 |
| Base | 124M | 768 | 12 | 12 | 3072 | 307 |

## Key Results

### 1. Performance
* **Text Classification (AG News):** KAN-based architectures achieve **performance parity** with MLPs ($\pm 0.5\%$ accuracy).
* **Language Modeling (FineWeb):** KANs consistently **underperform** compared to MLPs (Perplexity increases of +7 to +28 points).

### 2. Efficiency
* **Training Speed:** KANs induce a training overhead of **1.5x - 2.0x**. GR-KAN approaches baseline speed (1.0x - 1.5x).
* **Parameters:** KANs achieve significantly higher sparsity ($\approx$ 90%) compared to MLPs ($\approx$ 40%), allowing for compression factors of up to 1.6x with minimal loss in classification tasks.

### 3. Interpretability
* Although KANs are sparser, the learned activation functions often degenerate to quasi-linear transformations. The "inherent interpretability" claim could not be fully confirmed for NLP tasks.

## Limitations
The results presented are subject to limitations that need to be kept in mind when drawing conclusions from them. The limited scope of a bachelor's thesis in terms of time and ressources resulted in majorly over-parametrized models (i. e. very small datasets used for training), with $TPP_{Tiny} \approx 5.62$ (Tokens-per-parameter $= D/N$) and $TPP_{Base} \approx 0.35$ respectively. Desirable are $TPP \approx 20$ for compute-efficient training (Bergsma et al. (2025)). The architectures could behave differently with a sufficiently large dataset and some architecture might be more robust than others under different training regimes; this investigation remains for future work and exceeds the scope of this project. Also, the results are based on a single dataset per task and further experiments are required to validate the results. The tasks are solely in the NLP domain and allow no generalization to other domains.


## Citation

If you use this code or results in your research, please cite the thesis:

```bibtex
@thesis{muesch2025kat,
  author       = {Yannick Müsch},
  title        = {Kolmogorov-Arnold-Transformer: Untersuchung hybrider KAN-Transformer-Architekturen im Hinblick auf Skalierung, Effizienz und Interpretierbarkeit},
  school       = {Hochschule Rhein-Waal},
  year         = {2025},
  month        = {December},
  type         = {Bachelor's Thesis}
}

```

## Acknowledgements

This work builds upon several open-source projects:

* [efficient-kan](https://github.com/Blealtan/efficient-kan) 


* [Kolmogorov-Arnold Transformer (KAT)](https://arxiv.org/abs/2409.10594) 



---

*Created by Yannick Müsch as part of the B.Sc. in E-Government at Rhine-Waal University of Applied Sciences.* 
