# ActCIM-Robust: Sensitivity-Guided Randomized Nonlinearity-Aware Training for CIM Neural Networks

## Abstract

Computing-in-Memory (CIM) accelerators suffer from inherent nonlinear activation distortions that severely degrade the accuracy of deployed deep neural networks. We systematically analyze the impact of a parameterized cubic nonlinearity $f_\alpha(x)=m\cdot(\alpha\cdot(x/m)^3+(1-\alpha)\cdot(x/m))$ on ResNet-18 inference and propose three Nonlinearity-Aware Training (NAT) methods: Fixed-NAT (training with a fixed nonlinearity strength), Random-NAT (randomly sampled nonlinearity per forward pass), and SGR-NAT (sensitivity-guided dual-branch training with curriculum learning). Our key finding is a pronounced asymmetry: positive nonlinearity ($\alpha>0$, compressive) degrades accuracy far more severely than negative nonlinearity ($\alpha<0$, expansive). At $\alpha=+0.8$, the clean baseline drops from 94.23% to 81.25% accuracy, while $\alpha=-0.8$ retains 93.66%. Fixed-NAT with training $\alpha=+0.4$ achieves the best robustness: worst-case accuracy improves from 81.25% to 91.79% (+10.54 pp), AURC improves from 0.9283 to 0.9374, and the positive-negative asymmetry gap nearly vanishes (from -0.0314 to 0.0007). Random-NAT and SGR-NAT provide only marginal improvements (worst-case 81.30% and 82.06% respectively) in the fine-tuning regime. Our results demonstrate that targeted nonlinearity-aware fine-tuning is a practical and efficient path to CIM deployment robustness on consumer-grade hardware.

**Keywords**: computing-in-memory, nonlinear distortion, robustness, sensitivity-guided training, calibration

---

## 1. Introduction

Deep neural network inference on Computing-in-Memory (CIM) hardware promises orders-of-magnitude energy efficiency improvements by performing multiply-accumulate operations directly within memory arrays [1, 2]. However, analog computation in CIM chips introduces systematic nonlinear distortions that deviate from the ideal linear matrix-vector product. These distortions accumulate across layers and can catastrophically degrade model accuracy at deployment.

Prior work has primarily focused on hardware-aware training that co-optimizes for specific device characteristics [3, 4, 5]. However, systematic analysis of CIM nonlinearity robustness---particularly the direction-dependent asymmetry of distortion effects---remains underexplored. Furthermore, the question of whether lightweight fine-tuning can recover robustness for already-trained models has not been thoroughly investigated.

In this work, we make the following contributions:

1. We conduct a systematic sensitivity analysis of a parameterized cubic CIM nonlinearity $f_\alpha(x)$ across 11 nonlinearity strengths and 21 network layers, revealing a pronounced *positive-negative asymmetry* where compressive nonlinearity ($\alpha>0$) is approximately 22x more damaging than expansive nonlinearity of equal magnitude.

2. We propose and benchmark three Nonlinearity-Aware Training (NAT) methods---Fixed-NAT, Random-NAT, and SGR-NAT---on the task of recovering CIM robustness via fine-tuning.

3. We demonstrate that Fixed-NAT with a carefully chosen training $\alpha$ yields a +10.54 percentage point improvement in worst-case accuracy with only 5 minutes of fine-tuning on consumer-grade hardware, while preserving near-baseline clean accuracy.

4. We provide extensive calibration analysis showing that nonlinearity-induced ECE degradation can be partially mitigated through Fixed-NAT.

---

## 2. Related Work

### 2.1 Computing-in-Memory

CIM accelerators based on resistive RAM (RRAM), phase-change memory (PCM), and SRAM have demonstrated energy-efficient matrix-vector multiplication [1, 2]. The key principle is that Ohm's law ($I=VG$) naturally computes dot products when voltages are applied across programmable conductances. However, transistor nonlinearity, wire resistance, and device variability introduce systematic output errors that can be modeled as polynomial distortions [3, 5]. REFERENCES_TO_VERIFY

### 2.2 Hardware-Aware Training

Several approaches incorporate hardware non-idealities into the training loop. Joshi et al. [4] proposed modeling PCM conductance drift during training. Yao et al. [3] demonstrated fully hardware-implemented training with memristor crossbars. Our approach differs in that we focus on *post-hoc robustness recovery* via fine-tuning rather than training from scratch with hardware models. REFERENCES_TO_VERIFY

### 2.3 Model Calibration

Guo et al. [7] showed that modern neural networks are poorly calibrated, with ECE increasing as a function of model capacity. We extend this analysis to CIM-induced miscalibration, showing that nonlinear distortions can inflate ECE from 0.033 to 0.560---a 17x increase that Fixed-NAT partially recovers.

### 2.4 Robustness to Perturbations

Our work connects to the broader literature on neural network robustness [9], but focuses specifically on hardware-induced rather than adversarial perturbations. The asymmetry we observe---compressive distortion being more damaging than expansive---has parallels in the activation quantization literature, where clipping (compression) causes more information loss than expansion. REFERENCES_TO_VERIFY

---

## 3. Problem Formulation

### 3.1 CIM Nonlinearity Model

We model the CIM activation nonlinearity as a parameterized cubic function:

$$f_\alpha(x) = m \cdot \left[\alpha \cdot \left(\frac{x}{m}\right)^3 + (1-\alpha) \cdot \left(\frac{x}{m}\right)\right]$$

where $x$ is the layer output, $m=10.0$ is a scaling factor, and $\alpha \in [-1, 1]$ controls the nonlinearity strength and direction:
- $\alpha = 0$: $f_0(x) = x$ (ideal linear)
- $\alpha > 0$: compressive nonlinearity---large activations are attenuated
- $\alpha < 0$: expansive nonlinearity---large activations are amplified

This cubic parameterization captures the first-order symmetric nonlinearity of analog CIM crossbars while remaining analytically tractable. The function is odd-symmetric around zero, preserving the zero-crossing property of standard activation functions.

### 3.2 Problem Statement

Given a pre-trained model $\mathcal{M}$ with parameters $\theta$, we aim to produce a robust model $\mathcal{M}_{robust}$ such that:

$$\max_{\alpha \in \mathcal{A}} \mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathbf{1}\{\mathcal{M}_{robust}(x; f_\alpha) \neq y\}]$$

is minimized, where $f_\alpha$ is applied after every linear layer in $\mathcal{M}$. The set $\mathcal{A} = \{-0.8, -0.6, \dots, 0.8\}$ covers a realistic range of CIM nonlinearity strengths.

### 3.3 Evaluation Metrics

- **Clean Accuracy ($\alpha=0$)**: Standard test accuracy without nonlinearity
- **Worst-Case Accuracy**: Minimum test accuracy across $\alpha \in \mathcal{A}$
- **AURC (Area Under the Response Curve)**: Integrated accuracy over $\alpha$, defined as the mean of per-alpha accuracies
- **AURC$^+$ / AURC$^-$**: AURC restricted to positive/negative $\alpha$ values
- **Asymmetry Gap**: AURC$^+$ - AURC$^-$, measuring directional robustness asymmetry
- **ECE (Expected Calibration Error)**: Binned calibration error with 15 equal-width bins [8]

---

## 4. Sensitivity Analysis Methods

### 4.1 Alpha Sweep

We evaluate model accuracy across 11 nonlinearity strengths ($\alpha \in \{-0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8\}$) by injecting $f_\alpha$ after all 21 layers of ResNet-18. Each evaluation uses all 10,000 CIFAR-10 test samples.

**Key Findings (Clean Baseline):**

| $\alpha$ | Accuracy | ECE | Mean Confidence |
|----------|----------|-----|-----------------|
| -0.8 | 0.9366 | 0.0558 | 0.9924 |
| -0.4 | 0.9403 | 0.0480 | 0.9882 |
| 0.0 | 0.9423 | 0.0326 | 0.9746 |
| 0.4 | 0.9359 | 0.1235 | 0.8124 |
| 0.8 | 0.8125 | 0.5602 | 0.2523 |

The asymmetry is stark: $\alpha=+0.8$ causes a 12.98 pp accuracy drop, while $\alpha=-0.8$ causes only a 0.57 pp drop---a 22.8x difference. ECE shows a similar pattern, ballooning from 0.0326 at $\alpha=0$ to 0.5602 at $\alpha=+0.8$.

### 4.2 Layer Sensitivity Analysis

We isolate each of ResNet-18's 21 layers and apply $\alpha=\pm 0.4$ to that layer alone while keeping others linear.

**Key Findings:**

The maximum single-layer accuracy drop is only 0.78% (conv1 layers at $\alpha=-0.4$). Most layers show negligible or even *beneficial* effects from single-layer perturbation. The fc layer shows zero accuracy change when perturbed in isolation. This demonstrates that CIM degradation is fundamentally a *cumulative error problem*---individual layer perturbations are benign, but their cascade through a deep network is destructive.

### 4.3 Error Accumulation Analysis

We measure activation statistics at each layer output under full-network nonlinearity ($\alpha=\pm 0.4$):

| Layer | $\alpha=+0.4$ Rel. L2 | $\alpha=-0.4$ Rel. L2 | Key Observation |
|-------|----------------------|----------------------|-----------------|
| conv1--layer3 | 0.000 | 0.000 | No detectable error in early/mid layers |
| fc | 0.605 | 0.978 | Error concentrates at the final classifier |
| layer4.1.conv2 | 0.588 | 0.843 | Deepest conv shows significant accumulation |

At the fc layer under $\alpha=-0.4$: standard deviation nearly doubles (3.63→7.16), 3.4% of activation signs flip, and 53.4% of activations saturate. Under $\alpha=+0.4$: standard deviation compresses to 41% of original (3.63→1.49), 10.2% sign flip ratio---the higher sign flip rate explains why compressive nonlinearity is more damaging to classification.

---

## 5. Nonlinearity-Aware Training Methods

### 5.1 Fixed-NAT

**Training**: Fine-tune from pre-trained checkpoint with a *fixed* $\alpha=+0.4$ applied to all layers.

**Rationale**: We choose $\alpha=+0.4$ because it is the positive $\alpha$ where performance first meaningfully degrades (93.59% vs 94.23%) while still being within the range where training remains stable. Training at the exact deployment $\alpha$ allows the model to adapt its internal representations to the specific distortion pattern.

**Configuration**: SGD(lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov), cosine annealing with 1-epoch warmup, batch size 128, 10 epochs, early stopping patience 10.

### 5.2 Random-NAT

**Training**: Fine-tune with $\alpha \sim \mathcal{U}(-0.8, 0.8)$ randomly sampled per forward pass, applied uniformly to all layers.

**Rationale**: Random exposure to diverse nonlinearity during training should produce a model robust across the entire $\alpha$ range. This resembles data augmentation applied at the activation level.

**Configuration**: Identical hyperparameters to Fixed-NAT, but $\alpha$ is resampled for each batch.

### 5.3 SGR-NAT (Sensitivity-Guided Randomized NAT)

**Training**: A dual-branch training scheme combining:
1. **Clean branch**: Standard forward pass without nonlinearity
2. **Nonlinear branch**: Forward pass with nonlinearity applied to a *subset* of layers selected based on pre-computed sensitivity rankings, with curriculum learning that gradually increases injection scope

**Loss**: $\mathcal{L} = w_{clean} \cdot \mathcal{L}_{CE}^{clean} + w_{nonlinear} \cdot \mathcal{L}_{CE}^{nonlinear} + \lambda_{cons} \cdot \mathcal{L}_{KL}(p^{clean} \| p^{nonlinear})$

The consistency term $\mathcal{L}_{KL}$ encourages the nonlinear branch to match the clean branch's predictions, implementing a form of Jacobian regularization.

**Configuration**: $\lambda_{cons}=0.5$, $w_{clean}=0.25$, $w_{nonlinear}=0.75$, temperature=2.0, detach clean logits, ~4.76% layer injection rate, $\alpha \sim \mathcal{U}(-1, 1)$, curriculum over epochs. Same optimizer and scheduler as other methods.

---

## 6. Experiments

### 6.1 Setup

- **Hardware**: NVIDIA RTX 4060 (8GB), Intel i7-10700, 32GB RAM, Windows 11, Python 3.12.5, PyTorch 2.5.1+cu121
- **Dataset**: CIFAR-10 (50k train / 10k test), standard augmentations
- **Model**: ResNet-18 adapted for CIFAR-10 (11,183,582 parameters, 21 injectable layers)
- **Baseline training**: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov), cosine annealing with 3-epoch warmup, batch size 128, 50 epochs, AMP enabled
- **Fine-tuning**: SGD(lr=0.01), cosine annealing with 1-epoch warmup, 10 epochs, early stopping patience 10
- **Seeds**: 42, 3407, 2026 (Clean and Fixed-NAT); seed 42 only (Random-NAT and SGR-NAT)

### 6.2 Baseline Training

Clean baseline training achieves:
- Seed 42: best val_acc 94.84% at epoch 48, train time 23m55s
- Seed 3407: best val_acc 94.80% at epoch 47, train time 24m35s
- Seed 2026: best val_acc 94.98% at epoch 46, train time 23m55s

---

## 7. Results

### 7.1 Main Comparison Table

| Method | $\alpha=0$ Acc | Worst Acc ($\alpha=+0.8$) | AURC | AURC$^+$ | Asymmetry Gap | Mean Pert. Acc |
|--------|---------------|---------------------------|------|----------|---------------|----------------|
| Clean | 0.9423 | 0.8125 | 0.9283 | 0.9128 | -0.0314 | 0.9243 |
| Random-NAT | 0.9425 | 0.8130 | 0.9281 | 0.9116 | -0.0331 | 0.9242 |
| SGR-NAT | **0.9428** | 0.8206 | 0.9290 | 0.9141 | -0.0300 | 0.9252 |
| Fixed-NAT | 0.9402 | **0.9179** | **0.9374** | **0.9382** | **0.0007** | **0.9365** |

### 7.2 Detailed Per-Alpha Accuracy

| $\alpha$ | Clean | Random-NAT | SGR-NAT | Fixed-NAT |
|----------|-------|------------|---------|-----------|
| -0.8 | 0.9366 | 0.9384 | 0.9377 | 0.9323 |
| -0.6 | 0.9391 | 0.9397 | 0.9392 | 0.9342 |
| -0.4 | 0.9403 | 0.9405 | 0.9399 | 0.9359 |
| -0.2 | 0.9418 | 0.9421 | 0.9418 | 0.9386 |
| -0.1 | 0.9422 | 0.9428 | 0.9427 | 0.9396 |
| 0.0 | 0.9423 | 0.9425 | 0.9428 | 0.9402 |
| 0.1 | 0.9423 | 0.9412 | 0.9415 | 0.9413 |
| 0.2 | 0.9411 | 0.9402 | 0.9408 | 0.9415 |
| 0.4 | 0.9359 | 0.9346 | 0.9356 | **0.9427** |
| 0.6 | 0.9113 | 0.9091 | 0.9126 | **0.9407** |
| 0.8 | 0.8125 | 0.8130 | 0.8206 | **0.9179** |

### 7.3 Per-Alpha ECE

| $\alpha$ | Clean | Random-NAT | SGR-NAT | Fixed-NAT |
|----------|-------|------------|---------|-----------|
| 0.0 | 0.0326 | 0.0301 | 0.0329 | 0.0496 |
| 0.4 | 0.1235 | 0.1422 | 0.1022 | 0.0283 |
| 0.8 | 0.5602 | 0.5610 | 0.5513 | 0.4359 |

### 7.4 Multi-Seed Results

| Seed | Clean Val Acc | Clean Train Time | Fixed-NAT Val Acc | Fixed-NAT Train Time |
|------|--------------|------------------|-------------------|----------------------|
| 42 | 94.84% | 23m55s | 94.90% | 5m01s |
| 3407 | 94.80% | 24m35s | 94.84% | 4m53s |
| 2026 | 94.98% | 23m55s | 94.98% | 4m53s |

### 7.5 Key Observations

1. **Fixed-NAT dominates all metrics** except clean accuracy, where it sacrifices only 0.21 pp (94.23%→94.02%)
2. **Random-NAT provides zero benefit** over the clean baseline---its AURC is slightly *worse* (0.9281 vs 0.9283)
3. **SGR-NAT achieves the best clean accuracy** (94.28%) but its robustness gains are modest: +0.81 pp worst-case and +0.0006 AURC
4. **Asymmetry is the key challenge**: Fixed-NAT reduces the asymmetry gap from -0.0314 to 0.0007, suggesting it has "learned to compensate" for the directional distortion
5. **Fixed-NAT's optimal performance is at the training $\alpha$** ($\alpha=+0.4$, accuracy 94.27%), confirming that targeted training is more effective than randomized exposure

---

## 8. Discussion

### 8.1 Why Fixed-NAT Outperforms Randomized Approaches

The superiority of Fixed-NAT over Random-NAT and SGR-NAT in the fine-tuning regime can be explained by several factors:

1. **Focused gradient signal**: Training at a single $\alpha$ provides a consistent gradient direction, enabling more efficient optimization. Random $\alpha$ creates conflicting gradient signals that cancel out in expectation, especially with the small learning rate used in fine-tuning.

2. **Training-testing alignment**: Fixed-NAT trains at $\alpha=+0.4$, which is within the positive range where most performance degradation occurs. The model learns representations specifically adapted to compressive distortion, which is the dominant failure mode.

3. **Fine-tuning window constraint**: With only 10 epochs at lr=0.01, the model has limited capacity to adapt to the broad range of $\alpha$ values that Random-NAT requires. A full training-from-scratch with Random-NAT might yield different results---an experiment we were unable to run due to resource constraints.

### 8.2 The Positive-Negative Asymmetry

The strong asymmetry between positive and negative $\alpha$ has both theoretical and practical implications:

- **Theoretical**: Compressive nonlinearity ($\alpha>0$) reduces the dynamic range of activations, effectively *removing information* from the signal. Expansive nonlinearity ($\alpha<0$) amplifies existing differences, which is less destructive to classification boundaries.

- **Practical**: CIM chip designers should characterize whether their devices tend toward compressive or expansive distortion. If compressive, robustness-aware training (like Fixed-NAT) is essential. If expansive, the problem may be less severe.

### 8.3 Calibration Implications

Nonlinearity not only degrades accuracy but severely damages confidence calibration. The 15-bin reliability diagram for Clean at $\alpha=+0.8$ shows:
- 38.1% of test samples fall into the lowest two confidence bins (confidence < 0.20), yet these bins achieve ~59.5% accuracy---the model is *underconfident* about correct predictions
- No sample reaches the top two confidence bins (confidence > 0.87)---the model completely loses the ability to make high-confidence predictions
- Fixed-NAT reduces ECE at $\alpha=+0.8$ from 0.5602 to 0.4359, a meaningful but incomplete recovery

---

## 9. Limitations

1. **Single architecture**: Results are validated only on ResNet-18. Behavior on other architectures (VGG, MobileNet, Vision Transformers) is unknown.

2. **Single dataset**: CIFAR-10 is relatively simple. Results on larger-scale datasets (ImageNet, CIFAR-100) may differ.

3. **Simplified nonlinearity model**: The cubic polynomial $f_\alpha(x)$ is a first-order approximation of actual CIM device characteristics. Real devices exhibit more complex, asymmetric, and device-dependent nonlinearities.

4. **Limited statistical power**: Only 3 seeds for Clean and Fixed-NAT, 1 seed for Random-NAT and SGR-NAT. Variance estimates are unreliable.

5. **Narrow fine-tuning scope**: All NAT methods used the same fine-tuning protocol (10 epochs, lr=0.01). Different hyperparameters or training-from-scratch might yield different relative rankings of the methods.

6. **Static sensitivity**: SGR-NAT's sensitivity ranking is pre-computed at $\alpha=\pm 0.4$ and fixed. Dynamic sensitivity estimation during training could potentially improve its performance.

7. **No hardware validation**: All experiments are simulation-based. Actual CIM hardware deployment may reveal additional effects not captured by the cubic model.

---

## 10. Conclusion

We have presented a systematic study of CIM nonlinearity robustness and three fine-tuning methods for recovering deployment accuracy. Our key findings are:

1. **CIM nonlinearity exhibits strong asymmetry**: Compressive distortion ($\alpha>0$) is approximately 22x more harmful than expansive distortion ($\alpha<0$) at equivalent magnitude.

2. **Fixed-NAT is highly effective**: With only ~5 minutes of additional fine-tuning, worst-case accuracy improves from 81.25% to 91.79% (+10.54 pp), and the asymmetry gap is nearly eliminated.

3. **Randomized training provides marginal benefits**: In the fine-tuning regime with small learning rates, Random-NAT and SGR-NAT do not offer meaningful advantages over the clean baseline.

4. **Calibration degradation is severe but partially recoverable**: ECE inflates from 0.033 to 0.560 under strong nonlinearity; Fixed-NAT brings this down to 0.436.

For practitioners deploying models on CIM hardware, we recommend Fixed-NAT with a fine-tuning $\alpha$ chosen to match the expected worst-case device nonlinearity strength. This approach requires minimal computational overhead (5 GPU-minutes) and preserves near-baseline standard accuracy.

---

## References

[1] Sebastian, A., Le Gallo, M., Khaddam-Aljameh, R., & Eleftheriou, E. (2020). Memory devices and applications for in-memory computing. *Nature Nanotechnology*, 15(7), 529-544. REFERENCES_TO_VERIFY

[2] Ielmini, D., & Wong, H. S. P. (2018). In-memory computing with resistive switching devices. *Nature Electronics*, 1(6), 333-343. REFERENCES_TO_VERIFY

[3] Yao, P., Wu, H., Gao, B., Tang, J., Zhang, Q., Zhang, W., Yang, J. J., & Qian, H. (2020). Fully hardware-implemented memristor convolutional neural network. *Nature*, 577(7792), 641-646. REFERENCES_TO_VERIFY

[4] Joshi, V., Le Gallo, M., Haefeli, S., Boybat, I., Nandakumar, S. R., Piveteau, C., Dazzi, M., Rajendran, B., Sebastian, A., & Eleftheriou, E. (2020). Accurate deep neural network inference using computational phase-change memory. *Nature Communications*, 11(1), 2473. REFERENCES_TO_VERIFY

[5] Ambrogio, S., Narayanan, P., Tsai, H., Shelby, R. M., Boybat, I., di Nolfo, C., Sidler, S., Giordano, M., Bodini, M., Farinha, N. C. P., Killeen, B., Cheng, C., Jaoudi, Y., & Burr, G. W. (2018). Equivalent-accuracy accelerated neural-network training using analogue memory. *Nature*, 558(7708), 60-67. REFERENCES_TO_VERIFY

[6] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.

[7] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 1321-1330.

[8] Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 29(1).

[9] Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *International Conference on Learning Representations (ICLR)*.

[10] Zhang, H., Yu, Y., Jiao, J., Xing, E. P., El Ghaoui, L., & Jordan, M. I. (2019). Theoretically principled trade-off between robustness and accuracy. *Proceedings of the 36th International Conference on Machine Learning (ICML)*. REFERENCES_TO_VERIFY
