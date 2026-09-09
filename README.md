<h1>LabCora: Label Correlation-Aware Semi-Supervised Multi-Label Medical Image Classification</h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cae7d1ed-4d8d-4555-ae95-5c71b8ea37e4" width="800">
</p>

<p style="font-size: 12px;">
<strong>Abstract:</strong> Semi-supervised learning has shown great potential in reducing the annotation burden for multi-label medical image classification. However, most existing methods suffer from confirmation bias due to unreliable pseudo-labels, while the distribution mismatch between labeled and unlabeled data frequently limits the model's generalization capability. In this paper, we propose LabCora, a novel framework for semi-supervised multi-label medical image classification. Specifically, we propose a Label Correlation-guided Pseudo-Labeling module, which mitigates confirmation bias by explicitly modeling label correlations and using them to correct unreliable pseudo-labels. In addition, a Cross-Domain Adversarial module is introduced to alleviate the distribution mismatch via feature-level adversarial training. By integrating these modules within a Co-training paradigm, label correlation-guided pseudo-label refinement and cross-domain adversarial alignment work together to enhance both pseudo-label reliability and feature generalization. Extensive experimental results on Chest X-Ray14, CheXpert, and ODIR-5K demonstrate that LabCora consistently outperforms state-of-the-art methods, particularly in low-label regimes.
</p>
