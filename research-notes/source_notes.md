# Source Notes

## S1 — Recent advances and clinical applications of deep learning in medical image analysis.

Title: Recent advances and clinical applications of deep learning in medical image analysis.
Year: 2022
Source Tool: pubmed_search
Source Type: biomedical_literature
URL: https://pubmed.ncbi.nlm.nih.gov/35472844/
Authors: Xuxin Chen, Ximin Wang, Ke Zhang, Kar-Ming Fung, Theresa C Thai, Kathleen Moore, Robert S Mannel, Hong Liu, Bin Zheng, Yuchen Qiu

Problem:
Improving deep learning models for medical image analysis, which is bottlenecked by lack of large-sized and well-annotated datasets.

Method:
Review and summary of recent studies focusing on unsupervised and semi-supervised deep learning methods applied to classification, segmentation, detection, and image registration in medical imaging.

Dataset:
Not specified

Supervision Setting:
Unsupervised and semi-supervised learning methods emphasized

Metrics:
Not specified

Validation Setting:
Not specified

Main Findings:
State-of-the-art unsupervised and semi-supervised deep learning methods have made significant progress in various medical image analysis tasks; major technical challenges and future research directions are discussed.

Author-Stated Limitations:
Lack of large-sized and well-annotated datasets remains a major bottleneck for further improvement.

Uncertainty:
Not specified

---

## S2 — Evidence-based uncertainty-aware semi-supervised medical image segmentation.

Title: Evidence-based uncertainty-aware semi-supervised medical image segmentation.
Year: 2024
Source Tool: pubmed_search
Source Type: biomedical_literature
URL: https://pubmed.ncbi.nlm.nih.gov/38277924/
Authors: Yingyu Chen, Ziyuan Yang, Chenyu Shen, Zhiwen Wang, Zhongzhou Zhang, Yang Qin, Xin Wei, Jingfeng Lu, Yan Liu, Yi Zhang

Problem:
Reducing false pseudo labels in semi-supervised medical image segmentation to improve performance.

Method:
EVidential Inference Learning (EVIL) integrates Dempster-Shafer Theory of Evidence into SSL, using consistency regularization and uncertainty quantification to discard unreliable pseudo labels.

Dataset:
Public datasets: ACDC, MM-WHS, MonuSeg

Supervision Setting:
Semi-supervised learning

Metrics:
Not specified

Validation Setting:
Benchmarking against state-of-the-art methods on public datasets

Main Findings:
EVIL performs competitively with state-of-the-art methods and provides precise uncertainty quantification within a single forward pass.

Author-Stated Limitations:
Not explicitly stated in abstract.

Uncertainty:
Uncertainty estimation is integrated and theoretically assured within the method.

---

## S3 — Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation

Title: Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation
Year: 2020
Source Tool: arxiv_search
Source Type: preprint
URL: https://arxiv.org/abs/2006.16806v1
Authors: Yingda Xia, Dong Yang, Zhiding Yu, Fengze Liu, Jinzheng Cai, Lequan Yu, Zhuotun Zhu, Daguang Xu, Alan Yuille, Holger Roth

Problem:
Leveraging unlabeled data for medical image segmentation and domain adaptation while addressing uncertainty in pseudo labels.

Method:
Uncertainty-aware multi-view co-training (UMCT) framework using multiple rotated/permuted 3D views and enforcing multi-view consistency with uncertainty estimation.

Dataset:
NIH pancreas segmentation dataset, multi-organ segmentation dataset, Medical Segmentation Decathlon Datasets

Supervision Setting:
Semi-supervised learning and unsupervised domain adaptation

Metrics:
Not specified

Validation Setting:
Experiments on semi-supervised segmentation and domain adaptation tasks

Main Findings:
UMCT achieves state-of-the-art performance on semi-supervised segmentation and effectively adapts to new domains, even without labeled source data.

Author-Stated Limitations:
Not specified

Uncertainty:
Uncertainty estimation is used to achieve accurate labeling in co-training.

---

## S4 — Shifting to Machine Supervision: Annotation-Efficient Semi and Self-Supervised Learning for Automatic Medical Image Segmentation and Classification

Title: Shifting to Machine Supervision: Annotation-Efficient Semi and Self-Supervised Learning for Automatic Medical Image Segmentation and Classification
Year: 2023
Source Tool: arxiv_search
Source Type: preprint
URL: https://arxiv.org/abs/2311.10319v6
Authors: Pranav Singh, Raviteja Chukkapalli, Shravan Chaudhari, Luoyao Chen, Mei Chen, Jinqian Pan, Craig Smuda, Jacopo Cirrone

Problem:
Reducing dependency on large annotated datasets for medical image segmentation and classification.

Method:
S4MI pipeline leveraging self-supervised and semi-supervised learning with auxiliary tasks that do not require labeling.

Dataset:
Three distinct medical imaging datasets (not named)

Supervision Setting:
Self-supervised and semi-supervised learning

Metrics:
Not specified

Validation Setting:
Benchmarking on classification and segmentation tasks across datasets

Main Findings:
Self-supervised learning outperforms supervised methods in classification; semi-supervised learning outperforms fully supervised segmentation using 50% fewer labels.

Author-Stated Limitations:
Not specified

Uncertainty:
Not specified

---

## S5 — Continual learning approach to domain shifts in medical imaging

Title: Continual learning approach to domain shifts in medical imaging
Year: 2021
Source Tool: unknown
Source Type: biomedical_literature
URL: Not specified
Authors: Matthias Perkonigg, Johannes Hofmanninger, Christian J Herold, James A Brink, Oleg Pianykh, Helmut Prosch, Georg Langs

Problem:
Domain shifts in medical imaging due to evolving acquisition technology and protocols causing model degradation.

Method:
Continual learning with dynamic memory and pseudo-domain detection to mitigate catastrophic forgetting and adapt to new domains.

Dataset:
Cardiac MRI segmentation and lung nodule detection CT datasets

Supervision Setting:
Not specified

Metrics:
Not specified

Validation Setting:
Evaluation on two tasks: cardiac segmentation and lung nodule detection

Main Findings:
Method consistently outperforms baselines by adapting to domain shifts and mitigating forgetting.

Author-Stated Limitations:
Not specified

Uncertainty:
Not specified

---

## S6 — Boosting knowledge diversity, accuracy, and stability via tri-enhanced distillation for domain continual medical image segmentation.

Title: Boosting knowledge diversity, accuracy, and stability via tri-enhanced distillation for domain continual medical image segmentation.
Year: 2024
Source Tool: pubmed_search
Source Type: biomedical_literature
URL: Not specified
Authors: Zhanshi Zhu, Xinghua Ma, Wei Wang, Suyu Dong, Kuanquan Wang, Lianming Wu, Gongning Luo, Guohua Wang, Shuo Li

Problem:
Catastrophic forgetting in domain continual medical image segmentation.

Method:
Tri-enhanced distillation framework with stochastic knowledge augmentation, adaptive knowledge transfer, and global uncertainty-guided fusion.

Dataset:
Not specified

Supervision Setting:
Not specified

Metrics:
Not specified

Validation Setting:
Experimental validation against state-of-the-art methods

Main Findings:
Proposed method outperforms state-of-the-art and establishes a robust benchmark for domain continual segmentation.

Author-Stated Limitations:
Not specified

Uncertainty:
Global uncertainty-guided fusion reduces bias and promotes stable knowledge fusion.

---

## S7 — Incremental Learning Meets Transfer Learning: Application to Multi-site Prostate MRI Segmentation.

Title: Incremental Learning Meets Transfer Learning: Application to Multi-site Prostate MRI Segmentation.
Year: 2022
Source Tool: pubmed_search
Source Type: biomedical_literature
URL: Not specified
Authors: Chenyu You, Jinlin Xiang, Kun Su, Xiaoran Zhang, Siyuan Dong, John Onofrey, Lawrence Staib, James S Duncan

Problem:
Sequential training on multi-site medical image datasets to improve generalization and transfer without access to all data simultaneously.

Method:
Incremental-transfer learning (ITL) framework with site-agnostic encoder, multiple decoder heads, and site-level incremental loss to alleviate catastrophic forgetting.

Dataset:
Five benchmark prostate MRI datasets

Supervision Setting:
Not specified

Metrics:
Not specified

Validation Setting:
Experiments on multi-site prostate MRI segmentation

Main Findings:
ITL improves performance and alleviates catastrophic forgetting in incremental learning.

Author-Stated Limitations:
Not specified

Uncertainty:
Not specified

---

## S8 — Segment Anything Model for Medical Image Analysis: an Experimental Study

Title: Segment Anything Model for Medical Image Analysis: an Experimental Study
Year: 2023
Source Tool: arxiv_search
Source Type: preprint
URL: https://arxiv.org/abs/2304.07839
Authors: Maciej A. Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, Nicholas Konz, Yixin Zhang

Problem:
Evaluating the performance of the Segment Anything Model (SAM) on medical image segmentation tasks.

Method:
Extensive evaluation of SAM on 19 medical imaging datasets from various modalities and anatomies using different prompt types.

Dataset:
19 medical imaging datasets (various modalities and anatomies)

Supervision Setting:
Zero-shot segmentation

Metrics:
IoU (Intersection over Union)

Validation Setting:
Performance evaluation with single and multiple prompts, comparison with other interactive segmentation methods

Main Findings:
SAM shows impressive zero-shot segmentation on some datasets (IoU up to 0.8650) but poor on others (IoU as low as 0.1135); better with box prompts than point prompts; iterative prompting improves performance slightly.

Author-Stated Limitations:
Performance varies widely depending on dataset and task; moderate to poor performance on complex or ambiguous segmentation tasks.

Uncertainty:
Not specified

---

