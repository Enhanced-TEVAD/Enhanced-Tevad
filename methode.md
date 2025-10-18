# Weakly Supervised Video Anomaly Detection

## Method Overview
The proposed method is designed for **weakly supervised anomaly detection in videos**, where only **video-level labels** (normal or abnormal) are available. The model must learn to identify abnormal snippets within abnormal videos.

## Feature Extraction


- **Text Features**: Generated using **SimCSE**, which creates sentence embeddings from video captions.
- **Visual Features**: Extracted using **I3D with ResNet-50** as a backbone, with **multi-crop augmentation**.
- **Multi-scale Temporal Learning (MTN)**: Captures both short-term and long-term dependencies using **pyramid dilated convolutions (PDC)** and **non-local blocks (NLB)**.

## Feature Magnitude Computation
Research suggests that **abnormal snippets** tend to have **higher feature magnitudes** than normal ones. The feature magnitude for a video *v* is computed as:

# Method and Mathematical Formulation

## **Feature Magnitude Calculation**
The feature magnitude of a video \( v \) is computed as:

$$
f_{FM}(v;k) = \frac{1}{k} \sum_{i \in \text{topK}(v;k)} ||X_i||_2
$$

## **Training Loss**
The total training loss for normal and abnormal videos is:

$$
L_{fm} =
\begin{cases} 
c - f_{FM}(v_j; k), & \text{if } y_j = 1 \\
f_{FM}(v_j; k), & \text{if } y_j = 0
\end{cases}
$$

## **Anomaly Score Calculation**
The anomaly score is computed as:

$$
f_s(v; k) = \frac{1}{k} \sum_{i \in \text{topK}(v;k)} f_{\text{pred}}(X_i; \delta)
$$

## **Binary Classification Loss**
We train a binary classifier with the Binary Cross Entropy (BCE) loss:

$$
L_{bce} = - \frac{1}{|V|} \sum_{j=1}^{|V|} \left( y_j \log f_s(v_j; k) + (1 - y_j) \log (1 - f_s(v_j; k)) \right)
$$

## **Total Loss Function**
The final loss function is:

$$
L = \alpha L_{fm} + L_{bce}
$$

```

where `α` is a hyperparameter balancing the two loss terms.

## Key Takeaways
- **Only video-level labels (normal/abnormal) are available**, so the model must infer snippet-level anomalies.
- **Feature magnitude is used to differentiate normal vs. abnormal snippets** (higher magnitudes correspond to abnormal snippets).
- The training process optimizes two objectives:
  - `L_fm`: Maximizes the feature magnitude gap between normal and abnormal videos.
  - `L_bce`: Trains a classifier to predict anomaly scores.
- **The final loss function combines both losses**, allowing the model to distinguish abnormal snippets inside abnormal videos without snippet-level labels.
