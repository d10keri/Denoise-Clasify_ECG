# Phase 1 & 2 Technical Report: ECG Arrhythmia Classification

# 1. Introduction
This report documents the implementation and experimental results of the ECG arrhythmia classification project, covering Phase 1 (Dataset Pipeline) and Phase 2 (Baseline Models). The goal of the project is to classify ECG beats into five categories based on the AAMI standard using the MIT-BIH Arrhythmia Database. The current scope includes data preprocessing, inter-patient dataset splitting, and the training of two baseline models: a Multi-Layer Perceptron (MLP) and a 1D Convolutional Neural Network (1D-CNN). The results presented herein reflect the actual implementation and evaluation metrics extracted from the project's output artifacts.

# 2. Dataset

## 2.1 MIT-BIH Arrhythmia Database
- **Sampling rate:** 360 Hz
- **Total recordings:** 48 records
- **Excluded records:** 4 records (102, 104, 107, 217)
- **Reason for exclusion:** Paced rhythms [REFERENCE NEEDED]

## 2.2 AAMI 5-Class Mapping
The mapping from original MIT-BIH annotations to the 5 AAMI classes is implemented as follows:

| Class ID | Class | Original MITDB annotations |
|----------|-------|----------------------------|
| 0        | N     | N, L, R, e, j              |
| 1        | S     | A, a, J, S                 |
| 2        | V     | V, E                       |
| 3        | F     | F                          |
| 4        | Q     | /, f, Q                    |

## 2.3 Data Split
The dataset was split following the De Chazal 2004 inter-patient protocol [REFERENCE NEEDED], ensuring that beats from the same patient (record) do not appear in both the training and testing sets to prevent data leakage.

| Split | Record IDs | Number of records | Number of beats |
|-------|------------|-------------------|-----------------|
| Train | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 205, 207, 208, 209, 215 | 18 | 41,116 |
| Validation | 203, 220, 223, 230 | 4 | 9,885 |
| Test | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | 22 | 49,691 |

*Note: The split is strictly at the record level. The training set is used for model optimization, the validation set is used for epoch-level model selection, and the test set is used strictly for the final evaluation.*

# 3. Preprocessing
The preprocessing pipeline transforms raw ECG signals into fixed-size, normalized beat windows.

| Step | Method | Parameters | Output |
|------|--------|------------|--------|
| Filtering | Butterworth IIR Bandpass | Order: 4, Lowcut: 0.5 Hz, Highcut: 60.0 Hz | Filtered signal (removes baseline wander & powerline noise) |
| Segmentation | R-peak centered window | Window size: 260 samples (99 before R, 161 after R) | Fixed-length beat segments |
| Normalization | Per-beat Z-score | - | Normalized segments (mean ≈ 0, std ≈ 1) |

*Note: While FIR filtering is mentioned as an option in the codebase, the actual pipeline used during the build phase was the 4th-order Butterworth filter.*

# 4. Segmentation and Labeling
Each valid R-peak annotation is used as a reference point. A window of 260 samples is extracted, spanning 99 samples before the R-peak and 161 samples after the R-peak. The beat is then normalized individually using Z-score normalization. The annotation label is mapped to the corresponding AAMI class.

```mermaid
graph LR
    A[Raw ECG Record] --> B[R-peak Detection]
    B --> C[260-sample Window Extraction]
    C --> D[Per-beat Z-score Normalization]
    D --> E[AAMI Class Mapping]
    E --> F[Labeled Normalized Beat]
```

# 5. Dataset Validation
Extensive tests were successfully run on the generated dataset to ensure data integrity and model compatibility.

## 5.1 Leakage Tests
| Test | Result |
|------|--------|
| `test_no_record_overlap_in_split_definition` | PASS |
| `test_excluded_records_not_in_any_split` | PASS |
| `test_total_records_count` | PASS |
| `test_npz_record_ids_no_overlap` | PASS |
| `test_npz_beat_indices_no_cross_patient_leak` | PASS |
| `test_expected_record_ids_per_split` | PASS |

## 5.2 Shape Tests
| Test | Result |
|------|--------|
| `test_X_shape_all_splits` | PASS |
| `test_y_shape_all_splits` | PASS |
| `test_dtype_all_splits` | PASS |
| `test_y_values_valid_classes` | PASS |
| `test_X_no_nan_no_inf` | PASS |
| `test_X_normalized_per_beat` | PASS |
| `test_meta_arrays_consistency` | PASS |
| `test_mlp_input_shape` | PASS |
| `test_cnn_input_shape` | PASS |
| `test_label_tensor_for_crossentropy` | PASS |

# 6. Dataset Statistics
*Note: [CHƯA CÓ DỮ LIỆU — cần bổ sung] Per-class counts in each split are not directly available in `dataset_info.json`, only total beats. The test set support size is known from the evaluation metrics.*

Test set class distribution (from test metrics):
| Split | N | S | V | F | Q | Total |
|-------|---|---|---|---|---|-------|
| Test | 44,239 | 1,837 | 3,220 | 388 | 7 | 49,691 |

Class weights calculated from the training set configuration:
- N: 0.22
- S: 10.61
- V: 2.87
- F: 20.61
- Q: 2055.80

This highlights a massive class imbalance, particularly for classes F and Q.

# 7. MLP Baseline

## 7.1 Architecture
The MLP baseline consists of a simple feedforward network:
```mermaid
graph LR
    A[Input: 260 features] --> B[Linear 128]
    B --> C[ReLU]
    C --> D[Linear 64]
    D --> E[ReLU]
    E --> F[Linear 5 output logits]
```

## 7.2 Training Configuration
| Parameter | Value |
|-----------|-------|
| Loss | CrossEntropyLoss [REF-05] |
| Optimizer | Adam |
| Learning Rate | 1.0e-3 |
| Batch Size | 64 |
| Epochs | 30 |
| Seed | 42 |
| Class Weighting | Enabled |

## 7.3 Training and Validation
The model was trained over 30 epochs with validation performed after each epoch. The checkpoint rule was to save the model that achieved the highest `val_macro_f1` score.
- **Best Epoch:** 1
- **Best Validation Macro F1:** 0.3054
- **Training Loss (Epoch 1):** 0.4780
- **Validation Loss (Epoch 1):** 0.8821

## 7.4 Test Results
| Metric | MLP |
|--------|-----|
| Accuracy | 0.6236 |
| Macro Precision | 0.2602 |
| Macro Recall | 0.3395 |
| Macro F1 | 0.2566 |

Per-Class Results:
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| N | 0.9479 | 0.6314 | 0.7580 | 44,239 |
| S | 0.0300 | 0.1274 | 0.0485 | 1,837 |
| V | 0.3151 | 0.8665 | 0.4622 | 3,220 |
| F | 0.0079 | 0.0722 | 0.0142 | 388 |
| Q | 0.0000 | 0.0000 | 0.0000 | 7 |

## 7.5 Confusion Matrix
Please refer to `results/mlp/confusion_matrix.png` for the detailed confusion matrix.

# 8. 1D-CNN Baseline

## 8.1 Architecture
The 1D-CNN architecture leverages local spatial relationships in the ECG signal:
- `Conv1d(1, 32, kernel_size=5, padding=2) -> ReLU -> MaxPool1d(2)`
- `Conv1d(32, 64, kernel_size=5, padding=2) -> ReLU -> MaxPool1d(2)`
- `Conv1d(64, 128, kernel_size=3, padding=1) -> ReLU -> AdaptiveAvgPool1d(1)`
- `Linear(128, 5)`

## 8.2 Training Configuration
*Same as MLP configuration.*

## 8.3 Training and Validation
- **Best Epoch:** 22
- **Best Validation Macro F1:** 0.3111
- **Training Loss (Epoch 22):** 0.1992
- **Validation Loss (Epoch 22):** 1.3823

## 8.4 Test Results
| Metric | 1D-CNN |
|--------|--------|
| Accuracy | 0.7302 |
| Macro Precision | 0.3440 |
| Macro Recall | 0.3250 |
| Macro F1 | 0.3282 |

## 8.5 Per-Class Results
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| N | 0.9444 | 0.7606 | 0.8426 | 44,239 |
| S | 0.0172 | 0.0642 | 0.0272 | 1,837 |
| V | 0.7563 | 0.7795 | 0.7677 | 3,220 |
| F | 0.0021 | 0.0206 | 0.0038 | 388 |
| Q | 0.0000 | 0.0000 | 0.0000 | 7 |

## 8.6 Confusion Matrix
Please refer to `results/cnn1d/confusion_matrix.png` for the detailed confusion matrix.

# 9. MLP vs 1D-CNN Comparison

| Metric                 | MLP Baseline | 1D-CNN Baseline |
| ---------------------- | -----------: | --------------: |
| Accuracy               | 0.6236       | 0.7302          |
| Macro Precision        | 0.2602       | 0.3440          |
| Macro Recall           | 0.3395       | 0.3250          |
| Macro F1               | 0.2566       | 0.3282          |
| Best Validation Metric | 0.3054       | 0.3111          |
| Best Epoch             | 1            | 22              |
| Parameters             | [CHƯA CÓ DỮ LIỆU — cần bổ sung] | [CHƯA CÓ DỮ LIỆU — cần bổ sung] |
| Training Time          | [CHƯA CÓ DỮ LIỆU — cần bổ sung] | [CHƯA CÓ DỮ LIỆU — cần bổ sung] |

**Analysis:**
- The 1D-CNN significantly outperforms the MLP in terms of overall Accuracy (0.7302 vs 0.6236) and Macro-F1 (0.3282 vs 0.2566).
- The CNN improved the classification of the normal class (N) and the ventricular ectopic beat class (V). For class V, the F1 score jumped from 0.4622 (MLP) to 0.7677 (CNN), indicating that spatial features extracted by CNNs are highly beneficial for detecting V beats.
- Both models utterly failed to classify the Q class (F1 = 0), which is expected due to the extreme lack of support samples (only 7 samples in the test set).
- Performance on the S and F classes remains very poor for both models, highlighting the challenge of severe class imbalance despite the use of class weighting.

# 10. Discussion

## 10.1 Dataset imbalance
The dataset exhibits extreme class imbalance (e.g., class Q has a weight of 2055.80 compared to class N's 0.22). The experimental results demonstrate that class weighting alone is insufficient for the models to learn robust features for minority classes like S, F, and Q.

## 10.2 MLP limitations
The MLP baseline peaked very early (Epoch 1) with low accuracy. This suggests that flattening the temporal structure of the ECG signal prevents the network from learning meaningful local morphological features, leading to poor generalization and overfitting (as observed by the diverging training and validation losses).

## 10.3 CNN advantages/disadvantages
The 1D-CNN handles the temporal sequence naturally through convolutions. It was able to extract more resilient features, resulting in much better performance for class V. However, it still struggled with the rarest classes, indicating that architectural improvements alone cannot completely overcome fundamental data scarcity.

## 10.4 Generalization
Both models demonstrated a significant gap between validation and test performance, and the validation loss often spiked during training. This indicates difficulty in generalizing across different patients (inter-patient split protocol).

## 10.5 Current limitations
- Extremely poor detection of minority classes.
- Only baseline architectures are implemented.
- Lack of data augmentation beyond class weighting.

# 11. Conclusion
Phase 1 successfully established a robust, leakage-free dataset pipeline based on the MIT-BIH Arrhythmia Database. Phase 2 provided two working baselines: an MLP and a 1D-CNN. The experiments clearly show that the 1D-CNN is superior to the MLP in classifying ECG beats, although both models struggle severely with class imbalance. These results serve as a foundation for implementing more advanced architectures and augmentation strategies in future phases.

# 12. Reproducibility
- **Python Version:** 3.9.17
- **PyTorch Version:** [CHƯA CÓ DỮ LIỆU — cần bổ sung]
- **Dataset Path:** `C:/d2l-en/data/ECG_Project_Data/mitdb`
- **Seed:** 42
- **Command Build Dataset:** `python scripts/build_dataset.py` [CHƯA CÓ DỮ LIỆU — cần bổ sung (verified script name)]
- **Command Train MLP:** `python train.py` [CHƯA CÓ DỮ LIỆU — cần bổ sung (exact args)]
- **Command Train CNN:** `python train_cnn1d.py`
- **Command Evaluate CNN:** `python evaluate_cnn1d.py`
- **Output Files:** `results/mlp/`, `results/cnn1d/`, `checkpoints/`

# 13. References
- [REF-01] MIT-BIH source [REFERENCE NEEDED]
- [REF-02] AAMI / AAMI mapping [REFERENCE NEEDED]
- [REF-03] De Chazal et al. [REFERENCE NEEDED]
- [REF-04] Adam [REFERENCE NEEDED]
- [REF-05] PyTorch CrossEntropyLoss
- [REF-06] scikit-learn metrics [REFERENCE NEEDED]

---

# Report Completion Checklist
- [x] Dataset description
- [x] Data split
- [x] Preprocessing
- [x] AAMI mapping
- [x] Leakage testing
- [x] Dataset statistics
- [x] MLP results
- [x] CNN results
- [ ] Literature citations
- [ ] Discussion refinement
- [ ] Final figures
