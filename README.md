# Facial Emotion Classification

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Project-lightgrey)

An independent deep learning research project benchmarking multiple CNN architectures — a from-scratch baseline and four transfer-learning backbones — for classifying facial expressions into 7 emotion categories on the **FER2013** dataset. The project compares feature-extraction vs. fine-tuning strategies, evaluates the trade-offs between them, and includes a critical review of evaluation-metric choice.

**Best result: 58.3% test accuracy** (7-class problem, chance level ≈ 14%), achieved with a fine-tuned ResNet50V2 at 224x224 resolution.

## Skills Demonstrated

- **Transfer learning & fine-tuning**: layer-freezing strategies across VGG16, ResNet50, ResNet50V2, and InceptionResNetV2 (feature extraction vs. partial unfreezing)
- **CNN architecture design**: built LeNet-5 and a custom multi-block CNN from scratch for baseline comparison
- **Data pipeline engineering**: `ImageDataGenerator` pipelines with augmentation (rotation, shift, zoom, flips) to combat overfitting and class imbalance
- **Model evaluation**: confusion matrices, precision/recall/F1, classification reports, and diagnosing a mismatched evaluation metric (see Key Learnings)
- **Training diagnostics**: `EarlyStopping`, `ReduceLROnPlateau`, and checkpointing; reading train/val curves to identify overfitting

## Dataset

[FER2013](https://www.kaggle.com/msambare/fer2013) — 48x48 grayscale, pre-cropped and centered face images, labeled with one of 7 emotions.

| Class | Train | Test |
|---|---|---|
| Happy | 7,215 | 1,774 |
| Neutral | 4,965 | 1,233 |
| Sad | 4,830 | 1,247 |
| Fear | 4,097 | 1,024 |
| Angry | 3,995 | 958 |
| Surprise | 3,171 | 831 |
| Disgust | 436 | 111 |
| **Total** | **28,709** | **7,178** |

The class distribution is heavily imbalanced — `disgust` has roughly 6-18x fewer samples than the other classes, which consistently shows up as the weakest-performing class across every model below.

## Approach

Seven notebooks, each exploring a different architecture or training strategy:

| Notebook | Model | Input | Strategy |
|---|---|---|---|
| `fer-cnn-lenet-5.ipynb` | LeNet-5 (from scratch) | 48x48 grayscale | Full training, no pretrained weights |
| `convnet-fusion.ipynb` | Custom 3-block CNN | 48x48 grayscale | Full training, no pretrained weights |
| `RESNET-50V2.ipynb` | ResNet50V2 | 48x48 RGB | Feature extraction (ImageNet backbone frozen) |
| `InceptionResnet.ipynb` | InceptionResNetV2 | 150x150 RGB | Feature extraction on a 7,000-image subset |
| `emotion-recognition-with-resnet50.ipynb` | ResNet50 | 48x48 RGB | Fine-tuning (last 4 layers unfrozen) + custom dense head |
| `VGG16 code.ipynb` | VGG16 | 48x48 RGB | Fine-tuning (last 4 layers unfrozen) + custom dense head |
| `ResNet-50V2- Additional layers.ipynb` | ResNet50V2 | 224x224 RGB | Fine-tuning (last 50 layers unfrozen) + custom dense head |

All transfer-learning notebooks use ImageNet-pretrained weights via `tensorflow.keras.applications`, with data augmentation (rotation, shift, zoom, horizontal/vertical flip) applied through `ImageDataGenerator`.

## Results

| Model | Train Acc | Test/Val Acc | Notes |
|---|---|---|---|
| **ResNet50V2 + fine-tuning, 224x224** | — | **58.3%** | Best result — unfreezing the last 50 layers and upscaling to 224x224 outperformed every frozen-backbone run |
| LeNet-5 (from scratch) | 95.4% | 49.5% | Strong baseline; overfits after ~epoch 10 without pretrained features |
| InceptionResNetV2 (frozen backbone) | 47.0% | 44.9% | Evaluated on a smaller 7,000-image subset, not the full FER2013 test set |
| ResNet50V2 (frozen backbone) | 43.2% | 42.5% | Frozen ImageNet features underfit low-resolution 48x48 faces |
| VGG16 + fine-tuning | — | see Key Learnings below | Metric mismatch on this run — see below |
| ConvNet fusion / ResNet50 fine-tune | — | — | Trained; evaluation metrics not persisted in the notebook output |

## Key Learnings & Analysis

Beyond the headline numbers, working through seven architectures surfaced a few genuinely useful lessons:

- **Metric choice matters as much as the model.** The VGG16 run was compiled with `tf.keras.metrics.BinaryAccuracy(name='accuracy')`, which scores each of the 7 one-hot output positions independently rather than computing true multiclass accuracy — it reported 87.9%, but the same run's recall (0.24) and F1 (0.36) tell the real story. This is a common and easy-to-miss mistake on multi-class softmax problems, and catching it was a useful reminder to always sanity-check a headline metric against precision/recall rather than trust it in isolation.
- **Frozen ImageNet backbones don't transfer well to low-resolution faces.** Both ResNet50V2 and InceptionResNetV2, used purely as frozen feature extractors, underperformed a from-scratch LeNet-5 — ImageNet's 224x224 natural-image features don't map cleanly onto 48x48 grayscale-derived faces without fine-tuning or upscaling. Partial fine-tuning at higher resolution (the 224x224 ResNet50V2 run) closed most of that gap and became the best-performing model overall.
- **Class imbalance is the dominant error source.** `disgust`, with under 500 training images, is the weakest class in every confusion matrix in this project — a natural next step would be class weighting or oversampling rather than architecture changes.

## Requirements

- Python 3.11+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- OpenCV (`cv2`)
- Jupyter Notebook / JupyterLab

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python jupyter
```

## Usage

1. Download [FER2013](https://www.kaggle.com/msambare/fer2013) and extract it into `train/` and `test/` folders, each containing one subfolder per emotion class.
2. Update the dataset path variables at the top of each notebook to point to your local copy.
3. Run the notebook for the model you want to train or evaluate.

## Future Work

- Address class imbalance directly via class weighting or focal loss
- Standardize on a single, correct evaluation metric (categorical accuracy) across all notebooks
- Re-run the ConvNet fusion and ResNet50 fine-tuning notebooks with full evaluation reporting for a complete comparison
- Explore ensembling the strongest models (ResNet50V2 fine-tuned + LeNet-5) to see if error patterns are complementary
