# **README — Two-Stage Insect Detection and Classification Pipeline**

This repository implements a **two-stage detection + classification pipeline** for insect recognition:

1. **Stage 1 – YOLO Detector**
2. **Stage 2 – ResNet50 Classifier**
3. **Two-stage inference** (YOLO detect → crop → ResNet classify)

The system achieves significantly higher classification accuracy compared with pure YOLO classification.

---

## **1. Project Structure**

```text
.
│── train_detect_yolo.py        # YOLO detector training
│── crop_from_yolo_labels.py    # Crop patches using YOLO labels
│── train_resnet_classifier.py  # ResNet50 two-stage training + HP search
│── infer_two_stage.py          # Two-stage inference (YOLO + ResNet)
│── plot.ipynb                  # Notebook for plotting / analysis
│── yolo+resnet.ipynb           # End-to-end experiment notebook
│── data.yaml                   # Dataset config for YOLO
│── cls_data/                   # Generated classifier dataset
│── runs/train/                 # YOLO training outputs
│── runs/two_stage/             # Final JSON predictions
```

---

## **2. Requirements**

```bash
Python 3.9+
PyTorch
torchvision
ultralytics
opencv-python
PyYAML
matplotlib
```

Install:

```bash
pip install torch torchvision ultralytics opencv-python pyyaml matplotlib
```

---

## **3. Step 1 — Train YOLO Detector**

```bash
python train_detect_yolo.py --data data.yaml --model yolo11n.pt --epochs 50 --eval_test
```

This script:

* Trains a YOLO detector (YOLOv8 / YOLO11 compatible)
* Prints **one clean summary line per epoch** (no tqdm spam)
* Tracks **box loss** and **cls loss** separately
* Uses early stopping to avoid overfitting
* Saves loss curves:

  * `yolo_box_loss_curve.png`
  * `yolo_cls_loss_curve.png`

YOLO weights are saved under:

```text
runs/train/exp*/weights/best.pt
```

---

## **4. Step 2 — Generate Classifier Dataset**

```bash
python crop_from_yolo_labels.py
```

This script:

* Reads YOLO-style images and labels from `train/` and `valid/`
* Converts YOLO bbox format to `(x1, y1, x2, y2)` and applies small padding
* Crops object patches and saves them to:

```text
cls_data/train/<class_name>/
cls_data/valid/<class_name>/
```

This creates a **clean image classification dataset** for the ResNet classifier.

---

## **5. Step 3 — Train ResNet50 Classifier**

```bash
python train_resnet_classifier.py
```

Features:

* Small hyperparameter search over `(head_lr, ft_lr, weight_decay)`
* Two-stage training:

  * **HEAD stage**: freeze backbone, train only the final FC layer
  * **FT stage**: unfreeze backbone and fine-tune the whole ResNet50
* Early stopping + overfitting detection (based on train/val loss trends)
* Saves:

  * `resnet50_cls_best.pth` (best classifier weights)
  * `loss_curve_cls.png`
  * `val_acc_curve_cls.png`

---

## **6. Step 4 — Two-Stage Inference (YOLO + ResNet)**

```bash
python infer_two_stage.py
```

This script:

1. Automatically picks the **latest** YOLO `best.pt` from `runs/train/*/weights/`
2. Loads `resnet50_cls_best.pth` as the classification head
3. For each image in the `test` split:

   * YOLO detects bounding boxes
   * Each box is cropped (with padding)
   * The crop is classified by ResNet50
4. Writes all results into a JSON file under:

```text
runs/two_stage/pred_test_YYYYMMDD_HHMMSS.json
```

Example JSON entry:

```json
{
  "image": "Weevil-105.jpg",
  "detections": [
    {
      "bbox": [12, 44, 210, 190],
      "cls_name": "Beetles",
      "cls_id": 2,
      "cls_conf": 0.94
    }
  ]
}
```

---

## **7. Jupyter Notebooks**

### `plot.ipynb`

* Utility notebook for **visualization and analysis**. Typical uses:

  * Plot YOLO vs YOLO+ResNet performance (e.g., mAP, per-class AP)
  * Draw confusion matrices for the classifier
  * Compare different runs / configurations
* Not required for running the core pipeline, but useful for **report figures** and debugging.

### `yolo+resnet.ipynb`

* End-to-end **experiment notebook** that combines key steps:

  * Load trained YOLO and ResNet models
  * Run a few sample images through the two-stage pipeline
  * Visualize detections and predicted insect classes
* Good for interactive exploration and for taking **screenshots** for your report.

---

## **8. Summary**

This repository provides a complete **two-stage insect detection and classification system**:

* YOLO for accurate **localization**
* ResNet50 for high-quality **classification**
* Early-stopping, clean logging, and automatic cropping
* Jupyter notebooks for **analysis and visualization**


