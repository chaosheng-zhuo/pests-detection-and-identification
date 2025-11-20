# xAi.ipynb

This script performs explainability (XAI) on a trained **ResNet50 pest classifier**, producing **Original + Saliency + Integrated Gradients (IG)** visualizations.

## **What the script does**

1. Uses `crop_from_yolo_labels.py` to create a classification dataset from YOLO boxes.

2. Loads the trained model: `resnet50_cls_best.pth`.

3. Runs inference on `cls_data/valid` and separates correct vs. wrong predictions.

4. Computes **Saliency** and **IG** attribution maps.

5. Saves triplet visualizations to:

   ```
   xai_resnet/correct/
   xai_resnet/wrong/
   ```

## **Required directory layout**

```
ROOT/
├── data.yaml
├── train/ (YOLO images+labels)
├── valid/ (YOLO images+labels)
├── crop_from_yolo_labels.py
├── resnet50_cls_best.pth
└── xai.py
```

After running, it generates:

```
cls_data/…
xai_resnet/correct/*.png
xai_resnet/wrong/*.png
```

## **How to run**

1. `cd /content/9517_data`
2. Execute all cells in `xai.py` (or run as a script).
3. Check the generated XAI PNGs.

## **How to interpret results**

- **Correct samples:** model focuses on real insect features.
- **Wrong samples:** highlights misleading areas (background, glare, blur), useful for error analysis.

## **Common issues**

- Class names mismatch → ensure `cls_data` structure matches the one used during training.
- Captum / numpy conflict → script installs Captum with `--no-deps` to prevent version overwrite.