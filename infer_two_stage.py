# -*- coding: utf-8 -*-
# 两阶段推理：
# - 不保存可视化图片
# - 自动从 runs/train/*/weights/best.pt 选最新检测权重
# - 使用 resnet50_cls_best.pth 做粗分类
# - 输出一个 JSON 文件，记录每张测试图像的检测 + 分类结果

import os, glob, json, yaml, cv2, torch, numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import PIL.Image as Image
from pathlib import Path
from datetime import datetime

DATA_YAML   = "data.yaml"
DET_DIR     = "runs/train"
CLS_WEIGHTS = "resnet50_cls_best.pth"
IMGSZ_DET   = 640
CONF_TH     = 0.25
IOU_TH      = 0.5
PAD_RATIO   = 0.10
SPLIT       = "test"
OUT_DIR     = "runs/two_stage"

def pick_device_det():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def pad_box(x1, y1, x2, y2, w, h, ratio):
    pw = int((x2-x1) * ratio)
    ph = int((y2-y1) * ratio)
    return max(0, x1-pw), max(0, y1-ph), min(w, x2+pw), min(h, y2+ph)

def find_latest_best(det_dir):
    """在 runs/train/*/weights/best.pt 里找最近一个。"""
    pattern = os.path.join(det_dir, "*", "weights", "best.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"未在 {pattern} 找到任何 best.pt")
    paths_sorted = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)
    return paths_sorted[0]

def load_classifier(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt["classes"]
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    return model, classes, tf, device

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    det_weights = find_latest_best(DET_DIR)
    print(f"[TWO_STAGE] 使用检测权重: {det_weights}")
    det_model = YOLO(det_weights)

    cls_model, cls_names, cls_tf, cls_dev = load_classifier(CLS_WEIGHTS)
    device_det = pick_device_det()

    with open(DATA_YAML, "r") as f:
        y = yaml.safe_load(f)
    base = y.get("path", ".") or "."
    img_dir = os.path.join(base, f"{SPLIT}/images")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    print(f"[TWO_STAGE] 在 {img_dir} 找到 {len(img_paths)} 张 {SPLIT} 图片，开始两阶段推理...")

    results_all = []

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        det_out = det_model.predict(
            source=p,
            imgsz=IMGSZ_DET,
            conf=CONF_TH,
            iou=IOU_TH,
            device=device_det,
            verbose=False
        )[0]

        det_records = []
        for b in det_out.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, w, h, PAD_RATIO)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = cls_tf(Image.fromarray(crop_rgb)).unsqueeze(0).to(cls_dev)
            with torch.no_grad():
                logits = cls_model(x)
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            cid = int(prob.argmax())
            cname = cls_names[cid]
            cconf = float(prob.max())
            det_records.append({
                "bbox": [x1, y1, x2, y2],
                "cls_name": cname,
                "cls_id": cid,
                "cls_conf": cconf
            })

        results_all.append({
            "image": os.path.basename(p),
            "detections": det_records
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUT_DIR, f"pred_{SPLIT}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"[TWO_STAGE] 完成，两阶段结果保存在: {json_path}")

if __name__ == "__main__":
    main()
