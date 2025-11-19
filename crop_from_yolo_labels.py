# -*- coding: utf-8 -*-
# 按 YOLO 标签裁剪 patch，生成分类数据集 cls_data/{train,valid}
# 去掉 tqdm 进度条，不再刷 “裁剪 train: 4% ...” 之类的输出

import os, glob, yaml, cv2

DATA_YAML = "data.yaml"
SPLITS    = ["train", "valid"]
OUTDIR    = "cls_data"
PAD_RATIO = 0.10

def load_names(yaml_path):
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    assert isinstance(names, list) and len(names) > 0, "data.yaml 缺少 names"
    return names, y.get("path", ".")

def yolo2xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
    return x1, y1, x2, y2

def pad_box(x1, y1, x2, y2, w, h, ratio):
    pw = int((x2-x1) * ratio); ph = int((y2-y1) * ratio)
    return max(0, x1-pw), max(0, y1-ph), min(w, x2+pw), min(h, y2+ph)

def main():
    names, base = load_names(DATA_YAML)
    base = base or "."
    for sp in SPLITS:
        for nm in names:
            os.makedirs(os.path.join(OUTDIR, sp, nm), exist_ok=True)

    for sp in SPLITS:
        img_dir = os.path.join(base, sp, "images")
        lbl_dir = os.path.join(base, sp, "labels")
        paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        n_total = len(paths)
        print(f"[CROP] 开始裁剪 {sp}，共 {n_total} 张图像 ...")

        for p in paths:
            base_name = os.path.splitext(os.path.basename(p))[0]
            lp = os.path.join(lbl_dir, base_name + ".txt")
            if not os.path.exists(lp):
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(lp, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            for i, line in enumerate(lines):
                ss = line.split()
                cls = int(ss[0])
                cx, cy, bw, bh = map(float, ss[1:5])
                x1, y1, x2, y2 = yolo2xyxy([cx, cy, bw, bh], w, h)
                x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, w, h, PAD_RATIO)
                crop = img[y1:y2, x1:x2]
                save_dir = os.path.join(OUTDIR, sp, names[cls])
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"{base_name}_{i}.jpg"), crop)

        print(f"[CROP] 完成 {sp} 裁剪。")

    print("完成：cls_data/{train,valid}/<类名>/")

if __name__ == "__main__":
    main()
