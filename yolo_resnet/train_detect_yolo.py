#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_detect_yolo.py
—— Ultralytics YOLO 检测训练脚本（兼容 YOLOv8/YOLO11）。

满足你的要求：
1) 不打印 GPU_mem / batch 进度条，只保留每 epoch 一行 summary。
2) summary 只打印分开的 loss：
      train_box_loss, train_cls_loss, val_box_loss, val_cls_loss
   不打印总 loss，不关心 dfl（但 dfl 仍参与训练）。
3) 最多 50 epoch；命令行给 >50 会被截断。
4) 防过拟合：
   - 规则 1（你现在的要求）：
     如果某轮满足
       (train_box 下降 且 val_box 上升)  或
       (train_cls 下降 且 val_cls 上升)
     视为“一次过拟合轮次”，
     如果连续两轮都发生这种情况，则早停。
   - 规则 2：val_box + val_cls 连续 5 轮没有提升，早停。
5) 训练结束后，画两条 YOLO loss 曲线：
   - yolo_box_loss_curve.png  (train_box vs val_box)
   - yolo_cls_loss_curve.png  (train_cls vs val_cls)
6) 在 test split 上跑 model.val()，打印 4 个指标：
   Precision / Recall / mAP50 / mAP50-95
"""

import os
import sys
import time
import argparse
import glob
from pathlib import Path

import yaml
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("[BOOT] enter train_detect_yolo.py")
print("[BOOT] importing yaml ...")
print("[BOOT] yaml OK")
print("[BOOT] importing numpy ...")
print("[BOOT] numpy OK")
print("[BOOT] importing ultralytics ...")
print("[BOOT] ultralytics OK")
print("[BOOT] importing torch (this step may take a while on first run) ...")
print(f"[BOOT] torch OK | version={torch.__version__}")

# =========================
# 工具函数
# =========================

def _safe_float(v, default=float("nan")) -> float:
    """尽量把各种类型安全地转成 float；失败则返回 NaN。"""
    try:
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.item())
            else:
                return float(v.mean().item())
        if isinstance(v, (list, tuple)):
            vals = [_safe_float(x) for x in v]
            return float(np.nanmean(vals)) if len(vals) > 0 else float("nan")
        return float(v)
    except Exception:
        return default

def scan_yolo_split(split_dir: Path) -> tuple[int, int]:
    """统计某个 split（如 train/valid/test）下的图像和标签数量。"""
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.PNG")
    imgs = []
    for ptn in exts:
        imgs.extend(glob.glob(str(img_dir / ptn)))
    labels = glob.glob(str(label_dir / "*.txt"))
    return len(imgs), len(labels)

def read_yaml(path: Path) -> dict:
    """读取 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_data_paths(data_cfg: dict, data_yaml_path: Path) -> dict:
    """将 data.yaml 中的相对路径解析为绝对路径。"""
    base = data_yaml_path.parent
    out = {}
    for k in ("train", "val", "test"):
        v = data_cfg.get(k, None)
        if v is None:
            out[k] = None
            continue
        p = Path(v)
        if not p.is_absolute():
            p = (base / p).resolve()
        out[k] = str(p)
    for k, v in data_cfg.items():
        if k not in out:
            out[k] = v
    return out

# =========================
# 参数解析
# =========================

def build_argparser():
    parser = argparse.ArgumentParser("YOLO Detect Training")
    parser.add_argument("--data", type=str, default="data.yaml", help="path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="pretrained weight or cfg")

    # 默认最多 50 epoch；命令行给超过 50 会在 main 里截断
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--exist_ok", action="store_true", help="allow overwrite project/name")

    # 优化器/调度器
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--cos_lr", action="store_true")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=50)   # YOLO 自己的 patience，这里后面覆盖成 5
    parser.add_argument("--pretrained", type=bool, default=True)

    # 其它开关
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--val", type=bool, default=True)

    # 超参数搜索 & test 评估开关
    parser.add_argument("--hp_search", action="store_true",
                        help="在正式训练前对 (lr0, weight_decay, batch) 做一次粗略搜索")
    parser.add_argument("--eval_test", action="store_true",
                        help="训练结束后在 test split 上做一次评估（不参与早停/调参）")

    return parser

# =========================
# 回调状态 & 超参搜索配置
# =========================

EARLY_STOP_PATIENCE = 5   # 规则 2：val 没变好 5 轮就停
RULE1_PATIENCE = 2        # 规则 1：连续 2 轮“train↓ + val↑”就停

STATE = {
    "epoch_train_box": [],
    "epoch_train_cls": [],
    "epoch_val_box":   [],
    "epoch_val_cls":   [],

    "cur_box_sum": 0.0,
    "cur_cls_sum": 0.0,
    "cur_dfl_sum": 0.0,
    "cur_batches": 0,

    # 规则 2：用于“5 轮无提升”的 best_score
    "best_val_score": float("inf"),
    "no_improve": 0,

    # 规则 1：用于“连续多少轮 train↓ + val↑”
    "ovf_bad_epochs": 0,
}

YOLO_HP_CANDS = [
    {"name": "cfg01", "lr0": 0.01,  "weight_decay": 5e-4, "batch": 16},
    {"name": "cfg02", "lr0": 0.005, "weight_decay": 5e-4, "batch": 16},
    {"name": "cfg03", "lr0": 0.01,  "weight_decay": 1e-4, "batch": 32},
]

YOLO_HP_SEARCH_EPOCHS = 5
YOLO_HP_SEARCH_FRACTION = 0.25

def reset_state():
    """在一次完整训练开始前重置 STATE，用于超参搜索和正式训练分开统计。"""
    STATE["epoch_train_box"].clear()
    STATE["epoch_train_cls"].clear()
    STATE["epoch_val_box"].clear()
    STATE["epoch_val_cls"].clear()
    STATE["cur_box_sum"] = 0.0
    STATE["cur_cls_sum"] = 0.0
    STATE["cur_dfl_sum"] = 0.0
    STATE["cur_batches"] = 0
    STATE["best_val_score"] = float("inf")
    STATE["no_improve"] = 0
    STATE["ovf_bad_epochs"] = 0

# =========================
# 回调函数
# =========================

def on_train_epoch_start(trainer):
    """每个 train epoch 开始：重置本轮累积。"""
    STATE["cur_box_sum"] = 0.0
    STATE["cur_cls_sum"] = 0.0
    STATE["cur_dfl_sum"] = 0.0
    STATE["cur_batches"] = 0

def on_train_batch_end(trainer):
    """每个 batch 结束：累积 box/cls/dfl（不打印）。"""
    li = getattr(trainer, "loss_items", None)
    if li is None:
        return
    if torch.is_tensor(li):
        li = li.detach().cpu().tolist()
    if isinstance(li, (list, tuple)) and len(li) >= 2:
        box = _safe_float(li[0])
        cls = _safe_float(li[1])
        dfl = _safe_float(li[2]) if len(li) >= 3 else float("nan")
        STATE["cur_box_sum"] += box
        STATE["cur_cls_sum"] += cls
        if np.isfinite(dfl):
            STATE["cur_dfl_sum"] += dfl
        STATE["cur_batches"] += 1

def on_train_epoch_end(trainer):
    """一个 train epoch 结束：计算本轮 train 平均 box/cls。"""
    b = max(1, STATE["cur_batches"])
    avg_box = STATE["cur_box_sum"] / b
    avg_cls = STATE["cur_cls_sum"] / b
    STATE["epoch_train_box"].append(avg_box)
    STATE["epoch_train_cls"].append(avg_cls)

def on_fit_epoch_end(trainer):
    """
    整个 epoch（含 val）结束：
      - 从 trainer.metrics 里取 val_box/val_cls
      - 打印一行 summary
      - 规则 1：train_box/cls 下降且对应 val_box/cls 上升，统计连续轮数，满 2 轮早停
      - 规则 2：5 轮 val(box+cls) 未提升 => 早停
    """
    ep = int(getattr(trainer, "epoch", 0)) + 1
    E  = int(getattr(trainer, "epochs", 0))

    # ---- 当前 train box/cls ----
    if len(STATE["epoch_train_box"]) >= ep:
        train_box = STATE["epoch_train_box"][ep - 1]
    else:
        train_box = STATE["epoch_train_box"][-1] if STATE["epoch_train_box"] else float("nan")

    if len(STATE["epoch_train_cls"]) >= ep:
        train_cls = STATE["epoch_train_cls"][ep - 1]
    else:
        train_cls = STATE["epoch_train_cls"][-1] if STATE["epoch_train_cls"] else float("nan")

    # ---- 从 trainer.metrics 拿 val box/cls ----
    val_box = float("nan")
    val_cls = float("nan")

    metrics_obj = getattr(trainer, "metrics", None)
    rd = None
    if isinstance(metrics_obj, dict):
        rd = metrics_obj
    elif metrics_obj is not None and hasattr(metrics_obj, "results_dict"):
        rd = metrics_obj.results_dict

    if isinstance(rd, dict):
        if "val/box_loss" in rd:
            val_box = _safe_float(rd["val/box_loss"])
        elif "box_loss" in rd:
            val_box = _safe_float(rd["box_loss"])

        if "val/cls_loss" in rd:
            val_cls = _safe_float(rd["val/cls_loss"])
        elif "cls_loss" in rd:
            val_cls = _safe_float(rd["cls_loss"])

    STATE["epoch_val_box"].append(val_box)
    STATE["epoch_val_cls"].append(val_cls)

    # ---- 打印一行 summary（你要的格式）----
    print(
        f"[YOLO] E{ep:03d}/{E:03d} | "
        f"train_box_loss={train_box:.4f} | "
        f"train_cls_loss={train_cls:.4f} | "
        f"val_box_loss={val_box:.4f} | "
        f"val_cls_loss={val_cls:.4f}"
    )

    # ---- 规则 1：组件级 train↓ + val↑ 连续两轮 => 早停 ----
    if ep >= 2:
        prev_tb = STATE["epoch_train_box"][ep - 2]
        prev_tc = STATE["epoch_train_cls"][ep - 2]
        prev_vb = STATE["epoch_val_box"][ep - 2]
        prev_vc = STATE["epoch_val_cls"][ep - 2]

        cur_tb = train_box
        cur_tc = train_cls
        cur_vb = val_box
        cur_vc = val_cls

        overfit_box = (
            np.isfinite(prev_tb) and np.isfinite(cur_tb) and
            np.isfinite(prev_vb) and np.isfinite(cur_vb) and
            cur_tb < prev_tb and cur_vb > prev_vb
        )
        overfit_cls = (
            np.isfinite(prev_tc) and np.isfinite(cur_tc) and
            np.isfinite(prev_vc) and np.isfinite(cur_vc) and
            cur_tc < prev_tc and cur_vc > prev_vc
        )

        if overfit_box or overfit_cls:
            STATE["ovf_bad_epochs"] += 1
        else:
            STATE["ovf_bad_epochs"] = 0

        if STATE["ovf_bad_epochs"] >= RULE1_PATIENCE:
            print("[YOLO] ⚠️ 发现过拟合趋势："
                  "train_box/cls 连续下降且对应的 val_box/cls 连续上升，触发早停。")
            trainer.stop = True
            return

    # ---- 规则 2：5 轮 val(box+cls) 未提升 => 早停 ----
    cur_score = None
    if np.isfinite(val_box) and np.isfinite(val_cls):
        cur_score = val_box + val_cls

    if cur_score is not None:
        if cur_score + 1e-6 < STATE["best_val_score"]:
            STATE["best_val_score"] = cur_score
            STATE["no_improve"] = 0
        else:
            STATE["no_improve"] += 1
            if STATE["no_improve"] >= EARLY_STOP_PATIENCE:
                print("[YOLO] ⏹️ 连续 5 轮 val(box+cls) 未提升，触发早停。")
                trainer.stop = True

# =========================
# 单次训练封装（给超参搜索 & 正式训练共用）
# =========================

def train_one_yolo(args, data_yaml_path: Path,
                   lr0=None, weight_decay=None, batch=None,
                   epochs=None, name_suffix: str = "",
                   fraction: float | None = None,
                   note: str = ""):
    """
    使用给定超参数运行一次 YOLO 训练。
    返回 (best_val_score, model)，其中 best_val_score 是本次训练中 val_box+val_cls 的最小值。
    """
    reset_state()

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end",   on_train_batch_end)
    model.add_callback("on_train_epoch_end",   on_train_epoch_end)
    model.add_callback("on_fit_epoch_end",     on_fit_epoch_end)

    run_name = args.name + (name_suffix if name_suffix else "")

    train_kwargs = dict(
        data=str(data_yaml_path),
        epochs=epochs if epochs is not None else args.epochs,
        imgsz=args.imgsz,
        batch=batch if batch is not None else args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
        lr0=lr0 if lr0 is not None else args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=weight_decay if weight_decay is not None else args.weight_decay,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        patience=EARLY_STOP_PATIENCE,
        pretrained=args.pretrained,
        save=args.save,
        amp=args.amp,
        val=args.val,
        seed=args.seed,
        verbose=False,
    )

    if fraction is not None:
        train_kwargs["fraction"] = fraction

    if note:
        print(f"[YOLO] ▶ {note}")

    t0 = time.time()
    model.train(**train_kwargs)
    dt = time.time() - t0
    print(f"[YOLO] 训练结束，耗时 {dt:.1f}s | run_name={run_name}")

    best_score = STATE["best_val_score"]
    return best_score, model

# =========================
# 主流程
# =========================

def main():
    # 压低 Ultralytics 自己的日志，只保留我们打印的那几行
    LOGGER.setLevel(logging.WARNING)

    print(f"[RUN] \"{sys.executable} -u scripts/train_detect_yolo.py\"")

    # 解析参数
    ap = build_argparser()
    args_cli = ap.parse_args()

    # 强制最多 50 epoch
    if args_cli.epochs > 50:
        print(f"[YOLO] 注意：将 epochs 从 {args_cli.epochs} 截断为 50（按要求最多 50 轮）")
        args_cli.epochs = 50

    # 读取 data.yaml 并解析路径
    data_yaml_path = Path(args_cli.data).resolve()
    if not data_yaml_path.exists():
        print(f"[FATAL] data.yaml 不存在: {data_yaml_path}")
        sys.exit(1)

    print("[BOOT] importing data.yaml ...")
    raw_cfg = read_yaml(data_yaml_path)
    data_cfg = resolve_data_paths(raw_cfg, data_yaml_path)

    names = data_cfg.get("names", [])
    nc = data_cfg.get("nc", len(names) if names else None)
    print(
        f"[YOLO] data.yaml: {{'train': '{data_cfg.get('train')}', "
        f"'val': '{data_cfg.get('val')}', "
        f"'test': '{data_cfg.get('test')}', "
        f"'nc': {nc}, 'names': {names}}}"
    )

    # 数据统计，只做一次，方便你在日志里看到
    def _summ(path_str):
        if path_str is None:
            return (0, 0)
        return scan_yolo_split(Path(path_str).resolve().parent)

    train_images, train_labels = _summ(data_cfg.get("train"))
    val_images,   val_labels   = _summ(data_cfg.get("val"))
    test_images,  test_labels  = _summ(data_cfg.get("test"))

    print(
        f"[YOLO] 数据摘要 | "
        f"train: images={train_images}, labels={train_labels} | "
        f"valid: images={val_images}, labels={val_labels} | "
        f"test:  images={test_images}, labels={test_labels}"
    )

    if train_images == 0 or train_labels == 0:
        print("[FATAL] 训练集为空或无标签（images 或 labels 为 0）。")
        sys.exit(1)

    # =============================
    # Step 1: YOLO 超参数搜索（可选）
    # =============================

    best_cfg = {
        "lr0": args_cli.lr0,
        "weight_decay": args_cli.weight_decay,
        "batch": args_cli.batch,
        "name": "default",
        "score": float("inf"),
    }

    if args_cli.hp_search:
        print("[YOLO-HP] 开始粗略搜索学习率 / weight_decay / batch 组合 ...")
        for i, cfg in enumerate(YOLO_HP_CANDS, start=1):
            cfg_name = cfg["name"]
            lr0      = cfg["lr0"]
            wd       = cfg["weight_decay"]
            batch    = cfg["batch"]

            note = (f"[YOLO-HP] cfg{i:02d} ({cfg_name}) | "
                    f"lr0={lr0:.3g}, weight_decay={wd:.1e}, batch={batch}, "
                    f"epochs={min(args_cli.epochs, YOLO_HP_SEARCH_EPOCHS)}, "
                    f"fraction={YOLO_HP_SEARCH_FRACTION}")

            score, _ = train_one_yolo(
                args_cli,
                data_yaml_path,
                lr0=lr0,
                weight_decay=wd,
                batch=batch,
                epochs=min(args_cli.epochs, YOLO_HP_SEARCH_EPOCHS),
                name_suffix=f"_hp_{cfg_name}",
                fraction=YOLO_HP_SEARCH_FRACTION,
                note=note,
            )

            print(f"[YOLO-HP] cfg{i:02d} ({cfg_name}) | best_val_score={score:.4f}")
            if score < best_cfg["score"]:
                best_cfg.update(
                    {"lr0": lr0, "weight_decay": wd, "batch": batch,
                     "name": cfg_name, "score": score}
                )

        print(
            f"[YOLO-HP] 最优配置: name={best_cfg['name']} | "
            f"lr0={best_cfg['lr0']:.3g}, weight_decay={best_cfg['weight_decay']:.1e}, "
            f"batch={best_cfg['batch']} | best_val_score={best_cfg['score']:.4f}"
        )

        # 用最优配置覆盖后续正式训练参数
        args_cli.lr0 = best_cfg["lr0"]
        args_cli.weight_decay = best_cfg["weight_decay"]
        args_cli.batch = best_cfg["batch"]

    # =============================
    # Step 2: 正式 YOLO 训练（使用最优超参数）
    # =============================

    note = (f"正式训练 | lr0={args_cli.lr0:.3g}, weight_decay={args_cli.weight_decay:.1e}, "
            f"batch={args_cli.batch}, epochs={args_cli.epochs}")
    best_val_score, final_model = train_one_yolo(
        args_cli,
        data_yaml_path,
        note=note,
    )
    print(f"[YOLO] 正式训练完成 | best_val_score={best_val_score:.4f}")

    # =============================
    # Step 3: 画 YOLO loss 曲线（只画正式训练这一次的）
    # =============================

    epochs = min(len(STATE["epoch_train_box"]), len(STATE["epoch_val_box"]))
    if epochs > 0:
        x = np.arange(1, epochs + 1)
        tb = np.array(STATE["epoch_train_box"][:epochs])
        tc = np.array(STATE["epoch_train_cls"][:epochs])
        vb = np.array(STATE["epoch_val_box"][:epochs])
        vc = np.array(STATE["epoch_val_cls"][:epochs])

        plt.figure(figsize=(6, 4))
        plt.plot(x, tb, label="train_box_loss")
        plt.plot(x, vb, label="val_box_loss")
        plt.xlabel("epoch")
        plt.ylabel("box loss")
        plt.title("YOLO Box Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("yolo_box_loss_curve.png", dpi=180)

        plt.figure(figsize=(6, 4))
        plt.plot(x, tc, label="train_cls_loss")
        plt.plot(x, vc, label="val_cls_loss")
        plt.xlabel("epoch")
        plt.ylabel("cls loss")
        plt.title("YOLO Cls Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("yolo_cls_loss_curve.png", dpi=180)

        print("已保存 YOLO loss 曲线：yolo_box_loss_curve.png, yolo_cls_loss_curve.png")

    # =============================
    # Step 4: 可选 test split 最终评估（不参与调参 / 早停）
    # =============================

    if args_cli.eval_test:
        print("[YOLO][TEST] 开始在 test split 上做最终评估（仅报告，不参与训练/调参）...")
        try:
            res = final_model.val(
                data=str(data_yaml_path),
                split="test",
                imgsz=args_cli.imgsz,
                batch=args_cli.batch,
                device=args_cli.device,
                verbose=False,
            )

            metrics = None
            if isinstance(res, dict):
                metrics = res
            else:
                if hasattr(res, "results_dict"):
                    metrics = res.results_dict
                elif hasattr(res, "metrics") and hasattr(res.metrics, "results_dict"):
                    metrics = res.metrics.results_dict

            if isinstance(metrics, dict):
                p  = _safe_float(metrics.get("metrics/precision(B)", float("nan")))
                r  = _safe_float(metrics.get("metrics/recall(B)", float("nan")))
                m1 = _safe_float(metrics.get("metrics/mAP50(B)", float("nan")))
                m2 = _safe_float(metrics.get("metrics/mAP50-95(B)", float("nan")))
                print("[YOLO][TEST] Precision={:.4f} | Recall={:.4f} | mAP50={:.4f} | mAP50-95={:.4f}".format(p, r, m1, m2))
            else:
                print("[YOLO][TEST] 无法解析 metrics 结果。")
        except Exception as e:
            print(f"[YOLO][TEST] 评估 test split 失败：{e}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        print(f"SystemExit: [FATAL] {sys.executable} -u scripts/train_detect_yolo.py | returncode={code}")
        raise
