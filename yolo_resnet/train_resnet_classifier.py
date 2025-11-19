# -*- coding: utf-8 -*-
# ResNet50 分类训练（带小型超参数搜索）：
# - 先做 2~2 个 epoch 的“短跑搜索”，尝试 3 组 (LR_HEAD, LR_FT, WD)
# - 选出 best_val_loss 最小的一组作为最终配置
# - 再用这组超参数做完整 two-stage 训练：
#     阶段 1：只训 FC 头 (HEAD)
#     阶段 2：解冻全网微调 (FT)
# - 每个 epoch 打一行 summary，没有 tqdm 百分比
# - 有 early stopping：val_loss 连续若干轮不提升就停
# - 额外规则：train_loss 下降且 val_loss 上升，连续两轮视为过拟合，立即停止
# - 保存：
#     resnet50_cls_best.pth
#     loss_curve_cls.png
#     val_acc_curve_cls.png

import os, time, copy, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= 基本配置 =================
ROOT = "cls_data"
IMGSZ = 224
BATCH = 32
NUM_WORKERS = 2

# 最长训练轮数（HEAD + FT）
FREEZE_EPOCHS   = 5      # 正式训练时 HEAD 阶段最多轮数
FINETUNE_EPOCHS = 45     # 正式训练时 FT 阶段最多轮数
EPOCHS_MAX      = FREEZE_EPOCHS + FINETUNE_EPOCHS

# 正式训练默认超参数（会被搜索阶段的 best_cfg 覆盖）
LR_HEAD_DEFAULT = 1e-3
LR_FT_DEFAULT   = 1e-4
WD_DEFAULT      = 1e-4

EARLY_STOP_PATIENCE = 7  # val_loss 连续未提升的耐心轮数

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
)

# =============== ResNet 超参候选（小网格） ===============
RESNET_HP_CANDS = [
    {
        "name": "cfg01",
        "head_lr": 1e-3,
        "ft_lr":   1e-4,
        "weight_decay": 1e-4,
    },
    {
        "name": "cfg02",
        "head_lr": 5e-4,
        "ft_lr":   1e-4,
        "weight_decay": 1e-4,
    },
    {
        "name": "cfg03",
        "head_lr": 1e-3,
        "ft_lr":   5e-5,
        "weight_decay": 5e-5,
    },
]

# 搜索阶段用的 HEAD / FT 轮数（短一点）
SEARCH_HEAD_EPOCHS = 2
SEARCH_FT_EPOCHS   = 2


# ================= 数据部分 =================
def build_dataloaders(root=ROOT):
    train_tf = transforms.Compose([
        transforms.Resize(int(IMGSZ * 1.15)),
        transforms.RandomResizedCrop(IMGSZ, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(IMGSZ * 1.15)),
        transforms.CenterCrop(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "valid")
    test_dir  = os.path.join(root, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError("分类数据需存在 cls_data/train 与 cls_data/valid")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tf) if os.path.isdir(test_dir) else None

    classes = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True) if test_ds is not None else None

    print(f"[CLS] 数据摘要 | train={len(train_ds)} | valid={len(val_ds)} | "
          f"test={len(test_ds) if test_ds is not None else 0} | num_classes={len(classes)}")
    print(f"[CLS] 类别: {classes}")
    return train_loader, val_loader, test_loader, classes


# ================= 模型 & 评估 =================
def build_model(num_classes: int):
    # 用 torchvision 自带的 ImageNet 预训练权重
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(weights="IMAGENET1K_V2")
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    model.to(DEVICE)
    return model

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    if n == 0:
        return float("nan"), float("nan")
    return loss_sum / n, correct / n


# ============== 单次训练（可用于搜索 or 正式训练） ==============
def train_one_config(
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    classes,
    search_only: bool = False,
):
    """
    使用一组超参数 (head_lr, ft_lr, weight_decay) 进行一次 two-stage 训练：
      - search_only=True: 作为“短跑搜索”，HEAD+FT 各只跑少量 epoch，
                          返回 best_val_loss（不保存模型、不画图）
      - search_only=False: 正式训练，返回 (model, history, best_val_loss, test_metrics)
    """
    num_classes = len(classes)
    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()

    head_lr = cfg.get("head_lr", LR_HEAD_DEFAULT)
    ft_lr   = cfg.get("ft_lr", LR_FT_DEFAULT)
    wd      = cfg.get("weight_decay", WD_DEFAULT)

    # 搜索 vs 正式训练：epoch 上限不同
    if search_only:
        head_max = SEARCH_HEAD_EPOCHS
        ft_max   = SEARCH_FT_EPOCHS
        prefix_head = "[CLS-HP] HEAD"
        prefix_ft   = "[CLS-HP] FT  "
    else:
        head_max = FREEZE_EPOCHS
        ft_max   = FINETUNE_EPOCHS
        prefix_head = "[CLS] HEAD"
        prefix_ft   = "[CLS] FT  "

    total_max_epochs = head_max + ft_max

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_val_loss = float("inf")
    no_improve = 0
    epoch_idx = 0

    # 新增：用于“train_loss↓ + val_loss↑” 连续轮数统计
    ovf_bad_epochs = 0

    # ---------- 阶段 1：只训分类头 ----------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=head_lr, weight_decay=wd)

    while epoch_idx < head_max:
        epoch_idx += 1
        model.train()
        loss_sum, n, correct_sum = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            pred = logits.argmax(dim=1)
            correct_sum += (pred == y).sum().item()

        train_loss = loss_sum / n
        train_acc  = correct_sum / n
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["epoch"].append(epoch_idx)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"{prefix_head} E{epoch_idx:03d}/{total_max_epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # === 规则 A：按 val_loss 最优保存 + 连续未提升 early stopping ===
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"{prefix_head} 早停：val_loss 连续未提升")
                break

        # === 规则 B：train_loss 下降且 val_loss 上升，连续两轮判过拟合 ===
        if len(history["epoch"]) >= 2:
            prev_train = history["train_loss"][-2]
            prev_val   = history["val_loss"][-2]
            cur_train  = history["train_loss"][-1]
            cur_val    = history["val_loss"][-1]

            overfit = (
                np.isfinite(prev_train) and np.isfinite(prev_val) and
                np.isfinite(cur_train)  and np.isfinite(cur_val)  and
                cur_train < prev_train and cur_val > prev_val
            )
            if overfit:
                ovf_bad_epochs += 1
            else:
                ovf_bad_epochs = 0

            if ovf_bad_epochs >= 2:
                print(f"{prefix_head} 早停：检测到 train_loss 连续下降且 val_loss 连续上升（可能过拟合）")
                break

    # ---------- 阶段 2：全网微调 ----------
    # 回到当前 best_state 再微调
    if best_state is not None:
        model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=wd)
    no_improve = 0  # 重新计数 early stopping
    # ovf_bad_epochs 不清零，保持“跨阶段连续”的判定；如果你只想阶段内单独算，可以在这里设为 0

    while epoch_idx < head_max + ft_max:
        epoch_idx += 1
        model.train()
        loss_sum, n, correct_sum = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            pred = logits.argmax(dim=1)
            correct_sum += (pred == y).sum().item()

        train_loss = loss_sum / n
        train_acc  = correct_sum / n
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["epoch"].append(epoch_idx)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"{prefix_ft}   E{epoch_idx:03d}/{total_max_epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # === 规则 A：按 val_loss 最优保存 + 连续未提升 early stopping ===
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"{prefix_ft} 早停：val_loss 连续未提升")
                break

        # === 规则 B：train_loss 下降且 val_loss 上升，连续两轮判过拟合 ===
        if len(history["epoch"]) >= 2:
            prev_train = history["train_loss"][-2]
            prev_val   = history["val_loss"][-2]
            cur_train  = history["train_loss"][-1]
            cur_val    = history["val_loss"][-1]

            overfit = (
                np.isfinite(prev_train) and np.isfinite(prev_val) and
                np.isfinite(cur_train)  and np.isfinite(cur_val)  and
                cur_train < prev_train and cur_val > prev_val
            )
            if overfit:
                ovf_bad_epochs += 1
            else:
                ovf_bad_epochs = 0

            if ovf_bad_epochs >= 2:
                print(f"{prefix_ft} 早停：检测到 train_loss 连续下降且 val_loss 连续上升（可能过拟合）")
                break

    # ---------- 返回结果 ----------
    if search_only:
        # 搜索阶段只关心 best_val_loss
        return best_val_loss

    # 正式训练：加载 best_state，评估 test
    if best_state is not None:
        model.load_state_dict(best_state)
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion)
    else:
        test_loss, test_acc = float("nan"), float("nan")

    return model, history, best_val_loss, (test_loss, test_acc)


# ================= 整体流程（搜索 + 正式训练） =================
def train_classifier_with_hp_search():
    train_loader, val_loader, test_loader, classes = build_dataloaders(ROOT)

    # ---------- Step 1: 小型超参搜索 ----------
    print("[CLS-HP] 开始 ResNet50 超参数搜索 (head_lr, ft_lr, weight_decay)...")
    best_cfg = None
    best_score = float("inf")

    for i, cfg in enumerate(RESNET_HP_CANDS, start=1):
        name = cfg["name"]
        head_lr = cfg["head_lr"]
        ft_lr   = cfg["ft_lr"]
        wd      = cfg["weight_decay"]

        print(
            f"[CLS-HP] cfg{i:02d} ({name}) | "
            f"head_lr={head_lr:.1e}, ft_lr={ft_lr:.1e}, weight_decay={wd:.1e} | "
            f"HEAD_epochs={SEARCH_HEAD_EPOCHS}, FT_epochs={SEARCH_FT_EPOCHS}"
        )

        score = train_one_config(
            cfg,
            train_loader,
            val_loader,
            test_loader,
            classes,
            search_only=True,
        )
        print(f"[CLS-HP] cfg{i:02d} ({name}) | best_val_loss={score:.4f}")

        if score < best_score:
            best_score = score
            best_cfg = cfg.copy()

    if best_cfg is None:
        print("[CLS-HP] 超参数搜索失败，回退到默认配置")
        best_cfg = {
            "name": "default",
            "head_lr": LR_HEAD_DEFAULT,
            "ft_lr":   LR_FT_DEFAULT,
            "weight_decay": WD_DEFAULT,
        }

    print(
        f"[CLS-HP] 最优配置: {best_cfg['name']} | "
        f"head_lr={best_cfg['head_lr']:.1e}, ft_lr={best_cfg['ft_lr']:.1e}, "
        f"weight_decay={best_cfg['weight_decay']:.1e} | "
        f"best_val_loss={best_score:.4f}"
    )

    # ---------- Step 2: 用最优配置做正式训练 ----------
    print("[CLS] 使用最优 ResNet 超参进行正式训练 (two-stage: HEAD + FT)...")
    t0 = time.time()
    model, history, best_val_loss, (test_loss, test_acc) = train_one_config(
        best_cfg,
        train_loader,
        val_loader,
        test_loader,
        classes,
        search_only=False,
    )
    dt = time.time() - t0
    print(
        f"[CLS] 正式训练完成 | best_val_loss={best_val_loss:.4f} | "
        f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f} | 耗时 {dt:.1f}s"
    )

    # ---------- 保存权重 ----------
    torch.save({"state_dict": model.state_dict(), "classes": classes}, "resnet50_cls_best.pth")
    print("已保存：resnet50_cls_best.pth")

    # ---------- 画曲线 ----------
    x = np.array(history["epoch"])
    train_loss = np.array(history["train_loss"])
    val_loss   = np.array(history["val_loss"])
    val_acc    = np.array(history["val_acc"])

    plt.figure(figsize=(6, 4))
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, val_loss,   label="val_loss")
    plt.title("Loss Curve (Classifier)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve_cls.png", dpi=180)

    plt.figure(figsize=(6, 4))
    plt.plot(x, val_acc, label="val_acc")
    plt.title("Val Accuracy Curve (Classifier)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_acc_curve_cls.png", dpi=180)

    print("已保存曲线：loss_curve_cls.png | val_acc_curve_cls.png")


def main():
    print("[TWO_STAGE] 开始训练 ResNet50 分类器（含超参数搜索）")
    train_classifier_with_hp_search()


if __name__ == "__main__":
    main()
