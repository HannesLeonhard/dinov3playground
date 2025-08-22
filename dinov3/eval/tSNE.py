import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from distinctipy import get_colors, get_colormap


def make_robust_collate_fn(target_size: int = 224):
    """
    Collate that:
      - converts all tensors to float32 and contiguous
      - ensures 3 channels (repeat single-channel, trim >3 to first 3)
      - resizes each image to (target_size, target_size) with bilinear interpolation
      - returns batched tensor and metadata lists
    """
    def collate_fn(batch):
        imgs = []
        labels = []

        for b in batch:
            image, label = b
            t = image # expected tensor CHW
            # make contiguous and float32
            if not t.is_contiguous():
                t = t.contiguous()
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            # channel-handling
            if t.ndim != 3:
                raise RuntimeError(f"unexpected image ndim: {t.ndim}, expected 3 (C,H,W)")
            c, h, w = t.shape
            if c == 1:
                # repeat grayscale -> RGB
                t = t.repeat(3, 1, 1)
            elif c >= 3:
                # take first 3 channels
                if c > 3:
                    t = t[:3, :, :]
            # now t is 3 x H x W
            imgs.append(t)
            labels.append(label)

        # resize each to (3, target_size, target_size)
        resized = []
        for t in imgs:
            if t.shape[1] == target_size and t.shape[2] == target_size:
                resized.append(t)
            else:
                t4 = t.unsqueeze(0)  # 1,3,H,W
                t_res = F.interpolate(t4, size=(target_size, target_size), mode="bilinear", align_corners=False)
                resized.append(t_res.squeeze(0))
        batch_images = torch.stack(resized, dim=0)  # B,3,H,W

        return {
            "image": batch_images,
            "label": labels,
        }

    return collate_fn

def extract_embeddings(
    model,
    dataset,
    device="cuda",
    batch_size=64,
    num_workers=4,
    target_size=224,
):
    """
    Creates latent representation embeddings for a given dataset
    """
    collate_fn = make_robust_collate_fn(target_size=target_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = model.eval()
    model = model.to(device)

    all_emb = []
    all_labels = []

    print("Starting embedding extraction. Batches:", len(loader))
    with torch.inference_mode():
        for batch in tqdm(loader, desc="batches"):
            imgs = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)

            out = model(imgs)  # adapt extraction below if model returns dict/list
            # --- normalize extraction from possible outputs ---
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, dict):
                # try common keys; adjust to your model if required
                for k in ("output", "features", "global", "last_hidden_state"):
                    if k in out:
                        out = out[k]
                        break
                else:
                    out = next(iter(out.values()))
            # ensure tensor
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Model returned {type(out)} - expected Tensor (B, D) or dict/list")

            emb = out.detach().cpu().float().numpy()
            all_emb.append(emb)


            for lab in batch["label"]:
                all_labels.append(str(lab))

    embeddings = np.vstack(all_emb)
    return embeddings, all_labels

def compute_tsne_and_plot(embeddings, labels, out_path="tsne.png", pca_dim=50, tsne_perplexity=30, random_state=0):
    n, d = embeddings.shape
    if pca_dim is not None and pca_dim < d:
        pca = PCA(n_components=min(pca_dim, d), random_state=random_state)
        emb_r = pca.fit_transform(embeddings)
    else:
        emb_r = embeddings

    tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, max(5, n//4 - 1)), random_state=random_state, init="pca", n_jobs=8)
    emb2 = tsne.fit_transform(emb_r)

    unique_labels = sorted(set(labels))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    ints = [label2idx[l] for l in labels]
    colors = get_colors(len(unique_labels))        # list of (r,g,b) tuples, floats 0..1
    cmap = get_colormap(colors)  
    plt.figure(figsize=(10, 10))
    plt.scatter(emb2[:, 0], emb2[:, 1], c=ints, cmap=cmap, s=6, alpha=0.85)
    # get an indexable array of colors from the colormap
    if hasattr(cmap, "colors"):            # ListedColormap (common)
        color_pool = np.array(cmap.colors)
    else:
        # fallback: sample the continuous colormap at regular intervals
        color_pool = np.array(cmap(np.linspace(0, 1, len(unique_labels))))
    # limited legend
    handles = []
    for lab, idx in label2idx.items():
        color = color_pool[idx % len(color_pool)]
        handles.append(plt.Line2D([0], [0], marker='o', color="w",
                                  markerfacecolor=color, markersize=6, label=lab))
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
    plt.title("t-SNE of embeddings")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved TSNE to", out_path)
    return emb2
