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

# def compute_tsne_and_plot(embeddings, labels, out_path="tsne.png", pca_dim=50, tsne_perplexity=30, random_state=0):
#     n, d = embeddings.shape
#     if pca_dim is not None and pca_dim < d:
#         pca = PCA(n_components=min(pca_dim, d), random_state=random_state)
#         emb_r = pca.fit_transform(embeddings)
#     else:
#         emb_r = embeddings

#     tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, max(5, n//4 - 1)), random_state=random_state, init="pca", n_jobs=8)
#     emb2 = tsne.fit_transform(emb_r)

#     unique_labels = sorted(set(labels))
#     label2idx = {lab: i for i, lab in enumerate(unique_labels)}
#     ints = [label2idx[l] for l in labels]
#     colors = get_colors(len(unique_labels))        # list of (r,g,b) tuples, floats 0..1
#     cmap = get_colormap(colors)  
#     plt.figure(figsize=(10, 10))
#     plt.scatter(emb2[:, 0], emb2[:, 1], c=ints, cmap=cmap, s=6, alpha=0.85)
#     # get an indexable array of colors from the colormap
#     if hasattr(cmap, "colors"):            # ListedColormap (common)
#         color_pool = np.array(cmap.colors)
#     else:
#         # fallback: sample the continuous colormap at regular intervals
#         color_pool = np.array(cmap(np.linspace(0, 1, len(unique_labels))))
#     # limited legend
#     handles = []
#     for lab, idx in label2idx.items():
#         color = color_pool[idx % len(color_pool)]
#         handles.append(plt.Line2D([0], [0], marker='o', color="w",
#                                   markerfacecolor=color, markersize=6, label=lab))
#     plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
#     plt.title("t-SNE of embeddings")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     print("Saved TSNE to", out_path)
#     return emb2

def compute_tsne_and_plot(embeddings, labels, out_path="tsne.png",
                          pca_dim=50, tsne_perplexity=30, random_state=0,
                          max_legend=25, base_cmap_name="hsv"):
    """
    Compute PCA (optional) -> t-SNE and save a scatter plot with one unique color per label.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, n_features)
    labels     : iterable of label ids/names (length n_samples)
    out_path   : str, where to save the PNG
    pca_dim    : int or None, reduce to this dim before t-SNE if < original dim
    tsne_perplexity : int, t-SNE perplexity (auto-clamped to valid range)
    random_state : int, RNG seed
    max_legend : int, max number of labels to show in legend (for readability)
    base_cmap_name : str, base cmap to sample for discrete colors (e.g., 'hsv', 'gist_ncar')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings)
    n, d = embeddings.shape

    # Optional PCA
    if pca_dim is not None and pca_dim < d:
        pca = PCA(n_components=min(pca_dim, d), random_state=random_state)
        emb_r = pca.fit_transform(embeddings)
    else:
        emb_r = embeddings

    # Perplexity must be < n; keep user's heuristic but clamp safely
    if n <= 2:
        raise ValueError("Need at least 3 points for t-SNE.")
    heuristic = max(5, n // 4 - 1)
    perplexity = min(tsne_perplexity, heuristic, n - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        n_jobs=8  # keep user's setting if their sklearn supports it
    )
    emb2 = tsne.fit_transform(emb_r)

    # Map labels -> integer indices
    unique_labels = sorted(set(labels))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    ints = np.array([label2idx[l] for l in labels], dtype=int)
    N = len(unique_labels)

    # Build a discrete colormap with exactly N colors (no interpolation)
    base = plt.cm.get_cmap(base_cmap_name, N)  # sample N evenly spaced colors
    cmap = ListedColormap(base(np.arange(N)))
    norm = BoundaryNorm(np.arange(-0.5, N + 0.5, 1), N)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(emb2[:, 0], emb2[:, 1], c=ints, cmap=cmap, norm=norm, s=6, alpha=0.85)

    # Legend: cap to max_legend entries for readability
    show_labels = unique_labels[:max_legend]
    handles = []
    for lab in show_labels:
        idx = label2idx[lab]
        color = cmap(idx)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=6, label=str(lab)))
    if len(show_labels) > 0:
        title = f"Labels (showing {len(show_labels)}/{N})" if N > max_legend else "Labels"
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize='small', title=title, borderaxespad=0.0)

    plt.title("t-SNE of embeddings")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved TSNE to", out_path)

    return emb2


import numpy as np
from collections import Counter
from typing import Sequence, Iterable
import faiss

def _ensure_numpy(emb):
    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)
    return emb

def precision_at_k_from_labels(query_label, neighbor_labels: Sequence, k: int) -> float:
    neighbor_labels_k = neighbor_labels[:k]
    return sum(1 for l in neighbor_labels_k if l == query_label) / float(k)

def average_precision_for_query(query_label, neighbor_labels: Iterable) -> float:
    hits = 0
    sum_precisions = 0.0
    for i, lbl in enumerate(neighbor_labels, start=1):
        if lbl == query_label:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / hits

def compute_label_counts(labels: Sequence) -> Counter:
    return Counter(labels)

def evaluate_knn_faiss(
    embeddings,
    labels: Sequence,
    k_list=(1, 5, 10),
    metric='cosine',                  # 'cosine' or 'euclidean'
    sample_size: int | None = None,
    random_state: int = 0,
    use_gpu: bool = True,             # if a GPU is available, this is much faster
    batch_size: int = 50000,          # query batch size to control RAM
    exact: bool = True,               # False -> switch to IVF/HNSW below
):
    """
    FAISS-based kNN retrieval with the same outputs as your function.
    Returns dict: k -> {"topk_acc","precision@k","recall@k","mAP"}.
    """
    # --- prep ---
    x = _ensure_numpy(embeddings).astype('float32', copy=False)
    N, D = x.shape
    if N == 0:
        raise ValueError("Empty embeddings array")
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(random_state)

    if sample_size is not None and sample_size < N:
        idx = rng.choice(N, size=sample_size, replace=False)
        x = x[idx]
        labels = labels[idx]
        N = x.shape[0]

    # k handling
    k_list = sorted(int(k) for k in set(k_list))
    max_k = max(k_list)
    if N <= 1:
        return {k: {"topk_acc": 0.0, "precision@k": 0.0, "recall@k": 0.0, "mAP": 0.0} for k in k_list}

    # For excluding self, ask FAISS for +1 neighbor; we’ll drop self post hoc
    k_search = min(N, max_k + 1)

    # --- index ---
    if metric == 'cosine':
        # L2-normalize to turn cosine into inner product
        faiss.normalize_L2(x)
        index = faiss.IndexFlatIP(D) if exact else faiss.index_factory(D, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
        metric_is_ip = True
    elif metric == 'euclidean':
        index = faiss.IndexFlatL2(D) if exact else faiss.index_factory(D, "IVF4096,Flat", faiss.METRIC_L2)
        metric_is_ip = False
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    if not exact:
        # Train IVF on a sample
        if not index.is_trained:
            train_n = min(200000, N)
            train_idx = rng.choice(N, size=train_n, replace=False)
            index.train(x[train_idx])

    # GPU?
    res = None
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            if not exact and not metric_is_ip:
                # good default; tune if needed
                index.nprobe = 32
        except Exception:
            # fall back to CPU silently
            pass

    index.add(x)

    # --- search in batches to keep RAM in check ---
    sentinel = -1
    neighbor_idx = np.full((N, max_k), sentinel, dtype=np.int64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = x[start:end]
        Dists, Idxs = index.search(xb, k_search)   # shapes (B, k_search)
        # remove self and keep first max_k
        for i in range(end - start):
            row = Idxs[i]
            # global index of this query = start + i
            filtered = [int(j) for j in row if j != (start + i)]
            filtered = filtered[:max_k]
            neighbor_idx[start + i, :len(filtered)] = filtered

    # --- labels & metrics ---
    neighbor_labels_mat = np.empty((N, max_k), dtype=object)
    neighbor_labels_mat[:, :] = None
    for i in range(N):
        for j in range(max_k):
            jidx = neighbor_idx[i, j]
            if jidx != sentinel:
                neighbor_labels_mat[i, j] = labels[jidx]

    label_counts = compute_label_counts(labels)
    aps = [average_precision_for_query(labels[i], neighbor_labels_mat[i, :].tolist()) for i in range(N)]
    global_mAP = float(np.mean(aps))

    results = {}
    for k in k_list:
        if k <= 0:
            results[k] = {"topk_acc": 0.0, "precision@k": 0.0, "recall@k": 0.0, "mAP": global_mAP}
            continue
        kk = min(k, max_k)
        topk_hits = np.array([any(neighbor_labels_mat[i, :kk] == labels[i]) for i in range(N)], dtype=float)
        topk_acc = float(topk_hits.mean())

        precs = []
        for i in range(N):
            topk_labels = neighbor_labels_mat[i, :kk]
            num_correct = sum(1 for xlab in topk_labels if xlab == labels[i])
            precs.append(num_correct / float(k))
        precision_at_k = float(np.mean(precs))

        recalls = []
        for i in range(N):
            total_relevant = label_counts[labels[i]] - 1
            if total_relevant <= 0:
                continue
            topk_labels = neighbor_labels_mat[i, :kk]
            num_correct = sum(1 for xlab in topk_labels if xlab == labels[i])
            recalls.append(num_correct / float(total_relevant))
        recall_at_k = float(np.mean(recalls)) if len(recalls) else 0.0

        results[k] = {
            "topk_acc": topk_acc,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mAP": global_mAP,
        }
    return results
