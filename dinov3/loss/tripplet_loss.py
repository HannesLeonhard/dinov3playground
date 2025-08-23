import torch
import torch.nn as nn
import torch.nn.functional as F
import dinov3.distributed as distributed  # for get_rank()

class TripletLoss(nn.Module):
    """
    Triplet loss that uses a global pool of embeddings+labels and local_indices mapping
    (which rows in the global pool correspond to the local anchors).

    forward(
        anchors: Tensor[B_local, D]                 <- student embeddings (requires grad)
        global_emb: Tensor[N_total, D]              <- gathered embeddings (teacher/EMA) (will be detached)
        global_labels: Tensor[N_total] (int64)      <- labels produced by DBSCAN (-1 => noise)
        local_indices: Tensor[B_local] (int64)      <- indices in global_emb pointing to anchors
        margin: float (optional)
        seed: int | None (optional)                 <- deterministic RNG seed (recommended)
    ) -> scalar loss tensor
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def init_weights(self):
        # API parity with DINOLoss
        return

    def forward(
        self,
        anchors: torch.Tensor,
        global_emb: torch.Tensor,
        global_labels: torch.Tensor,
        local_indices: torch.Tensor,
        *,
        margin: float | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        if margin is None:
            margin = self.margin

        # basic checks
        if anchors is None or global_emb is None or global_labels is None or local_indices is None:
            return anchors.new_tensor(0.0)

        device = anchors.device
        anchors = anchors.contiguous().to(device)
        global_emb = global_emb.contiguous().to(device)
        global_labels = global_labels.contiguous().to(device)
        local_indices = local_indices.contiguous().to(device)

        B_local, D = anchors.shape
        N_total = global_emb.shape[0]

        if B_local == 0 or N_total == 0:
            return anchors.new_tensor(0.0)

        # Normalize both sides (cosine similarity)
        anchors = F.normalize(anchors, p=2, dim=1)
        # use detached global embeddings so no grad flows through them
        global_emb_det = F.normalize(global_emb.detach(), p=2, dim=1)

        # Bring labels to CPU for building small maps (N_total is small per-iteration)
        labels_cpu = global_labels.cpu().long().numpy()

        # Build mapping: cluster_label -> list of indices (excluding -1)
        cluster_to_indices = {}
        for idx, lab in enumerate(labels_cpu):
            lab = int(lab)
            if lab == -1:
                continue
            cluster_to_indices.setdefault(lab, []).append(idx)

        # deterministic generator: if seed provided use that, else build from iteration+rank
        if seed is None:
            # fallback deterministic seed based on rank only (you should pass iteration-based seed from caller)
            rank = distributed.get_rank() if hasattr(distributed, "get_rank") else 0
            seed = int(rank)
        # We'll sample using a CPU-based torch.Generator for determinism
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))

        pos_list = []
        neg_list = []
        valid_mask = []

        # Use numpy array for faster lookups
        # For reproducibility, we use torch.randint with generator `g` to pick random indices
        # (we generate indexes on CPU and then use them)
        for local_idx in local_indices.cpu().tolist():
            lab = int(labels_cpu[local_idx])
            if lab == -1:
                # anchor is noise -> skip
                valid_mask.append(False)
                pos_list.append(torch.zeros(D, device=device))
                neg_list.append(torch.zeros(D, device=device))
                continue

            # positive candidates (same cluster excluding anchor index)
            candidates = cluster_to_indices.get(lab, [])
            filtered = [c for c in candidates if c != int(local_idx)]
            if len(filtered) == 0:
                valid_mask.append(False)
                pos_list.append(torch.zeros(D, device=device))
                neg_list.append(torch.zeros(D, device=device))
                continue

            # sample positive deterministically
            if len(filtered) == 1:
                pos_idx = filtered[0]
            else:
                ridx = torch.randint(low=0, high=len(filtered), size=(1,), generator=g).item()
                pos_idx = filtered[ridx]

            # negative: pick a random other cluster, then a random element from that cluster
            other_clusters = [c for c in cluster_to_indices.keys() if c != lab]
            if len(other_clusters) == 0:
                # no negatives available
                valid_mask.append(False)
                pos_list.append(torch.zeros(D, device=device))
                neg_list.append(torch.zeros(D, device=device))
                continue

            if len(other_clusters) == 1:
                chosen_cluster = other_clusters[0]
            else:
                ridx = torch.randint(low=0, high=len(other_clusters), size=(1,), generator=g).item()
                chosen_cluster = other_clusters[ridx]

            neg_candidates = cluster_to_indices[chosen_cluster]
            if len(neg_candidates) == 1:
                neg_idx = neg_candidates[0]
            else:
                ridx = torch.randint(low=0, high=len(neg_candidates), size=(1,), generator=g).item()
                neg_idx = neg_candidates[ridx]

            # append embeddings (move to anchor device)
            pos_list.append(global_emb_det[pos_idx].to(device))
            neg_list.append(global_emb_det[neg_idx].to(device))
            valid_mask.append(True)

        # compute triplet loss using cosine similarity (anchors * pos / neg)
        if not any(valid_mask):
            return anchors.new_tensor(0.0)

        pos_tensor = torch.stack(pos_list, dim=0)  # [B_local, D]
        neg_tensor = torch.stack(neg_list, dim=0)  # [B_local, D]
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool, device=device)

        sim_ap = (anchors * pos_tensor).sum(dim=1)  # cosine similarity
        sim_an = (anchors * neg_tensor).sum(dim=1)

        # loss = max(0, sim_an - sim_ap + margin)  (we want sim_ap > sim_an + margin)
        raw = (sim_an - sim_ap + float(margin)).clamp(min=0.0)

        if valid_mask_tensor.any():
            loss_val = raw[valid_mask_tensor].mean()
        else:
            loss_val = anchors.new_tensor(0.0)

        return loss_val
