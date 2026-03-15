"""
Multi-View Drift Detector: fuses learned representations of persona-vector
differences with precomputed baseline signals (SBERT, LDA, TF-IDF, MMD) to
produce a single drift score per transition.

Unlike the simple contrastive encoder that projects individual vectors and
computes cosine distance (prone to saturation), this model directly predicts
drift magnitude by jointly reasoning over:
  - What changed semantically (persona-vector difference)
  - What complementary signals say (baseline features)

Training uses ALL transitions with continuous regression targets (normalized
SBERT distances) plus a contrastive term on weakly-labeled pairs.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


class MultiViewDriftDetector(nn.Module):
    """
    Input per transition:
      - v_t, v_{t+1}: L2-normalized persona vectors (persona_dim each)
      - baselines: [sbert_dist, lda_jsd, tfidf_jsd, mmd_dist] (n_baselines)

    Architecture:
      1. Difference encoder: |v_t - v_{t+1}| → low-dim projection
      2. Interaction encoder: v_t ⊙ v_{t+1} → low-dim projection
      3. Fusion: concat(diff_proj, interact_proj, baselines) → MLP → sigmoid → score
    """

    def __init__(
        self,
        persona_dim: int = 384,
        n_baselines: int = 4,
        proj_dim: int = 64,
        hidden_dim: int = 48,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.diff_encoder = nn.Sequential(
            nn.Linear(persona_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interact_encoder = nn.Sequential(
            nn.Linear(persona_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_in = proj_dim * 2 + n_baselines
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, v_t: torch.Tensor, v_t1: torch.Tensor, baselines: torch.Tensor
    ) -> torch.Tensor:
        diff = torch.abs(v_t - v_t1)
        interact = v_t * v_t1
        h_diff = self.diff_encoder(diff)
        h_interact = self.interact_encoder(interact)
        fused = torch.cat([h_diff, h_interact, baselines], dim=1)
        return torch.sigmoid(self.fusion(fused).squeeze(-1))


def drift_detection_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor = None,
    contrastive_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined loss for all transitions:
      1. Regression: Smooth-L1 between predicted scores and normalized SBERT targets
      2. Contrastive (on labeled pairs only): stable → low score, shift → high score

    pred: (N,) predicted drift scores in [0, 1]
    targets: (N,) regression targets (normalized SBERT distances) in [0, 1]
    labels: (N,) weak labels: 0=stable, 1=shift, -1=unlabeled
    """
    loss_reg = F.smooth_l1_loss(pred, targets)

    loss_contrast = torch.tensor(0.0, device=pred.device)
    if labels is not None:
        mask = labels >= 0
        if mask.sum() > 0:
            p = pred[mask]
            y = labels[mask].float()
            loss_contrast = (y * (1 - p).pow(2) + (1 - y) * p.pow(2)).mean()

    return loss_reg + contrastive_weight * loss_contrast


def _build_transition_data(
    persona_vectors: Dict[str, np.ndarray],
    window_ids: List[str],
    sbert_drift: List[float],
    lda_drift: List[float],
    tfidf_drift: List[float],
    mmd_drift: List[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build arrays for all N-1 transitions.
    Returns: V_t, V_t1, baselines, regression_targets
    """
    n = len(window_ids) - 1
    V_t = np.array([persona_vectors[window_ids[i]] for i in range(n)])
    V_t1 = np.array([persona_vectors[window_ids[i + 1]] for i in range(n)])

    sbert_arr = np.array(sbert_drift)
    lda_arr = np.array(lda_drift)
    tfidf_arr = np.array(tfidf_drift)

    if mmd_drift is not None and len(mmd_drift) == n:
        mmd_arr = np.array(mmd_drift)
    else:
        mmd_arr = np.zeros(n)

    baselines = np.column_stack([sbert_arr, lda_arr, tfidf_arr, mmd_arr])

    # Regression targets: normalize SBERT distances to [0.05, 0.95]
    d_min, d_max = sbert_arr.min(), sbert_arr.max()
    targets = 0.05 + 0.90 * (sbert_arr - d_min) / (d_max - d_min + 1e-9)

    return V_t, V_t1, baselines, targets


def train_drift_detector(
    model: nn.Module,
    persona_vectors: Dict[str, np.ndarray],
    window_ids: List[str],
    sbert_drift: List[float],
    lda_drift: List[float],
    tfidf_drift: List[float],
    mmd_drift: List[float] = None,
    weak_labels: np.ndarray = None,
    labeled_indices: List[int] = None,
    epochs: int = 150,
    lr: float = 0.001,
    patience: int = 30,
    contrastive_weight: float = 0.5,
    gradient_clip: float = 1.0,
    device: Optional[str] = None,
) -> List[float]:
    """
    Train the drift detector on ALL transitions.

    weak_labels:     0/1 labels for each of the N-1 transitions (-1 = unlabeled)
    labeled_indices: not used if weak_labels covers all transitions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    V_t, V_t1, baselines, targets = _build_transition_data(
        persona_vectors, window_ids,
        sbert_drift, lda_drift, tfidf_drift, mmd_drift,
    )
    n = len(targets)

    # Prepare weak labels for all transitions (-1 where unlabeled)
    all_labels = np.full(n, -1, dtype=np.int64)
    if weak_labels is not None and len(weak_labels) == n:
        all_labels = weak_labels.copy()
    elif weak_labels is not None and labeled_indices is not None:
        for i, idx in enumerate(labeled_indices):
            if 0 <= idx < n:
                all_labels[idx] = weak_labels[i]

    V_t_t = torch.FloatTensor(V_t).to(device)
    V_t1_t = torch.FloatTensor(V_t1).to(device)
    B_t = torch.FloatTensor(baselines).to(device)
    targets_t = torch.FloatTensor(targets).to(device)
    labels_t = torch.LongTensor(all_labels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        pred = model(V_t_t, V_t1_t, B_t)
        loss = drift_detection_loss(pred, targets_t, labels_t, contrastive_weight)

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        l = loss.item()
        losses.append(l)
        if l < best_loss:
            best_loss = l
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return losses


def predict_drift(
    model: nn.Module,
    persona_vectors: Dict[str, np.ndarray],
    window_ids: List[str],
    sbert_drift: List[float],
    lda_drift: List[float],
    tfidf_drift: List[float],
    mmd_drift: List[float] = None,
    smooth_sigma: float = 0.8,
    device: Optional[str] = None,
) -> List[float]:
    """
    Predict drift scores for all transitions and optionally apply Gaussian smoothing.
    Returns drift curve as list of floats.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    V_t, V_t1, baselines, _ = _build_transition_data(
        persona_vectors, window_ids,
        sbert_drift, lda_drift, tfidf_drift, mmd_drift,
    )
    with torch.no_grad():
        pred = model(
            torch.FloatTensor(V_t).to(device),
            torch.FloatTensor(V_t1).to(device),
            torch.FloatTensor(baselines).to(device),
        ).cpu().numpy()

    if smooth_sigma > 0 and len(pred) > 3:
        pred = gaussian_filter1d(pred.astype(float), sigma=smooth_sigma)

    return pred.tolist()


def detector_contrastiveness(
    model: nn.Module,
    persona_vectors: Dict[str, np.ndarray],
    window_ids: List[str],
    sbert_drift: List[float],
    lda_drift: List[float],
    tfidf_drift: List[float],
    mmd_drift: List[float],
    weak_labels_all: np.ndarray,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Separation = mean score(shift pairs) - mean score(stable pairs).
    Returns (sep_before [raw SBERT], sep_after [detector]).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sbert_arr = np.array(sbert_drift)
    mask_stable = weak_labels_all == 0
    mask_shift = weak_labels_all == 1
    if mask_stable.sum() == 0 or mask_shift.sum() == 0:
        return 0.0, 0.0

    sep_before = float(sbert_arr[mask_shift].mean() - sbert_arr[mask_stable].mean())

    model.eval()
    V_t, V_t1, baselines, _ = _build_transition_data(
        persona_vectors, window_ids,
        sbert_drift, lda_drift, tfidf_drift, mmd_drift,
    )
    with torch.no_grad():
        scores = model(
            torch.FloatTensor(V_t).to(device),
            torch.FloatTensor(V_t1).to(device),
            torch.FloatTensor(baselines).to(device),
        ).cpu().numpy()

    sep_after = float(scores[mask_shift].mean() - scores[mask_stable].mean())
    return sep_before, sep_after
