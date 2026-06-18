import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


def _candidate_local_dinov2_repos():
    candidates = []
    env_path = os.environ.get("DINOV2_REPO_DIR", "").strip()
    if env_path:
        candidates.append(env_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(repo_root, "dinov2"))

    hub_dir = torch.hub.get_dir()
    candidates.extend(
        [
            os.path.join(hub_dir, "facebookresearch_dinov2_main"),
            os.path.join(hub_dir, "facebookresearch_dinov2_master"),
        ]
    )
    out = []
    seen = set()
    for p in candidates:
        p = os.path.abspath(os.path.expanduser(p))
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_dinov2_model(model_name):
    local_errors = []
    for repo_dir in _candidate_local_dinov2_repos():
        hubconf = os.path.join(repo_dir, "hubconf.py")
        if not os.path.exists(hubconf):
            continue
        try:
            print(f"[DINOv2] loading local repo: {repo_dir} model={model_name}", flush=True)
            return torch.hub.load(repo_dir, model_name, source="local")
        except Exception as e:
            local_errors.append(f"{repo_dir}: {repr(e)}")
            print(f"[DINOv2][warn] local load failed: {repo_dir} err={repr(e)}", flush=True)
    if local_errors:
        print("[DINOv2][warn] falling back to remote torch.hub.load after local failures", flush=True)
    else:
        print("[DINOv2] no local repo found; falling back to remote torch.hub.load", flush=True)
    return torch.hub.load("facebookresearch/dinov2", model_name)


class CropResizePad:
    """Minimal crop-resize-pad utility copied for standalone matching."""

    def __init__(self, target_size, pad_value=0.0):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)
        self.pad_value = pad_value

    def __call__(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # images: [N, C, H, W], boxes: [N, 4] in XYXY
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            x1, y1, x2, y2 = box.long()
            image = image[:, y1:y2, x1:x2]
            image = F.interpolate(
                image.unsqueeze(0),
                scale_factor=scale.item(),
                mode="bilinear",
                align_corners=False,
            )[0]

            h, w = image.shape[1:]
            pad_top = max((self.target_h - h) // 2, 0)
            pad_bottom = self.target_h - h - pad_top
            pad_left = max((self.target_w - w) // 2, 0)
            pad_right = self.target_w - w - pad_left
            image = F.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                value=self.pad_value,
            )

            if image.shape[1] != self.target_h or image.shape[2] != self.target_w:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.target_h, self.target_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


class DINOv2CADMatcher:
    """
    Minimal DINOv2 matcher for CAD-template matching.

    Supports:
    - CAD template descriptor extraction
    - mask proposal descriptor extraction
    - cosine matching + aggregation + threshold filtering
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        proposal_size: int = 224,
        chunk_size: int = 16,
        device: str | None = None,
        background_mean_fill: bool = True,
        use_multi_layer_fusion: bool = True,
        fusion_layers: int = 8,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_dinov2_model(model_name).to(self.device)
        self.model.eval()
        self.chunk_size = chunk_size
        self.background_mean_fill = background_mean_fill
        self.use_multi_layer_fusion = use_multi_layer_fusion
        self.fusion_layers = fusion_layers
        self.proposal_processor = CropResizePad(proposal_size)
        self.imagenet_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(1, 3, 1, 1)
        self.imagenet_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(1, 3, 1, 1)
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def _compute_features_chunk(self, images: torch.Tensor) -> torch.Tensor:
        # images: [N, 3, H, W], return [N, D]
        final_feat = self.model(images)
        if not self.use_multi_layer_fusion:
            return final_feat

        if not hasattr(self.model, "get_intermediate_layers"):
            return final_feat

        try:
            inter = self.model.get_intermediate_layers(
                images,
                n=self.fusion_layers,
                reshape=False,
                return_class_token=True,
            )
            cls_tokens = []
            for item in inter:
                # Common DINOv2 format: tuple(patch_tokens, cls_token)
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    cls = item[1]
                else:
                    # Fallback: token sequence, use first token as cls
                    cls = item[:, 0]
                cls_tokens.append(cls)

            if len(cls_tokens) == 0:
                return final_feat

            inter_feat = torch.stack(cls_tokens, dim=0).mean(dim=0)  # [N, D]
            final_feat = F.normalize(final_feat, dim=-1)
            inter_feat = F.normalize(inter_feat, dim=-1)
            fused = F.normalize(0.6 * final_feat + 0.4 * inter_feat, dim=-1)
            return fused
        except Exception:
            return final_feat

    @torch.no_grad()
    def _compute_features(self, images: torch.Tensor) -> torch.Tensor:
        # images: [N, 3, H, W], chunk-safe wrapper
        if images.shape[0] <= self.chunk_size:
            return self._compute_features_chunk(images)

        outputs = []
        for i in range(0, images.shape[0], self.chunk_size):
            outputs.append(self._compute_features_chunk(images[i : i + self.chunk_size]))
        return torch.cat(outputs, dim=0)

    def _normalize_tensor(self, rgbs: torch.Tensor) -> torch.Tensor:
        mean = self.imagenet_mean.to(device=rgbs.device, dtype=rgbs.dtype)
        std = self.imagenet_std.to(device=rgbs.device, dtype=rgbs.dtype)
        return (rgbs - mean) / std

    def _fill_background_with_imagenet_mean(
        self,
        rgbs: torch.Tensor,
        masks: torch.Tensor,
        normalized: bool,
    ) -> torch.Tensor:
        """
        rgbs: [N, 3, H, W]
        masks: [N, H, W] in [0,1]
        Fill background pixels with ImageNet mean color.

        In normalized tensor space ImageNet mean is 0. In raw [0,1] RGB
        tensor space it is (0.485, 0.456, 0.406).
        """
        m = masks.unsqueeze(1).clamp(0.0, 1.0)
        if normalized:
            bg_value = torch.zeros((1, rgbs.shape[1], 1, 1), device=rgbs.device, dtype=rgbs.dtype)
        else:
            bg_value = self.imagenet_mean.to(device=rgbs.device, dtype=rgbs.dtype)
        return rgbs * m + bg_value * (1.0 - m)

    @torch.no_grad()
    def encode_templates(
        self,
        templates: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode CAD templates to DINOv2 descriptors.

        Args:
            templates: [N_template, 3, H, W], RGB in [0, 1]
            boxes: [N_template, 4], XYXY
            masks: optional [N_template, H, W], ROI mask in {0,1}
        Returns:
            [N_template, D]
        """
        templates = templates.to(self.device).float()
        boxes = boxes.to(self.device)
        if masks is not None:
            masks = masks.to(self.device).float()
            if self.background_mean_fill:
                templates = self._fill_background_with_imagenet_mean(templates, masks, normalized=False)
            else:
                templates = self._fill_background_with_imagenet_mean(templates, masks, normalized=False)
        templates = self._normalize_tensor(templates)
        cropped = self.proposal_processor(templates, boxes)
        return self._compute_features(cropped)

    @torch.no_grad()
    def encode_proposals(
        self,
        image_np: np.ndarray,
        masks: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode mask proposals from one RGB image.

        Args:
            image_np: uint8 HxWx3
            masks: [N_prop, H, W], binary/float
            boxes: [N_prop, 4], XYXY
        Returns:
            [N_prop, D]
        """
        rgb = self.rgb_normalize(image_np).to(self.device).float()
        masks = masks.to(self.device).float()
        boxes = boxes.to(self.device)

        rgbs = rgb.unsqueeze(0).repeat(masks.shape[0], 1, 1, 1)
        if self.background_mean_fill:
            masked_rgbs = self._fill_background_with_imagenet_mean(rgbs, masks, normalized=True)
        else:
            masked_rgbs = self._fill_background_with_imagenet_mean(rgbs, masks, normalized=True)
        cropped = self.proposal_processor(masked_rgbs, boxes)
        return self._compute_features(cropped)

    @staticmethod
    @torch.no_grad()
    def pairwise_similarity(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        query: [N_query, D]
        reference: [N_obj, N_template, D]
        return: [N_query, N_obj, N_template]
        """
        n_query = query.shape[0]
        n_obj, n_template = reference.shape[0], reference.shape[1]

        refs = reference.unsqueeze(0).repeat(n_query, 1, 1, 1)
        qs = query.unsqueeze(1).repeat(1, n_template, 1)
        qs = F.normalize(qs, dim=-1)
        refs = F.normalize(refs, dim=-1)

        sims = []
        for idx_obj in range(n_obj):
            sim = F.cosine_similarity(qs, refs[:, idx_obj], dim=-1)  # [N_query, N_template]
            sims.append(sim)
        sims = torch.stack(sims, dim=0).permute(1, 0, 2)  # [N_query, N_obj, N_template]
        return sims.clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def match(
        self,
        query_desc: torch.Tensor,
        reference_desc: torch.Tensor,
        aggregation_function: str = "max",
        confidence_thresh: float = 0.15,
        max_num_instances: int = 100,
    ):
        """
        Match proposals with CAD template descriptors.

        Args:
            query_desc: [N_query, D]
            reference_desc: [N_obj, N_template, D]
        Returns:
            idx_selected_proposals, pred_idx_objects, pred_scores, pred_score_distribution
        """
        scores = self.pairwise_similarity(query_desc, reference_desc)

        if aggregation_function == "mean":
            score_per_proposal_and_object = scores.mean(dim=-1)
        elif aggregation_function == "median":
            score_per_proposal_and_object = scores.median(dim=-1)[0]
        elif aggregation_function == "max":
            score_per_proposal_and_object = scores.max(dim=-1)[0]
        elif aggregation_function == "avg_5":
            k = min(5, scores.shape[-1])
            score_per_proposal_and_object = torch.topk(scores, k=k, dim=-1)[0].mean(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported aggregation_function: {aggregation_function}")

        pred_scores, pred_idx_objects = torch.max(score_per_proposal_and_object, dim=-1)
        idx_selected_proposals = torch.arange(
            len(pred_scores), device=pred_scores.device
        )[pred_scores > confidence_thresh]

        if len(idx_selected_proposals) > max_num_instances:
            _, idx = torch.topk(pred_scores[idx_selected_proposals], k=max_num_instances)
            idx_selected_proposals = idx_selected_proposals[idx]

        return (
            idx_selected_proposals,
            pred_idx_objects[idx_selected_proposals],
            pred_scores[idx_selected_proposals],
            score_per_proposal_and_object[idx_selected_proposals],
        )

if __name__ == "__main__":
    # Minimal usage example for CAD-template matching with DINOv2.
    # Replace the random tensors below with your real templates/proposals.

    matcher = DINOv2CADMatcher(
        model_name="dinov2_vitl14",
        proposal_size=224,
        chunk_size=16,
        background_mean_fill=True,
        use_multi_layer_fusion=True,
        fusion_layers=4,
    )

    # Example CAD templates for 2 objects, each object has 4 templates.
    n_obj, n_template, h, w = 2, 4, 256, 256
    all_template_desc = []
    for _ in range(n_obj):
        templates = torch.rand(n_template, 3, h, w)  # float in [0, 1]
        template_boxes = torch.tensor([[20, 20, 220, 220]] * n_template)
        desc = matcher.encode_templates(templates, template_boxes)  # [N_template, D]
        all_template_desc.append(desc)
    reference_desc = torch.stack(all_template_desc, dim=0)  # [N_obj, N_template, D]

    # Example proposals from one RGB image.
    image_np = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    n_prop = 6
    proposal_masks = (torch.rand(n_prop, h, w) > 0.5).float()
    proposal_boxes = torch.tensor(
        [
            [10, 10, 120, 140],
            [60, 40, 200, 220],
            [30, 80, 180, 240],
            [100, 20, 250, 170],
            [40, 30, 150, 160],
            [70, 70, 210, 230],
        ]
    )
    query_desc = matcher.encode_proposals(image_np, proposal_masks, proposal_boxes)  # [N_prop, D]

    idx_selected, pred_obj_idx, pred_scores, score_dist = matcher.match(
        query_desc=query_desc,
        reference_desc=reference_desc,
        aggregation_function="max",
        confidence_thresh=0.2,
        max_num_instances=100,
    )

    print("Selected proposal idx:", idx_selected.tolist())
    print("Predicted object idx:", pred_obj_idx.tolist())
    print("Scores:", pred_scores.tolist())
    print("Score distribution shape:", tuple(score_dist.shape))
