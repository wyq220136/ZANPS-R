import os
from concurrent.futures import ProcessPoolExecutor


def parse_gpu_ids(raw):
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_worker_counts(raw, gpu_ids, fallback_per_gpu):
    if not gpu_ids:
        return []

    text = str(raw or "").strip()
    if text:
        parts = [x.strip() for x in text.split(",") if x.strip()]
        if not parts:
            raise ValueError("--sam-workers-per-gpu was set but no counts were parsed.")
        if len(parts) == 1:
            counts = [int(parts[0])] * len(gpu_ids)
        elif len(parts) == len(gpu_ids):
            counts = [int(x) for x in parts]
        else:
            raise ValueError(
                "--sam-workers-per-gpu must be one integer or have the same "
                f"number of entries as --sam-gpu-ids ({len(gpu_ids)})."
            )
    else:
        counts = [max(1, int(fallback_per_gpu))] * len(gpu_ids)

    if any(c < 0 for c in counts):
        raise ValueError("--sam-workers-per-gpu values must be non-negative.")
    if sum(counts) <= 0:
        raise ValueError("At least one SAM worker is required across selected GPUs.")
    return counts


def _interleaved_gpu_slots(gpu_ids, worker_counts):
    slots = []
    for idx in range(max(worker_counts)):
        for gpu_id, count in zip(gpu_ids, worker_counts):
            if idx < count:
                slots.append(str(gpu_id))
    return slots


def _chunk_round_robin(items, num_chunks):
    chunks = [[] for _ in range(max(1, int(num_chunks)))]
    for idx, item in enumerate(items):
        chunks[idx % len(chunks)].append(item)
    return [c for c in chunks if c]


def _load_sam_utils():
    try:
        from segmentation.sam_utils import create_mask_generator, generate_candidate_masks, save_mask
    except ImportError:
        from sam_utils import create_mask_generator, generate_candidate_masks, save_mask
    return create_mask_generator, generate_candidate_masks, save_mask


def _run_sam_frame(image_path, ext_mask_path, pred_frame_dir, cfg, mask_generator):
    _, _, candidates = _RUNNER_GENERATE_CANDIDATE_MASKS(
        image_path=image_path,
        mask_path=ext_mask_path if ext_mask_path and os.path.exists(ext_mask_path) else None,
        model_cfg=cfg["model_cfg"],
        sam2_checkpoint=cfg["sam2_checkpoint"],
        sam_checkpoint=cfg["sam_checkpoint"],
        sam_model_type=cfg["sam_model_type"],
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=cfg["pred_iou_thresh"],
        stability_score_thresh=cfg["stability_score_thresh"],
        min_mask_region_area=cfg["min_mask_region_area"],
        iou_threshold=cfg["duplicate_iou_threshold"],
        mask_generator=mask_generator,
    )

    os.makedirs(pred_frame_dir, exist_ok=True)
    for name in os.listdir(pred_frame_dir):
        if name.startswith("mask_") and name.lower().endswith(".png"):
            path = os.path.join(pred_frame_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    for i, ann in enumerate(candidates):
        _RUNNER_SAVE_MASK(ann["segmentation"].astype(bool), os.path.join(pred_frame_dir, f"mask_{i}.png"))
    return len(candidates)


def _sam_chunk_worker(payload):
    gpu_id, tasks, worker_cfg = payload
    if gpu_id not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    global _RUNNER_GENERATE_CANDIDATE_MASKS, _RUNNER_SAVE_MASK
    create_mask_generator, _RUNNER_GENERATE_CANDIDATE_MASKS, _RUNNER_SAVE_MASK = _load_sam_utils()
    mask_generator = create_mask_generator(
        model_cfg=worker_cfg["model_cfg"],
        sam2_checkpoint=worker_cfg["sam2_checkpoint"],
        sam_checkpoint=worker_cfg["sam_checkpoint"],
        sam_model_type=worker_cfg["sam_model_type"],
        points_per_side=worker_cfg["points_per_side"],
        points_per_batch=worker_cfg["points_per_batch"],
        pred_iou_thresh=worker_cfg["pred_iou_thresh"],
        stability_score_thresh=worker_cfg["stability_score_thresh"],
        min_mask_region_area=worker_cfg["min_mask_region_area"],
    )

    out = []
    for obj_name, frame_id, image_path, ext_mask_path, pred_frame_dir in tasks:
        num_cand = _run_sam_frame(
            image_path=image_path,
            ext_mask_path=ext_mask_path,
            pred_frame_dir=pred_frame_dir,
            cfg=worker_cfg,
            mask_generator=mask_generator,
        )
        out.append((obj_name, frame_id, num_cand, "" if gpu_id is None else str(gpu_id)))
    return out


def run_sam_tasks(
    tasks,
    worker_cfg,
    num_workers=1,
    task_chunksize=1,
    sam_gpu_ids="",
    sam_procs_per_gpu=1,
    sam_workers_per_gpu="",
):
    if not tasks:
        return []

    gpu_ids = parse_gpu_ids(sam_gpu_ids)
    if gpu_ids:
        counts = _parse_worker_counts(sam_workers_per_gpu, gpu_ids, sam_procs_per_gpu)
        slots = _interleaved_gpu_slots(gpu_ids, counts)
    else:
        slots = [""] * max(1, int(num_workers))

    slots = slots[: max(1, min(len(slots), len(tasks)))]
    chunks = _chunk_round_robin(tasks, len(slots))
    payloads = [(slots[i % len(slots)], chunk, worker_cfg) for i, chunk in enumerate(chunks)]

    if len(payloads) == 1:
        flat = _sam_chunk_worker(payloads[0])
    else:
        flat = []
        with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            for part in executor.map(_sam_chunk_worker, payloads, chunksize=max(1, int(task_chunksize))):
                flat.extend(part)
    return flat
