from __future__ import annotations

import itertools
import json
import logging
import os
import random
import re
import shutil
import multiprocessing as mp
from os.path import join, exists
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import datasets
import polars as pl
from jinja2 import Template
from PIL import Image
from torchvision.transforms import functional as VF
from tqdm import tqdm

from olmo import tokenizer
from olmo.data.dataset import (
    WEB_DATA_HOME,
    Dataset,
    DatasetBase,
)
from olmo.data.pixmo_datasets import save_local_dataset
from olmo.util import split_into_groups

"""Where to save local version of the data after URLs filtering"""

RNG_SEED = 42
random.seed(RNG_SEED)
WEB_GROUNDING_TEMPLATES = [
    'click "{description}".',
    'Click "{description}".',
    'Click on the element "{description}".',
    'Click the "{description}" element.',
    'Select "{description}".'
    'Find the element: "{description}" and click on it.',
    "Click on {description}.",
    "Click on the element that matches the description: {description}",
]


def normalize_click_coords(
    x, y, image_w, image_h, upper_bound=100, num_digits=1
):
    """
    Normalize the coordinates to [0, upper_bound]
    Args:
        x: the x coordinate
        y: the y coordinate
        image_w: the width of the image
        image_h: the height of the image
        upper_bound: the upper bound of the normalized coordinates
        num_digits: the number of digits to round to
    Returns:
        x: the normalized x coordinate
        y: the normalized y coordinate
    """
    x = round(x / image_w * upper_bound, num_digits)
    y = round(y / image_h * upper_bound, num_digits)
    # add min and max clipping to ensure normalized coords are between 0 and upperbound
    x = max(0, min(x, upper_bound))
    y = max(0, min(y, upper_bound))
    return x, y


def normalize_scroll_deltas(
    delta_x, delta_y, image_w, image_h, upper_bound=100, num_digits=1
):
    def _normalize(delta, dim):
        if dim == 0:
            return 0.0  # avoid divide by zero
        normalized = abs(delta) / dim * upper_bound
        normalized = round(normalized, num_digits)
        return normalized if delta >= 0 else -normalized

    norm_x = _normalize(delta_x, image_w)
    norm_y = _normalize(delta_y, image_h)
    return norm_x, norm_y


def gaussian_sample_around_bbox_center(
    bbox, margin_percent=0.1, max_dist_from_center=None
):
    """
    Generate a random point around the center of a bounding box using Gaussian distribution.
    The standard deviation is set to 1/3 * (1 - margin_percent) of the distance to boundaries.
    The sampled x and y are optionally clipped to be within max_dist_from_center after sampling.
    Args:
        bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
        tuple: A randomly sampled (x, y) coordinate.
    """
    x_min, y_min, x_max, y_max = bbox

    # Compute the center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    half_width = (x_max - x_min) / 2
    half_height = (y_max - y_min) / 2

    x_margin = (
        1 - margin_percent
    ) * half_width  # sample x within (1 - margin_percent) of half_width
    y_margin = (
        1 - margin_percent
    ) * half_height  # sample y within (1 - margin_percent) of half_height

    x_sigma = x_margin / 3
    y_sigma = y_margin / 3

    # Sample from Gaussian distribution centered at the center
    sampled_x = random.gauss(center_x, x_sigma)
    sampled_y = random.gauss(center_y, y_sigma)

    if (
        max_dist_from_center
    ):  # clip x, y based on max_dist_from_center if provided
        sampled_x = min(sampled_x, center_x + max_dist_from_center)
        sampled_x = max(sampled_x, center_x - max_dist_from_center)

        sampled_y = min(sampled_y, center_y + max_dist_from_center)
        sampled_y = max(sampled_y, center_y - max_dist_from_center)

    return (sampled_x, sampled_y)


def get_click_coords_from_bbox(bbox, mode="center"):
    """
    Get the click point from the bounding box.
    Args:
        bbox: a list of four coordinates [x1, x2, y1, y2]
        mode: the mode to get the click point. "center", "top_left", "random_gaussian", or "random_uniform"
    Returns:
        x: the x coordinate of the click point
        y: the y coordinate of the click point
    """
    assert len(bbox) == 4, f"Invalid bbox: {bbox}"

    x1, y1, x2, y2 = bbox
    assert x2 >= x1 and y2 >= y1, f"Invalid bbox values: {bbox}"

    if mode == "top_left":
        return x1, y1
    elif mode == "center":
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return x, y
    elif mode == "random_uniform":
        x = random.uniform(x1, x2)
        y = random.uniform(y1, y2)
        return x, y
    elif mode == "random_gaussian":
        x, y = gaussian_sample_around_bbox_center(bbox)
        return x, y
    else:
        raise ValueError(f"Unknown mode: {mode}")


def format_elem_description(elem_type: str = "", elem_content: str = ""):
    if len(elem_content) > 0 and len(elem_type) > 0:
        templates = [
            "{elem_type} with content: {elem_content}",
            "{elem_type}: {elem_content}",
            "{elem_type} that says {elem_content}",
            "{elem_content} {elem_type}",
        ]
    elif len(elem_type) > 0:
        templates = [
            "{elem_type} element",
            "{elem_type}",
            "element of type {elem_type}",
            "A {elem_type} element",
        ]
    elif len(elem_content) > 0:
        templates = [
            "element with content: {elem_content}",
            "element that says {elem_content}",
            "{elem_content} element",
            "an element that contains {elem_content}",
        ]
    else:
        raise ValueError(f"Either element type or content must be non-empty str: {elem_type}, {elem_content}")

    description = random.choice(templates).format(
        elem_type=elem_type, elem_content=elem_content
    )
    return description


class ScreenSpot(DatasetBase):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=None, cache_only=False):
        local_name = join(WEB_DATA_HOME, "screenspot")
        img_dir = join(local_name, "imgs")
        if os.path.exists(local_name) and os.path.exists(img_dir):
            logging.info("ScreenSpot already downloaded, skipping.")
            return
        if not os.path.exists(local_name):
            logging.info("Downloading ScreenSpot from rootsautomation/ScreenSpot...")
            ds = datasets.load_dataset("rootsautomation/ScreenSpot")
            save_local_dataset(ds, local_name, n_procs, n_val=n_val)
            logging.info(f"ScreenSpot saved to {local_name}.")
        else:
            ds = datasets.load_from_disk(local_name)
        logging.info(f"Extracting images to {img_dir}...")
        os.makedirs(img_dir, exist_ok=True)
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"Saving {split_name} images"):
                row["image"].save(join(img_dir, row["file_name"]))
        logging.info(f"Images saved to {img_dir}.")

    def __init__(self, split, sample=None, keep_in_memory=False):
        self.data = datasets.load_from_disk(
            join(WEB_DATA_HOME, "screenspot"), keep_in_memory=keep_in_memory
        )[split]
        self.image_dir = join(WEB_DATA_HOME, "screenspot", "imgs")
        assert split in ["test"], f"Invalid split: {split}"
        self.raw_data = datasets.load_from_disk(
            join(WEB_DATA_HOME, "screenspot"), keep_in_memory=keep_in_memory
        )[split]
        super().__init__(split, sample)

    def load(self):
        formatted_data = []
        for i in range(len(self.raw_data)):
            item = self.raw_data[i]
            image = item["file_name"]
            image = os.path.join(self.image_dir, image)
            if not os.path.exists(image):
                raise ValueError(f"Image not found: {image}")
            bbox = [round(coord * 100, 1) for coord in item["bbox"]]
            formatted_item = dict(
                image=image,
                question=item["instruction"],
                answer=str(bbox),
                task_description=item["instruction"],
                style="web_grounding",
                metadata=dict(
                    sample_id=f"{i}",
                    data_type=item["data_type"],
                    data_source=item["data_source"],
                    bbox=bbox,
                ),
            )
            formatted_data.append(formatted_item)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        return self.data[item]


class ScreenSpotV2(DatasetBase):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=None, cache_only=False):
        from huggingface_hub import snapshot_download
        local_name = join(WEB_DATA_HOME, "screenspot_v2")
        img_dir = join(local_name, "imgs")
        ann_dir = join(local_name, "test")
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            logging.info("ScreenSpotV2 already downloaded, skipping.")
            return
        logging.info("Downloading ScreenSpotV2 from likaixin/ScreenSpot-v2-variants...")
        snapshot_path = snapshot_download("likaixin/ScreenSpot-v2-variants", repo_type="dataset")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        for subset in ["web", "mobile", "desktop"]:
            shutil.copy2(join(snapshot_path, "annotations", f"{subset}.json"), join(ann_dir, f"{subset}.json"))
        src_img_dir = join(snapshot_path, "images")
        for fname in tqdm(os.listdir(src_img_dir), desc="Copying ScreenSpotV2 images"):
            shutil.copy2(join(src_img_dir, fname), join(img_dir, fname))
        logging.info(f"ScreenSpotV2 saved to {local_name}.")

    def __init__(self, split, subset="all", sample=None, keep_in_memory=False):
        assert subset in [
            "all",
            "web",
            "mobile",
            "desktop",
        ], f"Invalid subset: {subset}"
        assert split in ["test"], f"Invalid split: {split}"
        self.subset = subset
        self.image_dir = join(WEB_DATA_HOME, "screenspot_v2", "imgs")
        super().__init__(split, sample)

    def get(self, item, rng):
        return self.data[item]

    def load(self):
        raw_data = []
        if self.subset == "all":
            for self.subset in ["web", "mobile", "desktop"]:
                data_path = join(
                    WEB_DATA_HOME,
                    "screenspot_v2",
                    self.split,
                    f"{self.subset}.json",
                )
                data = json.load(open(data_path, "r"))
                raw_data += data
        else:
            data_path = join(
                WEB_DATA_HOME,
                "screenspot_v2",
                self.split,
                f"{self.subset}.json",
            )
            data = json.load(open(data_path, "r"))
            raw_data = data

        formatted_data = []
        for i in range(len(raw_data)):
            item = raw_data[i]
            image = item["img_filename"]
            full_image_path = os.path.join(self.image_dir, image)

            image_w, image_h = item["img_size"]
            image = full_image_path
            if not os.path.exists(image):
                raise ValueError(f"Image not found: {image}")
            bbox = item["bbox"]
            bbox = [
                bbox[0] / image_w,
                bbox[1] / image_h,
                bbox[2] / image_w,
                bbox[3] / image_h,
            ]
            bbox = [round(coord * 100, 1) for coord in bbox]

            formatted_item = dict(
                image=image,
                question=f"{item['instruction']}",
                answer=str(bbox),
                task_description=item["instruction"],
                style="web_grounding",
                metadata=dict(
                    sample_id=item["id"],
                    data_type=item["ui_type"],
                    data_source=item["application"],
                    platform=item["platform"],
                    bbox=bbox,
                ),
            )
            formatted_data.append(formatted_item)
        return formatted_data



def _save_hf_images_to_disk(hf_dataset, image_dir: str) -> dict:
    """Save each row's embedded image bytes to {image_dir}/{sha256[:16]}.png.

    Filename is the first 16 hex chars of the SHA-256 of the raw bytes, making
    it stable across re-downloads and naturally deduplicating identical images.
    Skips files that already exist so re-runs are safe.

    Returns a dict mapping original row index → absolute file path, suitable
    for serialising to image_index.json.
    """
    import hashlib
    os.makedirs(image_dir, exist_ok=True)
    index: dict[int, str] = {}
    no_decode = hf_dataset.cast_column("image", datasets.Image(decode=False))
    for i, row in enumerate(tqdm(no_decode, desc="Saving images to disk")):
        raw = row["image"]["bytes"]
        if raw:
            img_hash = hashlib.sha256(raw).hexdigest()
            img_path = join(image_dir, f"{img_hash}.png")
            if not exists(img_path):
                with open(img_path, "wb") as f:
                    f.write(raw)
            index[i] = img_path
    return index


def _save_traj_images_to_disk(rows, image_dir: str) -> None:
    """Save per-screenshot images from trajectory rows to disk.

    Each image is written to {image_dir}/{sample_id}/{screenshot_name}.
    Rows are expected to have 'sample_id', 'images', and optionally
    'image_paths' fields (same schema as the Polars/HF trajectory datasets).
    Skips files that already exist so re-runs are safe.
    """
    os.makedirs(image_dir, exist_ok=True)
    for row in tqdm(rows, desc="Saving trajectory images to disk"):
        sample_id = row["sample_id"]
        raw_images = row.get("images") or []
        paths = row.get("image_paths") or []
        if not paths:
            paths = [f"screenshot_{i+1:03d}.png" for i in range(len(raw_images))]
        sample_dir = join(image_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        for path, img in zip(paths, raw_images):
            img_bytes = img["bytes"] if isinstance(img, dict) else img
            out_path = join(sample_dir, Path(path).name)
            if not exists(out_path) and img_bytes:
                with open(out_path, "wb") as f:
                    f.write(img_bytes)


def _iter_polars_parquet(hf_source: str, split: str, hf_shards: list | None = None):
    """
    Yield rows from HuggingFace parquet files using Polars.

    Workaround for PyArrow 19's ``Repetition level histogram size mismatch``
    crash on list<binary> columns (e.g. the ``images`` column in skill datasets).
    Images are kept as raw bytes in each row dict.

    Args:
        hf_source: HuggingFace repo id (e.g. "allenai/MolmoWeb-SyntheticSkills")
        split: dataset split (e.g. "train")
        hf_shards: explicit list of shard paths (e.g. ["data/train-00000.parquet"]).
            If None, the repo file list is fetched from HuggingFace (may be slow).
    """
    import polars as pl
    from huggingface_hub import list_repo_files, hf_hub_download

    if hf_shards is not None:
        shard_paths = hf_shards
    else:
        repo_files = list(list_repo_files(hf_source, repo_type="dataset"))
        prefix = f"data/{split}-"
        shard_paths = sorted(f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"No parquet shards found for {hf_source} split={split}")

    from concurrent.futures import ThreadPoolExecutor

    def _download_and_read(shard):
        local_path = hf_hub_download(hf_source, shard, repo_type="dataset")
        return pl.read_parquet(local_path)

    n_threads = min(8, len(shard_paths))
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        dfs = list(ex.map(_download_and_read, shard_paths))

    df = pl.concat(dfs, rechunk=False)
    for row in df.iter_rows(named=True):
        yield row


def _load_hf_dataset(hf_source, split, local_name, config=None, drop_columns=None):
    """Load an HF dataset, caching locally under WEB_DATA_HOME.

    If a local copy exists at WEB_DATA_HOME/{local_name}, loads from disk.
    Otherwise downloads from HF and saves locally.

    Args:
        drop_columns: column names to strip before saving to disk (e.g. image
            columns whose bytes have already been extracted to individual files).
            The returned dataset always contains all columns.
    """
    local_dir = join(WEB_DATA_HOME, local_name) if WEB_DATA_HOME else None

    if local_dir and exists(local_dir):
        logging.info(f"Loading {hf_source} config={config} split={split} from {local_dir}")
        return datasets.load_from_disk(local_dir)

    logging.info(f"Downloading {hf_source} config={config} split={split} from HuggingFace")
    def _is_hf_load_error(exc):
        """Check if exc or any cause is a known HF dataset loading error we can work around."""
        _MARKERS = (
            "CastError",
        )
        _MSG_MARKERS = (
            "column names don't match",
            "Repetition level histogram size mismatch",
        )
        seen = set()
        cur = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if any(m in type(cur).__name__ for m in _MARKERS):
                return True
            msg = str(cur)
            if any(m in msg for m in _MSG_MARKERS):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    try:
        ds = datasets.load_dataset(hf_source, config, split=split, verification_mode="no_checks")
    except Exception as e:
        if _is_hf_load_error(e):
            # Stale cache is the most common cause — retry with a fresh download first.
            logging.warning(
                f"Stale cache detected for {hf_source} — retrying with force_redownload. "
                f"Original error: {e}"
            )
            try:
                ds = datasets.load_dataset(
                    hf_source, config, split=split, download_mode="force_redownload",
                    verification_mode="no_checks",
                )
            except Exception as e2:
                if not _is_hf_load_error(e2):
                    raise
                # force_redownload also failed; fall through to manual snapshot path.
                logging.warning(
                    f"force_redownload also failed for {hf_source} — "
                    f"falling back to manual snapshot load. Error: {e2}"
                )
                e = e2
            else:
                e = None  # retry succeeded

        if e is not None and _is_hf_load_error(e):
            from huggingface_hub import snapshot_download, list_repo_files
            import glob as _glob
            import re as _re
            import collections as _collections

            all_files = list(list_repo_files(hf_source, repo_type="dataset"))

            # --- Parquet repo (e.g. MolmoWeb-SyntheticTrajs) ---
            # Files are data/{config}-NNNNN.parquet with no nested split dir.
            parquet_prefix = f"data/{config}-" if config else "data/"
            parquet_names = [
                f for f in all_files
                if f.startswith(parquet_prefix) and f.endswith(".parquet")
            ]
            if parquet_names:
                logging.warning(
                    f"HF dataset builder failed for {hf_source} config={config} "
                    f"— falling back to Polars parquet load. Original error: {e}"
                )
                ds = datasets.Dataset.from_list(
                    list(_iter_polars_parquet(hf_source, split, hf_shards=parquet_names))
                )

            # --- Arrow repo (e.g. MolmoWeb-SyntheticQA) ---
            # Multiple shard-count sets; pick the largest (most up-to-date files).
            else:
                logging.warning(
                    f"Schema mismatch in HF repo for {hf_source} — falling back to "
                    f"direct arrow load of largest shard set. Original error: {e}"
                )
                split_prefix = f"{split}/" if config is None else f"{config}/{split}/"
                arrow_names = [f for f in all_files if f.startswith(split_prefix) and f.endswith(".arrow")]
                by_count = _collections.defaultdict(list)
                for name in arrow_names:
                    m = _re.search(r"-of-(\d+)\.arrow$", name)
                    if m:
                        by_count[int(m.group(1))].append(name)
                if not by_count:
                    raise RuntimeError(f"No arrow or parquet files found for {hf_source} split={split}") from e

                # Probe the first shard of each count (smallest first) to find
                # the set whose schema matches the expected nested format.
                # "Largest count = newest" is not always true after a re-upload.
                import pyarrow as _pa

                def _probe_schema(shard_name):
                    probe_dir = snapshot_download(
                        repo_id=hf_source, repo_type="dataset",
                        allow_patterns=[shard_name],
                    )
                    return _pa.ipc.open_stream(join(probe_dir, shard_name)).schema.names

                best_count = None
                for probe_count in sorted(by_count.keys()):
                    try:
                        names = _probe_schema(sorted(by_count[probe_count])[0])
                        if "messages" in names or "metadata" in names:
                            best_count = probe_count
                            break
                    except Exception:
                        continue
                if best_count is None:
                    best_count = max(by_count)  # fallback: largest count

                target_files = sorted(by_count[best_count])
                logging.info(f"Downloading {len(target_files)} arrow files (of-{best_count:05d}) from {hf_source}")
                repo_dir = snapshot_download(
                    repo_id=hf_source, repo_type="dataset",
                    allow_patterns=target_files,
                )
                arrow_paths = [join(repo_dir, f) for f in target_files]
                ds = datasets.load_dataset("arrow", data_files={split: arrow_paths}, split=split, verification_mode="no_checks")
        elif not _is_hf_load_error(e):
            raise

    if local_dir:
        logging.info(f"Saving to {local_dir}")
        ds_to_save = ds
        if drop_columns:
            cols_present = [c for c in drop_columns if c in ds.column_names]
            if cols_present:
                ds_to_save = ds.remove_columns(cols_present)
        ds_to_save.save_to_disk(local_dir)

    return ds


def _process_ground_row(args):
    """Process one screenshot row (no PIL image). Returns (row_idx, list_of_examples)."""
    (row_idx, metadata, messages,
     action_only, flatten, max_msg_per_screenshot, style) = args

    msgs = []
    results = []
    url = metadata["url"]
    website = metadata["website"]
    image_w, image_h = metadata["image_w"], metadata["image_h"]
    for msg_data in messages:
        question = msg_data["question"]
        answer_dict = json.loads(msg_data["answer"])
        bbox_norm = json.loads(msg_data["bbox"])

        final_answer = (
            json.dumps(answer_dict["action"]) if action_only else json.dumps(answer_dict)
        )
        msg = {
            "question": question,
            "answer": final_answer,
            "style": style,
            "task_description": question,
            "bbox": bbox_norm,
        }
        msgs.append(msg)

        if flatten:
            results.append({
                "question": question,
                "task_description": question,
                "answer": final_answer,
                "style": style,
                "metadata": {
                    "dataset": website,
                    "url": url,
                    "bbox": bbox_norm,
                    "image_w": image_w,
                    "image_h": image_h,
                },
            })

    if not flatten and msgs:
        groups = (
            list(split_into_groups(msgs, max_msg_per_screenshot))
            if max_msg_per_screenshot > 0 and len(msgs) > max_msg_per_screenshot
            else [msgs]
        )
        for group in groups:
            results.append({
                "message_list": group,
                "metadata": {
                    "dataset": website,
                    "url": url,
                    "image_w": image_w,
                    "image_h": image_h,
                },
            })

    return row_idx, results


def _process_qa_row(args):
    """Process one screenshot row for ScreenshotQA (no PIL image). Returns (row_idx, list_of_examples)."""
    row_idx, website, url, messages, style, flat = args

    msgs = []
    results = []
    for qa in messages:
        q_text = qa["question"]
        a_text = qa["answer"]
        answer_action = json.dumps({"name": "send_msg_to_user", "msg": a_text})
        question_type = qa.get("question_type")
        question_form = qa.get("question_form")
        message = {
            "question": q_text,
            "task_description": q_text,
            "answer": answer_action,
            "style": style,
        }
        meta = {
            "url": url,
            "website": website,
            "answer": answer_action,
            "type_of_question": question_type,
            "question_form": question_form,
        }
        if flat:
            results.append({
                "task_description": q_text,
                "message_list": [message],
                "metadata": meta,
            })
        else:
            msgs.append(message)

    if not flat and msgs:
        results.append({
            "message_list": msgs,
            "metadata": {"url": url, "website": website},
        })

    return row_idx, results


class MolmoWebSyntheticGround(DatasetBase):
    """
    Synthetic dataset for web grounding tasks.
    Downloads data directly from HuggingFace: allenai/MolmoWeb-SyntheticGround.
    """

    HF_SOURCE = "allenai/MolmoWeb-SyntheticGround"
    HF_SPLIT = "train"
    HF_CONFIGS = ("template", "gpt")
    LOCAL_NAME = "MolmoWeb-SyntheticGround"

    @classmethod
    def download(cls, n_procs=None):
        for config in cls.HF_CONFIGS:
            local_dir = join(WEB_DATA_HOME, f"{cls.LOCAL_NAME}/{config}")
            logging.info(f"Downloading {cls.LOCAL_NAME} config={config} from {cls.HF_SOURCE}...")
            hf_dataset = _load_hf_dataset(
                cls.HF_SOURCE, cls.HF_SPLIT,
                f"{cls.LOCAL_NAME}/{config}", config=config,
                drop_columns=["image"],
            )
            index_path = join(local_dir, "image_index.json")
            if not exists(index_path):
                img_dir = join(local_dir, "images")
                logging.info(f"Saving images for {cls.LOCAL_NAME} config={config} to {img_dir}...")
                index = _save_hf_images_to_disk(hf_dataset, img_dir)
                with open(index_path, "w") as f:
                    json.dump({str(k): v for k, v in index.items()}, f)
                logging.info(f"Saved {len(index)} images and index to {index_path}.")
            else:
                logging.info(f"Image index already exists for {cls.LOCAL_NAME} config={config}, skipping.")

    def __init__(
        self,
        dataset_names: list[str],
        split: Literal["train"],
        flatten: bool = False,
        action_only: bool = False,
        gpt: bool = False,
        max_msg_per_screenshot: int = -1,
        style: str = "web_grounding",
        n_procs: int = 1,
    ):
        if split != "train":
            raise ValueError(
                f"MolmoWebSyntheticGround only supports split='train' "
                f"(HuggingFace has no val/test split). Got: {split!r}"
            )
        self.dataset_names = dataset_names
        self.split = split
        self.action_only = action_only
        self.flatten = flatten
        self.max_msg_per_screenshot = max_msg_per_screenshot
        self.gpt = gpt
        self.style = style
        self.n_procs = n_procs
        super().__init__(split=self.split)

    def __len__(self):
        return len(self.data)

    def load(self):
        config = "gpt" if self.gpt else "template"

        hf_dataset = _load_hf_dataset(
            self.HF_SOURCE, self.HF_SPLIT,
            f"{self.LOCAL_NAME}/{config}", config=config,
        )

        # Load image index (maps original HF row index → file path)
        index_path = join(WEB_DATA_HOME, f"{self.LOCAL_NAME}/{config}", "image_index.json")
        with open(index_path) as f:
            image_index: dict[int, str] = {int(k): v for k, v in json.load(f).items()}

        # Collect per-row args without images
        print(f"Loading MolmoWebSyntheticGround ({config}) with {self.n_procs} process(es)...")
        args = [
            (i, row["metadata"], row["messages"],
             self.action_only, self.flatten,
             self.max_msg_per_screenshot, self.style)
            for i, row in enumerate(tqdm(hf_dataset, desc="Reading metadata"))
        ]

        # Process rows (JSON parsing + coordinate math) in parallel
        result_map: dict[int, list] = {}
        if self.n_procs > 1:
            with mp.Pool(self.n_procs) as pool:
                for row_idx, examples in tqdm(
                    pool.imap(_process_ground_row, args, chunksize=32),
                    total=len(args), desc="Processing rows",
                ):
                    if examples:
                        result_map[row_idx] = examples
        else:
            for a in tqdm(args, desc="Processing rows"):
                row_idx, examples = _process_ground_row(a)
                if examples:
                    result_map[row_idx] = examples

        # Attach image paths via the stable hash-based index
        formatted_data = []
        for orig_idx in sorted(result_map.keys()):
            img_path = image_index[orig_idx]
            for ex in result_map[orig_idx]:
                ex["image"] = img_path
            formatted_data.extend(result_map[orig_idx])

        return formatted_data

    def get(self, item, rng):
        return self.data[item]


class MolmoWebSyntheticQA(DatasetBase):
    """
    Synthetic screenshot QA dataset.
    Downloads data directly from HuggingFace: allenai/MolmoWeb-SyntheticQA.
    """

    HF_SOURCE = "allenai/MolmoWeb-SyntheticQA"
    HF_SPLIT = "train"
    LOCAL_NAME = "MolmoWeb-SyntheticQA"

    @classmethod
    def download(cls, n_procs=None):
        local_dir = join(WEB_DATA_HOME, cls.LOCAL_NAME)
        logging.info(f"Downloading {cls.LOCAL_NAME} from {cls.HF_SOURCE}...")
        hf_dataset = _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME, drop_columns=["image"])
        index_path = join(local_dir, "image_index.json")
        if not exists(index_path):
            img_dir = join(local_dir, "images")
            logging.info(f"Saving images for {cls.LOCAL_NAME} to {img_dir}...")
            index = _save_hf_images_to_disk(hf_dataset, img_dir)
            with open(index_path, "w") as f:
                json.dump({str(k): v for k, v in index.items()}, f)
            logging.info(f"Saved {len(index)} images and index to {index_path}.")
        else:
            logging.info(f"Image index already exists for {cls.LOCAL_NAME}, skipping.")

    def __init__(
        self,
        split: str,
        *,
        style: str = "screenshot_qa",
        dataset_names: Sequence[str] | None = None,
        flat: bool = False,
        n_procs: int = 1,
    ):
        self.style = style
        self.dataset_names = list(dataset_names) if dataset_names is not None else None
        self.flat = flat
        self.n_procs = n_procs
        super().__init__("train")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize_row(row):
        """Normalize old flat schema → new nested schema if needed."""
        if "messages" in row:
            return row
        # Old flat schema: question/answer/type_of_question/question_form/website/url/...
        return {
            **row,
            "messages": [{"question": row["question"], "answer": row["answer"],
                           "question_type": row.get("type_of_question"),
                           "question_form": row.get("question_form")}],
            "metadata": {"website": row["website"], "url": row["url"]},
        }

    def load(self):
        hf_dataset = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)

        # Normalize schema if dataset was loaded from old flat parquet files
        if "messages" not in hf_dataset.column_names:
            logging.warning("MolmoWeb-SyntheticQA: old flat schema detected, normalizing...")
            hf_dataset = hf_dataset.map(self._normalize_row)

        # Load image index (maps original HF row index → file path)
        index_path = join(WEB_DATA_HOME, self.LOCAL_NAME, "image_index.json")
        with open(index_path) as f:
            image_index: dict[int, str] = {int(k): v for k, v in json.load(f).items()}

        # Collect per-row args without images
        print(f"Loading MolmoWebSyntheticQA with {self.n_procs} process(es)...")
        args = [
            (i, row["metadata"]["website"], row["metadata"]["url"],
             row["messages"], self.style, self.flat)
            for i, row in enumerate(tqdm(hf_dataset, desc="Reading metadata"))
        ]

        # Process rows in parallel
        result_map: dict[int, list] = {}
        if self.n_procs > 1:
            with mp.Pool(self.n_procs) as pool:
                for row_idx, examples in tqdm(
                    pool.imap(_process_qa_row, args, chunksize=32),
                    total=len(args), desc="Processing rows",
                ):
                    if examples:
                        result_map[row_idx] = examples
        else:
            for a in tqdm(args, desc="Processing rows"):
                row_idx, examples = _process_qa_row(a)
                if examples:
                    result_map[row_idx] = examples

        # Attach image paths via the stable hash-based index
        formatted: List[Dict[str, Any]] = []
        for orig_idx in sorted(result_map.keys()):
            img_path = image_index[orig_idx]
            for ex in result_map[orig_idx]:
                ex["image"] = img_path
            formatted.extend(result_map[orig_idx])

        return formatted

    def get(self, idx, rng):
        return self.data[idx]


class MolmoWebSyntheticTrajs(DatasetBase):
    """
    Synthetic trajectory dataset.
    Downloads data directly from HuggingFace: allenai/MolmoWeb-SyntheticTrajs.
    """

    HF_SOURCE = "allenai/MolmoWeb-SyntheticTrajs"
    HF_SPLIT = "train"
    HF_CONFIGS = ("from_template", "task_seeded_wv", "task_seeded_om2w", "multi_agent", "node_traversal")
    LOCAL_NAME = "MolmoWeb-SyntheticTrajs"
    DATASET_TAG: str | None = None  # if set, written into metadata["dataset"] for each step

    @classmethod
    def download(cls, n_procs=None):
        for config in cls.HF_CONFIGS:
            logging.info(f"Downloading {cls.LOCAL_NAME} config={config} from {cls.HF_SOURCE}...")
            _load_hf_dataset(
                cls.HF_SOURCE, cls.HF_SPLIT,
                f"{cls.LOCAL_NAME}/{config}", config=config,
                drop_columns=["images"],
            )
        img_dir = join(WEB_DATA_HOME, cls.LOCAL_NAME, "images")
        if not exists(img_dir):
            logging.info(f"Saving trajectory images for {cls.LOCAL_NAME} to {img_dir}...")
            from huggingface_hub import list_repo_files
            all_files = list(list_repo_files(cls.HF_SOURCE, repo_type="dataset"))
            shards = sorted(
                f for f in all_files
                if f.endswith(".parquet")
                and any(f.startswith(f"data/{config}-") for config in cls.HF_CONFIGS)
            )
            logging.info(f"Found {len(shards)} parquet shards for {cls.LOCAL_NAME}.")
            rows = list(_iter_polars_parquet(cls.HF_SOURCE, cls.HF_SPLIT, hf_shards=shards))
            _save_traj_images_to_disk(rows, img_dir)
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} saved to {img_dir}.")
        else:
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} already exist at {img_dir}, skipping.")

    def __init__(
        self,
        split: Literal["train"],
        configs: tuple = None,  # subset of HF_CONFIGS to load; None means all
        mode: str = "center",
        detail_level: str = "all",
        style: str = "webolmo_base",
        max_past_steps: int = 3,
        max_msg_char_len: int = 1000,
        n_procs: int = 1,
    ):
        if split != "train":
            raise ValueError(
                f"MolmoWebSyntheticTrajs only supports split='train' "
                f"(HuggingFace has no val/test split). Got: {split!r}"
            )
        self.mode = mode
        self.style = style
        self.detail_level = detail_level
        self.max_past_steps = max_past_steps
        self.max_msg_len = max_msg_char_len
        self.n_procs = n_procs
        self.configs = list(configs) if configs is not None else list(self.HF_CONFIGS)
        super().__init__(split)

    def __len__(self):
        return len(self.data)

    def _build_image_index(self, row: dict) -> dict:
        """Return a dict mapping screenshot filename → absolute file path on disk."""
        sample_id = row["sample_id"]
        img_dir = join(WEB_DATA_HOME, self.LOCAL_NAME, "images", sample_id)
        if exists(img_dir):
            return {f: join(img_dir, f) for f in sorted(os.listdir(img_dir))}
        # Fallback: reconstruct from in-memory images (e.g. Polars row before download).
        raw_images = row.get("images") or []
        paths = row.get("image_paths") or []
        if not paths:
            first = raw_images[0] if raw_images else None
            if isinstance(first, dict) and "path" in first:
                paths = [img["path"] for img in raw_images]
            else:
                paths = [f"screenshot_{i+1:03d}.png" for i in range(len(raw_images))]
        return {Path(p).name: join(img_dir, Path(p).name) for p in paths}

    def truncate_str(self, some_str: str, max_len: int, postfix: str = "... (truncated)"):
        if len(some_str) <= max_len:
            return some_str
        return some_str[: max_len - len(postfix)] + postfix

    def truncate_urls_or_titles(self, urls_or_titles: list[str] | str, max_len: int = 100):
        if isinstance(urls_or_titles, str):
            return self.truncate_str(urls_or_titles, max_len)
        elif isinstance(urls_or_titles, list):
            return [self.truncate_str(url_or_title, max_len) for url_or_title in urls_or_titles]
        else:
            return self.truncate_str(str(urls_or_titles), max_len)

    def get_formatted_action(self, action_output, image_w, image_h):
        action_name = action_output["action_name"]
        formatted_action = {"name": action_name}
        bbox = None
        if action_name not in ["click", "scroll", "scroll_at", "mouse_drag_and_drop"]:
            formatted_action.update(
                {k: v for k, v in action_output["action"].items()}
            )
            if action_name == "send_msg_to_user":
                formatted_action["msg"] = self.truncate_str(
                    formatted_action.get("msg", ""),
                    max_len=self.max_msg_len,
                )
        else:
            if action_name == "click":
                x, y = None, None
                if "bbox" not in action_output["action"]:
                    x, y = float(action_output["action"]["x"]), float(action_output["action"]["y"])
                else:
                    x1, y1, w, h = action_output["action"]["bbox"]
                    bbox = [x1, y1, x1 + w, y1 + h]
                    coords = get_click_coords_from_bbox(bbox, mode=self.mode)
                    x, y = float(coords[0]), float(coords[1])
                normalized_coords = normalize_click_coords(x, y, image_w, image_h)
                formatted_action["x"] = normalized_coords[0]
                formatted_action["y"] = normalized_coords[1]
                formatted_action["button"] = action_output["action"].get("button", "")
                formatted_action["click_type"] = action_output["action"].get("click_type", "")
            elif action_name == "scroll":
                delta_x = action_output["action"]["delta_x"]
                delta_y = action_output["action"]["delta_y"]
                normalized_coords = normalize_scroll_deltas(delta_x, delta_y, image_w, image_h)
                formatted_action["delta_x"] = normalized_coords[0]
                formatted_action["delta_y"] = normalized_coords[1]
            elif action_name == "scroll_at":
                x = action_output["action"]["x"]
                y = action_output["action"]["y"]
                norm_x, norm_y = normalize_click_coords(x, y, image_w, image_h)
                formatted_action["x"] = norm_x
                formatted_action["y"] = norm_y
                delta_x = action_output["action"]["delta_x"]
                delta_y = action_output["action"]["delta_y"]
                norm_dx, norm_dy = normalize_scroll_deltas(delta_x, delta_y, image_w, image_h)
                formatted_action["delta_x"] = norm_dx
                formatted_action["delta_y"] = norm_dy
            elif action_name == "mouse_drag_and_drop":
                from_x = action_output["action"]["from_x"]
                from_y = action_output["action"]["from_y"]
                norm_from_x, norm_from_y = normalize_click_coords(from_x, from_y, image_w, image_h)
                to_x = action_output["action"]["to_x"]
                to_y = action_output["action"]["to_y"]
                norm_to_x, norm_to_y = normalize_click_coords(to_x, to_y, image_w, image_h)
                formatted_action["from_x"] = norm_from_x
                formatted_action["from_y"] = norm_from_y
                formatted_action["to_x"] = norm_to_x
                formatted_action["to_y"] = norm_to_y
        return formatted_action, bbox



    def _select_goal(self, instruction: dict) -> tuple[str, str]:
        """Select goal text from a parsed instruction dict. Returns (goal_text, level_name).

        Degrades gracefully when some instruction levels are absent in HF data:
          - task_seeded_wv / task_seeded_om2w / multi_agent: HF only has low_level
          - node_traversal: HF only has low_level (high/mid absent)
          - from_template: HF has high/mid/low but no steps list
        """
        hl = instruction.get("high_level", "")
        ml = instruction.get("mid_level", "")
        ll = instruction.get("low_level", "")
        steps_list = instruction.get("steps", [])
        steps_text = "\n".join(steps_list) if steps_list else ""
        # best single fallback: first non-empty level
        _best = hl or ml or ll

        if self.detail_level == "HL":
            return hl or ml or ll, "high_level"
        elif self.detail_level == "ML":
            return ml or hl or ll, "mid_level"
        elif self.detail_level == "LL":
            return ll or ml or hl, "low_level"
        elif self.detail_level == "steps":
            return steps_text or _best, "steps"
        elif self.detail_level == "goal":
            # Old local data stored a flat "goal" string; HF wraps it in the
            # instruction dict. Return the best available level so trajectories
            # whose HF instruction only has low_level are not silently dropped.
            return _best, "goal"
        elif self.detail_level in ("all", "all_no_goal"):
            candidates = [(k, v) for k, v in [
                ("high_level", hl), ("mid_level", ml), ("low_level", ll), ("steps", steps_text)
            ] if v]
            if not candidates:
                return _best, "high_level"
            label, text = random.choice(candidates)
            return text, label
        elif self.detail_level == "mix_hml":
            # Weights renormalize automatically over whichever levels are present.
            # node_traversal (only ll) → all samples get ll; that is expected.
            candidates = [(k, v, w) for k, v, w in [
                ("high_level", hl, 0.4), ("mid_level", ml, 0.4), ("low_level", ll, 0.2)
            ] if v]
            if not candidates:
                return _best, "high_level"
            labels, texts, weights = zip(*candidates)
            idx = random.choices(range(len(texts)), weights=weights, k=1)[0]
            return texts[idx], labels[idx]
        elif self.detail_level == "mix_hmls":
            candidates = [(k, v, 0.25) for k, v in [
                ("high_level", hl), ("mid_level", ml), ("low_level", ll), ("steps", steps_text)
            ] if v]
            if not candidates:
                return _best, "high_level"
            labels, texts, weights = zip(*candidates)
            idx = random.choices(range(len(texts)), weights=weights, k=1)[0]
            return texts[idx], labels[idx]
        return _best, "high_level"

    def _process_step(
        self,
        traj_step: dict,
        image,
        goal: str,
        past_actions: list,
        past_urls: list,
        step_idx: str,
        sample_id: str,
        instruction: dict,
    ) -> tuple[dict, dict, str, bool]:
        """Process one step. Returns (formatted_example, answer_dict, page_url, has_required_keys)."""
        action_output = traj_step["action"]["action_output"]
        if "image_w" in traj_step and "image_h" in traj_step:
            image_w, image_h = traj_step["image_w"], traj_step["image_h"]
        else:
            image_w, image_h = Image.open(image).size

        other_obs = traj_step.get("other_obs") or {}
        has_required_keys = bool(
            other_obs
            and "page_index" in other_obs
            and "open_pages_titles" in other_obs
            and "open_pages_urls" in other_obs
        )
        if has_required_keys:
            page_index = other_obs["page_index"]
            open_pages_titles = [t if t is not None else "New Tab" for t in other_obs["open_pages_titles"]]
            open_pages_urls = [u if u is not None else "about:blank" for u in other_obs["open_pages_urls"]]
            page_title = self.truncate_urls_or_titles(open_pages_titles[page_index])
            page_url = self.truncate_urls_or_titles(open_pages_urls[page_index])
            open_pages_urls = self.truncate_urls_or_titles(open_pages_urls)
            open_pages_titles = self.truncate_urls_or_titles(open_pages_titles)
            open_pages_titles_and_urls = list(zip(open_pages_titles, open_pages_urls))
            last_action_error = traj_step.get("error") or "The action was successful with no error."
        else:
            page_index = 0
            page_title = "Unknown"
            page_url = "Unknown"
            open_pages_urls = []
            open_pages_titles = []
            open_pages_titles_and_urls = []
            last_action_error = "No observation data (other_obs is empty or missing keys)."

        formatted_action, bbox = self.get_formatted_action(action_output, image_w, image_h)
        effective_style = self.style
        if effective_style == "molmo_web_mixed":
            effective_style = "molmo_web_base" if random.random() < 0.5 else "molmo_web_think"

        if effective_style == "molmo_web_think":
            answer_dict = {
                "thought": traj_step["action"]["action_output"]["thought"].strip(),
                "action": formatted_action,
            }
        else:  # molmo_web_base
            answer_dict = {"action": formatted_action}

        message = dict(
            answer=json.dumps(answer_dict, ensure_ascii=False),
            task_description=goal,
            past_actions=past_actions[-self.max_past_steps:],
            past_urls=past_urls[-self.max_past_steps:],
            page_index=page_index,
            page_title=page_title,
            page_url=page_url,
            open_pages_titles_and_urls=open_pages_titles_and_urls,
            last_action_error=last_action_error,
            style=effective_style,
        )

        formatted_example = {
            "image": image,
            "message_list": [message],
            "metadata": dict(
                traj_id=sample_id,
                dataset=self.configs[0] if len(self.configs) == 1 else "molmoweb_synthetic",
                step_id=step_idx,
                high_level=instruction.get("high_level", ""),
                mid_level=instruction.get("mid_level", ""),
                low_level=instruction.get("low_level", ""),
                steps=instruction.get("steps", []),
                bbox=bbox,
                open_pages_titles=open_pages_titles,
                open_pages_urls=open_pages_urls,
                image_w=image_w,
                image_h=image_h,
                answer=json.dumps(answer_dict),
            ),
        }
        return formatted_example, answer_dict, page_url, has_required_keys

    def _load_rows(self):
        """Yield (row, config) for each HF row across all requested configs."""
        for config in self.configs:
            hf_dataset = _load_hf_dataset(
                self.HF_SOURCE, self.HF_SPLIT,
                f"{self.LOCAL_NAME}/{config}", config=config,
            )
            for row in tqdm(hf_dataset, desc=f"Loading {config}"):
                yield row, config

    def _process_one_row(self, row_idx: int, row: dict, config: str) -> tuple[int, list]:
        """Process one trajectory row. Used directly and via mp.Pool workers."""
        sample_id = row["sample_id"]
        try:
            instruction = json.loads(row["instruction"])
            trajectory = json.loads(row["trajectory"])
        except Exception:
            return row_idx, []


        try:
            goal, _ = self._select_goal(instruction)
            if not goal:
                goal = self._select_goal_fallback(trajectory)
        except Exception as e:
            logging.warning(f"Error selecting goal for {sample_id}: {e}")
            return row_idx, []
        if not goal:
            return row_idx, []

        image_index = self._build_image_index(row)
        past_actions, past_urls = [], []
        results = []

        for step_idx, traj_step in sorted(trajectory.items(), key=lambda x: int(x[0])):
            screenshot_key = traj_step.get("screenshot", "")
            image_path = image_index.get(screenshot_key)
            if image_path is None:
                continue
            try:
                formatted_example, answer_dict, page_url, _ = self._process_step(
                    traj_step=traj_step, image=image_path, goal=goal,
                    past_actions=past_actions, past_urls=past_urls,
                    step_idx=step_idx, sample_id=sample_id, instruction=instruction,
                )
            except Exception as e:
                if "uni-directional" not in str(e):
                    logging.debug(f"Skipping step {step_idx} in {sample_id}: {e}")
                continue

            if self.DATASET_TAG:
                formatted_example["metadata"]["dataset"] = self.DATASET_TAG
            results.append(formatted_example)
            past_actions.append({**answer_dict, "index": step_idx})
            past_urls.append(page_url)

        return row_idx, results

    def load(self):
        global _synthetic_traj_rows
        print(f"Loading {self.LOCAL_NAME} with {self.n_procs} process(es)...")

        _synthetic_traj_rows = list(self._load_rows())

        result_map: dict[int, list] = {}
        if self.n_procs > 1:
            with mp.get_context("fork").Pool(
                self.n_procs,
                initializer=_init_synthetic_traj_worker,
                initargs=(self,),
            ) as pool:
                for row_idx, examples in tqdm(
                    pool.imap(_call_synthetic_traj_row, range(len(_synthetic_traj_rows)), chunksize=8),
                    total=len(_synthetic_traj_rows), desc="Processing trajectories",
                ):
                    if examples:
                        result_map[row_idx] = examples
        else:
            for row_idx, (row, config) in tqdm(
                enumerate(_synthetic_traj_rows), total=len(_synthetic_traj_rows),
                desc="Processing trajectories",
            ):
                _, examples = self._process_one_row(row_idx, row, config)
                if examples:
                    result_map[row_idx] = examples

        formatted_data = [ex for i in sorted(result_map) for ex in result_map[i]]
        logging.info(f"Loaded {len(formatted_data)} steps from configs={self.configs}")
        return formatted_data

    def _select_goal_fallback(self, trajectory: dict) -> str:
        """Override to recover a goal when the instruction field is empty."""
        raise ValueError("_select_goal returned no goal and no fallback is defined")

    def get(self, item, rng):
        return self.data[item]


# ── MolmoWebSyntheticTrajs multiprocessing helpers ───────────────────────────
_synthetic_traj_rows: list | None = None  # list of (row, config) tuples
_synthetic_traj_dataset = None


def _init_synthetic_traj_worker(dataset):
    global _synthetic_traj_dataset
    _synthetic_traj_dataset = dataset


def _call_synthetic_traj_row(row_idx: int):
    row, config = _synthetic_traj_rows[row_idx]
    return _synthetic_traj_dataset._process_one_row(row_idx, row, config)


# ── MolmoWebHumanTrajs multiprocessing helpers ────────────────────────────────
# On Linux (fork), _human_traj_rows is inherited COW by worker processes, so
# only the small `dataset` config object is actually pickled per-worker.
_human_traj_rows: list | None = None
_human_traj_dataset = None


def _init_human_traj_worker(dataset):
    global _human_traj_dataset
    _human_traj_dataset = dataset


def _call_human_traj_row(row_idx: int):
    return _human_traj_dataset._process_one_row(row_idx, _human_traj_rows[row_idx])


class MolmoWebHumanTrajs(MolmoWebSyntheticTrajs):
    """
    Human trajectory dataset.
    Downloads data directly from HuggingFace: allenai/MolmoWeb-HumanTrajs.
    """

    HF_SOURCE = "allenai/MolmoWeb-HumanTrajs"
    HF_SPLIT = "train"
    HF_CONFIGS = ("default",)
    LOCAL_NAME = "MolmoWeb-HumanTrajs"

    BLACKLIST_IDS = {
        "20260131_snorkel_batch007_procedural__flight_search__9__steps_004_to_041",
        "20260131_snorkel_batch012_procedural__news_search__9",
    }
    REQUIRE_GOTO_START = True  # skip trajs whose first action isn't goto()
    DATASET_TAG = "molmoweb_human"

    @classmethod
    def download(cls, n_procs=None):
        logging.info(f"Downloading {cls.LOCAL_NAME} from {cls.HF_SOURCE}...")
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME, drop_columns=["images"])
        img_dir = join(WEB_DATA_HOME, cls.LOCAL_NAME, "images")
        if not exists(img_dir):
            logging.info(f"Saving trajectory images for {cls.LOCAL_NAME} to {img_dir}...")
            rows = list(_iter_polars_parquet(cls.HF_SOURCE, cls.HF_SPLIT))
            _save_traj_images_to_disk(rows, img_dir)
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} saved to {img_dir}.")
        else:
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} already exist at {img_dir}, skipping.")

    def __init__(self, split: Literal["train"], n_procs: int = 1, **kwargs):
        kwargs.setdefault("configs", list(self.HF_CONFIGS))
        super().__init__(split, n_procs=n_procs, **kwargs)

    def _process_one_row(self, row_idx: int, row: dict) -> tuple[int, list]:
        """Process one trajectory row. Used directly and via mp.Pool workers."""
        sample_id = row["sample_id"]
        try:
            instruction = json.loads(row["instruction"])
            trajectory = json.loads(row["trajectory"])
        except Exception:
            return row_idx, []


        sorted_steps = sorted(trajectory.items(), key=lambda x: int(x[0]))
        if self.REQUIRE_GOTO_START and sorted_steps:
            first_action_str = sorted_steps[0][1].get("action", {}).get("action_str", "")
            if not first_action_str.startswith("goto("):
                return row_idx, []

        try:
            goal, _ = self._select_goal(instruction)
        except Exception as e:
            logging.warning(f"Error selecting goal for {sample_id}: {e}")
            return row_idx, []
        if not goal:
            return row_idx, []

        image_index = self._build_image_index(row)
        past_actions, past_urls = [], []
        results = []

        for step_idx, traj_step in sorted_steps:
            screenshot_key = traj_step.get("screenshot", "")
            image_path = image_index.get(screenshot_key)
            if image_path is None:
                continue
            try:
                formatted_example, answer_dict, page_url, _ = self._process_step(
                    traj_step=traj_step, image=image_path, goal=goal,
                    past_actions=past_actions, past_urls=past_urls,
                    step_idx=step_idx, sample_id=sample_id, instruction=instruction,
                )
            except Exception as e:
                if "uni-directional" not in str(e):
                    logging.debug(f"Skipping step {step_idx} in {sample_id}: {e}")
                continue

            formatted_example["metadata"]["dataset"] = self.DATASET_TAG
            results.append(formatted_example)
            past_actions.append({**answer_dict, "index": step_idx})
            past_urls.append(page_url)

        return row_idx, results

    def _load_rows(self):
        # Use Polars to read parquet shards directly — same approach as MolmoWebSyntheticTrajs,
        # avoids the datasets library's slow Arrow→Python dict conversion for image bytes.
        for row in tqdm(
            _iter_polars_parquet(self.HF_SOURCE, self.HF_SPLIT),
            desc="Loading MolmoWeb-HumanTrajs",
        ):
            if row["sample_id"] in self.BLACKLIST_IDS:
                continue
            yield row, "default"

    def load(self):
        global _human_traj_rows
        print(f"Loading MolmoWeb-HumanTrajs with {self.n_procs} process(es)...")

        # Collect all rows first so mp.Pool workers can access them via the
        # fork-inherited global (no per-task pickling of large image bytes).
        _human_traj_rows = [row for row, _ in self._load_rows()]

        result_map: dict[int, list] = {}
        if self.n_procs > 1:
            with mp.get_context("fork").Pool(
                self.n_procs,
                initializer=_init_human_traj_worker,
                initargs=(self,),
            ) as pool:
                for row_idx, examples in tqdm(
                    pool.imap(_call_human_traj_row, range(len(_human_traj_rows)), chunksize=8),
                    total=len(_human_traj_rows), desc="Processing trajectories",
                ):
                    if examples:
                        result_map[row_idx] = examples
        else:
            for row_idx, row in tqdm(
                enumerate(_human_traj_rows), total=len(_human_traj_rows),
                desc="Processing trajectories",
            ):
                _, examples = self._process_one_row(row_idx, row)
                if examples:
                    result_map[row_idx] = examples

        formatted_data = [ex for i in sorted(result_map) for ex in result_map[i]]
        logging.info(f"Loaded {len(formatted_data)} steps from MolmoWeb-HumanTrajs")
        return formatted_data


class MolmoWebSyntheticSkills(MolmoWebSyntheticTrajs):
    """
    Synthetic (Gemini-generated) atomic skills dataset.
    Downloads data directly from HuggingFace: allenai/MolmoWeb-SyntheticSkills.

    Corresponds to the old webolmoSynthetic__atomic_actions_*__goal__... datasets.
    The HF ``instruction`` field is JSON with a single ``goal`` key containing
    a multi-line string of chained instruction steps (goto / find_and_open /
    fill_form / find_and_click).
    """

    HF_SOURCE = "allenai/MolmoWeb-SyntheticSkills"
    HF_SPLIT = "train"
    HF_CONFIGS = ("default",)
    LOCAL_NAME = "MolmoWeb-SyntheticSkills"
    # Known parquet shards; avoids a live list_repo_files() network call.
    HF_SHARDS = ["data/train-00000.parquet", "data/train-00001.parquet"]
    DATASET_TAG = "molmoweb_synthetic_skills"

    @classmethod
    def download(cls, n_procs=None):
        logging.info(f"Downloading {cls.LOCAL_NAME} from {cls.HF_SOURCE}...")
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME, drop_columns=["images"])
        img_dir = join(WEB_DATA_HOME, cls.LOCAL_NAME, "images")
        if not exists(img_dir):
            logging.info(f"Saving trajectory images for {cls.LOCAL_NAME} to {img_dir}...")
            rows = list(_iter_polars_parquet(cls.HF_SOURCE, cls.HF_SPLIT, cls.HF_SHARDS))
            _save_traj_images_to_disk(rows, img_dir)
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} saved to {img_dir}.")
        else:
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} already exist at {img_dir}, skipping.")

    def __init__(self, split: Literal["train"], **kwargs):
        kwargs.setdefault("configs", list(self.HF_CONFIGS))
        super().__init__(split, **kwargs)

    def _load_rows(self):
        # datasets.load_dataset crashes on PyArrow 19 for list<binary> columns;
        # use Polars-based reader with hardcoded shard paths instead.
        rows = list(_iter_polars_parquet(self.HF_SOURCE, self.HF_SPLIT, self.HF_SHARDS))
        for row in tqdm(rows, desc="Loading MolmoWeb-SyntheticSkills"):
            yield row, "default"

    def _select_goal(self, instruction: dict) -> tuple[str, str]:
        """SyntheticSkills instructions have a single 'goal' key instead of
        high/mid/low_level keys, so read it directly."""
        goal = instruction.get("goal", "")
        return goal, "goal"

    def _select_goal_fallback(self, trajectory: dict) -> str:
        """Override to recover a goal when the instruction field is empty."""
        raise ValueError("_select_goal returned no goal and no fallback is defined")


class MolmoWebHumanSkills(MolmoWebHumanTrajs):
    """
    Human-annotated atomic skills dataset.
    Downloads data directly from HuggingFace: allenai/MolmoWeb-HumanSkills.

    HF instruction has high_level, mid_level, low_level, steps.
    """

    HF_SOURCE = "allenai/MolmoWeb-HumanSkills"
    HF_SPLIT = "train"
    HF_CONFIGS = ("default",)
    LOCAL_NAME = "MolmoWeb-HumanSkills"
    # Known parquet shards; avoids a live list_repo_files() network call.
    HF_SHARDS = [f"data/train-{i:05d}.parquet" for i in range(21)]

    BLACKLIST_IDS: set = set()  # no known blacklisted IDs for skills
    REQUIRE_GOTO_START = False  # skills steps don't necessarily start with goto()
    DATASET_TAG = "molmoweb_human_skills"

    @classmethod
    def download(cls, n_procs=None):
        logging.info(f"Downloading {cls.LOCAL_NAME} from {cls.HF_SOURCE}...")
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME, drop_columns=["images"])
        img_dir = join(WEB_DATA_HOME, cls.LOCAL_NAME, "images")
        if not exists(img_dir):
            logging.info(f"Saving trajectory images for {cls.LOCAL_NAME} to {img_dir}...")
            rows = list(_iter_polars_parquet(cls.HF_SOURCE, cls.HF_SPLIT, cls.HF_SHARDS))
            _save_traj_images_to_disk(rows, img_dir)
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} saved to {img_dir}.")
        else:
            logging.info(f"Trajectory images for {cls.LOCAL_NAME} already exist at {img_dir}, skipping.")

    def _load_rows(self):
        # datasets.load_dataset crashes on PyArrow 19 for list<binary> columns;
        # use Polars-based reader with hardcoded shard paths instead.
        rows = list(_iter_polars_parquet(self.HF_SOURCE, self.HF_SPLIT, self.HF_SHARDS))
        for row in tqdm(rows, desc="Loading MolmoWeb-HumanSkills"):
            if row["sample_id"] not in self.BLACKLIST_IDS:
                yield row, "default"


if __name__ == "__main__":
    from olmo.data.get_dataset import get_dataset_by_name
    dataset_names = [
        "molmoweb_synthetic_ground__template",
        "molmoweb_synthetic_ground__gpt",
        "molmoweb_screenshot_qa",
        "molmoweb_synthetic_trajs",
        "molmoweb_human_trajs",
        "molmoweb_synthetic_skills",
        "molmoweb_human_skills",
        "pixmo_points_single_web",
        "screenspot",
        "screenspot_v2"
    ]
    split = "train"
    for ds in dataset_names:
        ds = get_dataset_by_name(new, split=split)
        print(f"Loaded {ds} with {len(ds)} examples.")
        ex = ds[0]
        print(f"New example: {ex}")