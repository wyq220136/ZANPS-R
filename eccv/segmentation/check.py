import os
import io
import re
import base64
import numpy as np
from PIL import Image, ImageFile
from openai import OpenAI
from dotenv import load_dotenv
import scipy.ndimage as ndimage

# ==========================================
# Interface variables for standalone debugging
# ==========================================
IMAGE_PATH = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/rgb/Remote_101028_0_0.png"
MASK_PATH = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs/Remote_101028/mask/Remote_101028_0_0.png"
SAVE_PATH = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs/Remote_101028/matched_pred_mask_direct_match_adaptive/Remote_101028_0_0/mask_0000.png"
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(SCRIPT_DIR, ".env")
if not load_dotenv(DOTENV_PATH):
    load_dotenv()
ImageFile.LOAD_TRUNCATED_IMAGES = True
CHECK_MODEL_NAME = "qwen/qwen3.7-plus"
BASE_URL = "https://zenmux.ai/api/v1"

EVALUATE_MASK_PROMPT = """
You are a robotic vision expert for functional part identification in manipulation tasks.

You are given:
1. Binary Mask Image (white = selected region, black = background)
2. Overlay Image (same mask overlaid on RGB, with cyan boundary)

Your task is to determine whether the selected region is a  precise, complete, and user-operable ariculated part.
Use the binary mask for precise region extent and the overlay for semantic context.

The part must be explicitly designed for direct human interaction, not for internal mechanical support, connection, or motion transmission.
The part must be intentionally designed for human interaction, not merely capable of movement or having a mechanical function.
### TARGET PARTS CATEGORIES:
Identify ONLY: handle, knob, button, switch, key, drawer, lid, door.

### REJECTION RULES (Return False if):
- Mask covers the whole object, the base, or a large static surface.
- The mask is partial, incomplete, or only covers a fragment of the part.
- The mask is broken, grainy, or fragmented.
- The mask covers more than one part or spans across unrelated areas.
- The region is a logo, label, texture, reflection, or static clutter.
- If the region does not belong to one of the predefined functional categories (handle, knob, button, switch, key, drawer, lid, or door), classify it as False.
- The mask covers a non-functional, static part of the object's body.
- The mask covers the internal storage area, partitions, or compartments of a container instead of the moving closure part itself.

### OUTPUT:
Result: [True or False]
"""

_GLOBAL_CLIENT = None


def get_openai_client():
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        api_key = os.getenv("ZENMUX_API_KEY") or os.getenv("QWEN_KEY")
        if not api_key:
            raise RuntimeError(
                f"ZENMUX_API_KEY is not set. Expected it in {DOTENV_PATH} or shell env. "
                "QWEN_KEY is also accepted as a backward-compatible fallback."
            )
        _GLOBAL_CLIENT = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _GLOBAL_CLIENT


def image_to_base64(image_pil):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_advanced_overlay(image_np, mask_np):
    img_float = image_np.astype(np.float32)
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    mask_binary = mask_np > 128

    dimmed_bg = img_float * 0.25
    final_img_np = dimmed_bg.copy()
    final_img_np[mask_binary] = img_float[mask_binary]

    struct = ndimage.generate_binary_structure(2, 2)
    erosion = ndimage.binary_erosion(mask_binary, structure=struct)
    edge = mask_binary ^ erosion
    final_img_np[edge] = [0, 255, 255]

    return Image.fromarray(np.uint8(np.clip(final_img_np, 0, 255)))


def evaluate_segmentation(image_np, mask_np, client=None, save_path=None, original_b64=None):
    visual_img = create_advanced_overlay(image_np, mask_np)
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    bw_img = Image.fromarray(((mask_np > 128).astype(np.uint8) * 255), mode="L")

    # Skip disk I/O when save_path is None (faster default path in pipeline).
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        visual_img.save(save_path)

    visual_b64 = image_to_base64(visual_img)
    bw_b64 = image_to_base64(bw_img)
    _ = original_b64  # Kept for backward compatibility, not used now.

    try:
        if client is None:
            client = get_openai_client()

        response = client.chat.completions.create(
            model=CHECK_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Binary Mask Image (white=selected region)"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bw_b64}"}},
                        {"type": "text", "text": "Overlay Image (mask over RGB)"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{visual_b64}"}},
                        {"type": "text", "text": EVALUATE_MASK_PROMPT},
                    ],
                }
            ],
            extra_body={"enable_thinking": False},
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Model returned no content.")
        res_text = content.strip()

        result_match = re.search(r"Result:\s*(True|False)", res_text, re.IGNORECASE)
        if result_match:
            return result_match.group(1).lower() == "true", ""

        plain_match = re.search(r"\b(True|False)\b", res_text, re.IGNORECASE)
        if plain_match:
            return plain_match.group(1).lower() == "true", ""

        return False, "No valid boolean result."

    except Exception as e:
        return False, f"Error occurred: {str(e)}"


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
        print(f"[ERROR] Please check your paths.\nImage: {IMAGE_PATH}\nMask: {MASK_PATH}")
    else:
        img = Image.open(IMAGE_PATH).convert("RGB")
        mask = Image.open(MASK_PATH).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask)

        print("[PROCESS] Evaluating segmentation quality...")
        is_perfect, reason = evaluate_segmentation(img_np, mask_np, save_path=SAVE_PATH)

        print("\n" + "=" * 30)
        print(f"DECISION: {is_perfect}")
        print(f"REASON  : {reason}")
        print("=" * 30)
        if is_perfect:
            print("OK")
