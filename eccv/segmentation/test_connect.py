import os
import base64
import mimetypes

from dotenv import load_dotenv
from openai import OpenAI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(SCRIPT_DIR, ".env")

CHECK_MODEL_NAME = "qwen/qwen3.7-plus"
BASE_URL = "https://zenmux.ai/api/v1"

IMAGE_PATH = r"D:\research\PartNet\dataset_train\test\bottle_3517\rgb\000000.png"

TEST_PROMPT = "图片上的是什么东西，图上的物体的盖子是什么颜色的？"


def load_local_env():
    print("====================================")
    print("[INIT] Loading environment...")

    loaded = load_dotenv(DOTENV_PATH)

    if not loaded:
        print("[WARN] .env not found at script dir, trying fallback load_dotenv()")
        load_dotenv()

    print("[INIT] dotenv path =", DOTENV_PATH)
    print("[INIT] dotenv loaded =", loaded)
    print("====================================")

    return loaded


def get_openai_client():
    print("[DEBUG] Creating OpenAI client...")

    api_key = os.getenv("ZENMUX_API_KEY") or os.getenv("QWEN_KEY")

    if not api_key:
        raise RuntimeError(
            f"ZENMUX_API_KEY is not set. Expected it in {DOTENV_PATH} or shell env. "
            "QWEN_KEY is also accepted as a backward-compatible fallback."
        )

    print("[DEBUG] API key loaded (prefix):", api_key[:6] + "******")
    print("[DEBUG] BASE_URL =", BASE_URL)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    print("[DEBUG] OpenAI client created successfully")
    return client


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def test_connection():
    client = get_openai_client()
    image_url = image_to_data_url(IMAGE_PATH)

    print("====================================")
    print("[DEBUG] Preparing request...")
    print("[DEBUG] MODEL =", CHECK_MODEL_NAME)
    print("[DEBUG] BASE_URL =", BASE_URL)
    print("[DEBUG] IMAGE =", IMAGE_PATH)
    print("[DEBUG] PROMPT =", TEST_PROMPT)
    print("====================================")

    try:
        print("[DEBUG] Sending request to server...")

        response = client.chat.completions.create(
            model=CHECK_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TEST_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
        )

        print("[DEBUG] Request completed successfully")
        print("[DEBUG] Raw response object type:", type(response))
        print("[DEBUG] Raw response:", response)

    except Exception as e:
        print("====================================")
        print("[ERROR] Request failed at API call stage")
        print("[ERROR] Exception type:", type(e))
        print("[ERROR] Exception repr:", repr(e))
        print("====================================")
        raise

    print("====================================")
    msg = response.choices[0].message
    content = msg.content
    print("[DEBUG] finish_reason =", response.choices[0].finish_reason)
    print("[DEBUG] message.content =", content)
    print("[DEBUG] message.reasoning =", getattr(msg, "reasoning", None))

    if content is None:
        raise RuntimeError("Model returned no content.")

    return content.strip()


def main():
    env_loaded = load_local_env()

    print("====================================")
    print("[INIT SUMMARY]")
    print("dotenv_loaded =", env_loaded)
    print("dotenv_path   =", DOTENV_PATH)
    print("model         =", CHECK_MODEL_NAME)
    print("base_url      =", BASE_URL)
    print("====================================")

    try:
        res_text = test_connection()

    except Exception as e:
        print("====================================")
        print("[FATAL] Request failed")
        print("[FATAL] Exception:", repr(e))
        print("====================================")
        raise SystemExit(1)

    print("====================================")
    print("[MODEL RESPONSE]")
    print(res_text)
    print("====================================")
    print("[OK] Vision chat request completed.")


if __name__ == "__main__":
    main()
