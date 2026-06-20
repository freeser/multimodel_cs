from config import ASR_MODEL, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL
import requests

def speech_to_text(file_bytes: bytes):
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    }

    files = {
        "file": ("audio.wav", file_bytes, "audio/wav"),
    }

    data = {
        "model": ASR_MODEL,
        "language": "zh"
    }

    response = requests.post(
        f"{SILICONFLOW_BASE_URL}/audio/transcriptions",
        headers=headers,
        files=files,
        data=data
    )

    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"ASR Error: {response.status_code}, {response.text}")

# if __name__ == "__main__":
#     # 现在想导入自定的包/模块
#     import sys
#     sys.path.append(r"D:\Study\python\multimodel_cs")
#     with open(r"D:\Study\python\multimodel_cs\static\test.wav", "rb") as f:
#         text = speech_to_text(f.read())
#         print(text)

