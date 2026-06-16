from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
assert SILICONFLOW_API_KEY, "Please set the SILICONFLOW_API_KEY environment variable"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ASR_MODEL = "FunAudioLLM/SenseVoiceSmall"

# TTS模型可以调用API，进行一些参数的配置（音色、音量、情感等等）
# 也可以直接使用浏览器自带的（不收费）
TTS_MODEL = "FunAudioLLM/CosyVoice2-0.5B"

YOLO_MODEL = "yolo11n.pt"

VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SQLITE_DB = "chat.db"




