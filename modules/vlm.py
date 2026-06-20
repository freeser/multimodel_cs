import base64
from PIL import Image
import cv2
import io
import numpy as np
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, VLM_MODEL

# 初始化客户端
client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL
)

# 定义函数
def detect_defect_with_vlm(image, prompt:str = None):
    """image应该是numpy格式或者PIL格式"""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    if prompt is None:
        prompt = """你是一个专业的电商客服助手。请观察图片，简洁的告诉我商品可能存在什么瑕疵（
        例如：物资、破洞、脱线、勾丝等）。
        回复要求：
            1. 只需要直接回复瑕疵描述，不要有多余的解释。
            2. 如果没有发现明显瑕疵，直接回复“未检测到明显异常”。
            3. 总字数不要超过40个字。
            4. 用口语化、关切的语言。例如：“亲，我看到衣服胸口这里有一块瑕疵。”            
        """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in detect_defect_with_vlm: {str(e)}")
        return "未检测到明显异常"
