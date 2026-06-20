import gradio as gr
import requests
import io
import soundfile as sf
import base64
from PIL import Image

API_BASE = "http://127.0.0.1:8000"

def text_chat(user_text, session_id, use_rag):
    response = requests.post(
        f"{API_BASE}/api/chat/text",
        data={
            "text": user_text,
            "session_id": session_id if session_id else "",
            "use_rag": use_rag    # 传递给后端
        }
    )
    data = response.json()
    new_session_id = data.get("session_id", session_id)
    return f"【意图】{data.get('intent')}  {data.get('reply', '')}", new_session_id

def image_chat(image, text, session_id):
    if image is None:
        return "请上传图片", session_id

    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    files = {"image": ("image.jpg", buffer.getvalue(), "image/jpeg")}
    data = {
        "text": text,
        "session_id": session_id if session_id else ""
    }

    response = requests.post(
        f"{API_BASE}/api/chat/image",
        files=files,
        data=data
    )
    result = response.json()
    new_session_id = result.get("session_id", session_id)

    return (
        f"【检测结果】{result.get('detections')}\n"
        f"【意图】{result.get('intent')}\n"
        f"{result.get('reply')}"
    ), new_session_id

def voice_chat(audio, text, session_id):
    if audio is None:
        return "请录制或上传语音", None, session_id

    sample_rate, audio_data = audio

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
    wav_buffer.seek(0)

    files = {"audio": ("audio.wav", wav_buffer.getvalue(), "audio/wav")}
    data = {
        "text": text,
        "session_id": session_id if session_id else ""
    }

    response = requests.post(
        f"{API_BASE}/api/chat/voice",
        files=files,
        data=data
    )
    result = response.json()

    audio_base64 = result.get("audio_base64", "")
    audio_output = None
    if audio_base64:
        audio_output = base64.b64decode(audio_base64)

    text_output = (
        f"【识别文本】{result.get('transcript')}\n"
        f"【意图】{result.get('intent')}\n"
        f"{result.get('reply')}"
    )

    new_session_id = result.get("session_id", session_id)
    return text_output, audio_output, new_session_id

# 清除会话的函数
def clear_session():
    return "", ""  # 返回空字符串给 session_id 和文本显示

with gr.Blocks(title="多模态智能客服演示系统") as demo:
    gr.Markdown("# 大模型驱动的多模态智能客服")

    # 为每个 Tab 维护独立的 session state（也可以共用一个，看你的需求）
    text_session = gr.State(value="")
    image_session = gr.State(value="")
    audio_session = gr.State(value="")

    with gr.Tab("文本客服"):
        text_input = gr.Textbox(label="输入问题")
        rag_checkbox = gr.Checkbox(label="启用知识库检索 (RAG)", value=False)   # 新增
        text_output = gr.Textbox(label="回复", interactive=False)
        text_button = gr.Button("发送")
        text_button.click(
            text_chat,
            inputs=[text_input, text_session, rag_checkbox],
            outputs=[text_output, text_session]
        )

    with gr.Tab("图文客服"):
        image_input = gr.Image(label="上传商品图片")
        image_text = gr.Textbox(label="补充描述（可选）")
        image_output = gr.Textbox(label="回复", interactive=False)
        image_button = gr.Button("发送")
        image_button.click(
            image_chat,
            inputs=[image_input, image_text, image_session],
            outputs=[image_output, image_session]
        )

    with gr.Tab("语音客服"):
        audio_input = gr.Audio(label="录制/上传语音", type="numpy")
        audio_text = gr.Textbox(label="补充描述（可选）")
        audio_output_text = gr.Textbox(label="识别与回复", interactive=False)
        audio_output_audio = gr.Audio(label="合成语音回复")
        audio_button = gr.Button("发送")
        audio_button.click(
            voice_chat,
            inputs=[audio_input, audio_text, audio_session],
            outputs=[audio_output_text, audio_output_audio, audio_session]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)