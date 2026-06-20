import uuid
import numpy as np
import cv2
import base64
import os
from fastapi import APIRouter, UploadFile, File, Form, Query
from modules.llm import generate_response, ChatHistoryManager, generate_response_with_rag
from modules.asr import speech_to_text
from modules.tts import text_to_speech
from modules.detector import detect_image
from modules.vlm import detect_defect_with_vlm
from storage.db import save_chat, init_db, save_session_history, load_session_history, delete_session_history

# 定义路由
router = APIRouter()
# 初始化数据库
init_db()

def get_or_create_session(session_id: str):
    """ 获取或创建会话 （从数据库加载历史）"""
    chat_history = ChatHistoryManager()
    # 从数据库加载会话历史
    if session_id:
        history = load_session_history(session_id)
        for msg in history:
            chat_history.history.append(msg)
    return chat_history

def save_session(chat_history: ChatHistoryManager, session_id: str):
    """ 保存会话历史到数据库 （只保存新增的消息） """
    # 获取已保存的消息数量
    existing_count = len(load_session_history(session_id))
    # 只保存新增的消息
    for msg in chat_history.history[existing_count:]:
        save_session_history(session_id, msg["role"], msg["content"])

# @router.post("/api/chat/text")
# async def chat_text(text: str = Form(...), session_id: str = Form(None)):
#     """ 文本聊天接口 """
#     # 生成或使用现有session_id
#     cuurent_seession_id = session_id or str(uuid.uuid4())

#     # 获取或创建会话
#     chat_history = get_or_create_session(cuurent_seession_id)

#     # 生成回复
#     response = generate_response(text, chat_history=chat_history)
#     # 保存对话记录
#     save_chat(text, response["intent"], response["reply"])
#     # 保存会话历史
#     save_session(chat_history, cuurent_seession_id)
#     return {
#         "session_id": cuurent_seession_id,
#         **response
#     }

#####################################################################
# 集成 RAG 优化模块
from modules.rag import ProductRAG
rag = ProductRAG()

@router.post("/api/chat/text")
async def chat_text(text: str = Form(...), session_id: str = Form(None), use_rag: bool = Form(False)):
    """ 文本聊天接口 """
    # 生成或使用现有session_id
    cuurent_seession_id = session_id or str(uuid.uuid4())
    # 获取或创建会话
    chat_history = get_or_create_session(cuurent_seession_id)
    # 如果启用rag
    if use_rag:
        rag_context = rag.retrieve(text)
    # 生成回复
    response = generate_response_with_rag(text, chat_history=chat_history, extra_context=rag_context)
    # 保存对话记录
    save_chat(text, response["intent"], response["reply"])
    # 保存会话历史
    save_session(chat_history, cuurent_seession_id)
    return {
        "session_id": cuurent_seession_id,
        **response
    }





@router.post("/api/chat/image")
async def chat_image(
    image: UploadFile = File(...),
    text: str = Form(""),
    session_id: str = Form(None)
):
    """ 图片聊天接口 """
    # 生成或使用现有session_id
    cuurent_seession_id = session_id or str(uuid.uuid4())
    # 处理图像
    image_bytes = await image.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_matrix = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 检测图像
    # features = detect_image(image_matrix)
    features = detect_defect_with_vlm(image_matrix)
    combined_text = text if text else "请描述图片中的问题"
    # 获取或创建会话
    chat_history = get_or_create_session(cuurent_seession_id)
    # 生成回复
    response = generate_response(combined_text, features, chat_history=chat_history)
    # 保存对话记录
    save_chat(combined_text, response["intent"], response["reply"])
    # 保存会话历史
    save_session(chat_history, cuurent_seession_id)
    return {
        "session_id": cuurent_seession_id,
        "detections": features,
        **response
    }

@router.post("/api/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    text: str = Form(""),
    session_id: str = Form(None)
):
    # 生成或使用现有session_id
    cuurent_session_id = session_id or str(uuid.uuid4())
    # 处理语音
    audio_bytes = await audio.read()
    print(f"接收到音频：{len(audio_bytes)} bytes")
    transcript = speech_to_text(audio_bytes)
    print(f"识别结果：{transcript}")
    # 如果没有识别到语音内容
    if not transcript.strip():
        # 获取或创建会话
        chat_history = get_or_create_session(cuurent_session_id)
        # 保存到对话历史
        chat_history.add_message("user", "[语音未识别]")
        chat_history.add_message("assistant", "抱歉，我没有听到您的问题，请再说一遍。")
        # 保存会话历史
        save_session(chat_history, cuurent_session_id)
        return {
            "session_id": cuurent_session_id,
            "transcript": "",
            "intent": "咨询",
            "slots": {"product":"", "issue":""},
            "reply": "抱歉，我没有听到您的问题，请再说一遍。",
            "audio_base64": ""
        }
    
    combined_text = (text + " " + transcript) if text else transcript
    # 获取或创建会话
    chat_history = get_or_create_session(cuurent_session_id)
    # 生成回复
    response = generate_response(combined_text, chat_history=chat_history)
    # 转语音
    tts_path = await text_to_speech(response["reply"])
    with open(tts_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    os.remove(tts_path)
    # 保存对话记录
    save_chat(combined_text, response["intent"], response["reply"])
    # 保存会话历史
    save_session(chat_history, cuurent_session_id)
    return {
        "session_id": cuurent_session_id,
        "transcript": transcript,
        "intent": response["intent"],
        "slots": response["slots"],
        "reply": response["reply"],
        "audio_base64": audio_base64
    }

@router.post("/api/session/create")
async def create_session():
    """ 创建新会话 """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@router.delete("/api/session/delete")
async def delete_session(session_id: str = Form(...)):
    """ 删除会话 """
    delete_session_history(session_id)
    return {"message": "会话已删除"}

@router.get("/api/session/history")
async def get_session_history(session_id: str = Query(...)):
    """ 获取会话历史 """
    history = load_session_history(session_id)
    return {
        "session_id": session_id,
        "history": history
    }

@router.delete("/api/session/clear")
async def clear_session(session_id: str = Form(...)):
    """ 清空会话历史 """
    delete_session_history(session_id)
    return {"message": "会话历史已清空"}


