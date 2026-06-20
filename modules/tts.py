import edge_tts # Edge浏览器自带的语音合成工具
import tempfile # 临时文件

# 语音合成的参数 "中文女声 晓晓"
VOICE = "zh-CN-XiaoxiaoNeural"

# 语音合成的函数
async def text_to_speech(text: str) -> bytes:
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,   # 关闭自动删除
        suffix='.mp3'   # 指定文件后缀
    )
    temp_file.close()   # 关闭文件句柄，后面edge_tts直接往这个文件中写数据
    tts_engine = edge_tts.Communicate(text, VOICE)
    await tts_engine.save(temp_file.name)
    return temp_file.name

if __name__ == "__main__":
    import asyncio
    path = asyncio.run(text_to_speech("你好，我是小助手，很高兴为你服务。"))
    print(path)