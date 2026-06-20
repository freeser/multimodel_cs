import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 根路径
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent

class ProductRAG:
    def __init__(self, kb_path: str = str(ROOT_PATH / "data" / "knowledge_base.json")):
        """
        初始化产品知识库RAG系统
        kb_path: 知识库文件路径
        """
        # 加载嵌入模型
        self.encoder = SentenceTransformer(str(ROOT_PATH / "models" / "bge_small_zh"))
        # 初始化chromadb客户端 allow_reset=True 允许重置数据库  anonymized_telemetry=False 禁用匿名遥测
        self.client = chromadb.Client(Settings(allow_reset=True, anonymized_telemetry=False))
        # 获取或新建向量知识库
        self.collenction = self.client.get_or_create_collection(name="product_kb")
        if self.collenction.count() == 0 and os.path.exists(kb_path):
            self._load_knowledge_base(kb_path)

    # 加载知识库
    def _load_knowledge_base(self, kb_path: str):
        """
        从JSON文件加载知识库
        kb_path: 知识库文件路径
        """
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        docs = []
        metas = []
        ids = []
        embeddings = []
        # 类似于 项目1 中，使用langchain框架进行 文档加载 分块 嵌入 存储 的过程
        for idx, item in enumerate(kb_data):
            text = f"{item['title']}\n{item['content']}"
            docs.append(text)
            metas.append({"source": item['title']})
            ids.append(str(idx))
            embeddings.append(self.encoder.encode(text).tolist())
        self.collenction.add(embeddings=embeddings, documents=docs, metadatas=metas, ids=ids)

    # 检索
    def retrieve(self, query: str, top_k: int = 2):
        """
        检索最相似的文档
        query: 查询文本
        top_k: 返回的文档数量
        """
        if self.collenction.count() == 0:
            return ""
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collenction.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    

# 测试
# if __name__ == "__main__":
#     rag = ProductRAG()
#     print(rag.retrieve("蕾丝 A 字裙"))