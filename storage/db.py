import sqlite3
from config import SQLITE_DB

def init_db():
    """ 初始化连接 与 建表 """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    # 原有对话记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            intent TEXT,
            reply TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 新增会话历史表 （用于持久化会话）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 添加索引
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON session_history (session_id)")

    conn.commit()
    conn.close()

def save_chat(user_input: str, intent: str, reply: str):
    """ 保存对话记录 """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (user_input, intent, reply) VALUES (?, ?, ?)",
        (user_input, intent, reply)
    )
    conn.commit()
    conn.close()

def save_session_history(session_id: str, role: str, content: str):
    """ 保存会话历史 """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO session_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def load_session_history(session_id: str):
    """ 加载会话历史 """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM session_history WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    )
    history = cursor.fetchall()
    conn.close()
    return [{"role":msg[0], "content": msg[1]} for msg in history]

def delete_session_history(session_id: str):
    """ 删除会话历史 """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM session_history WHERE session_id = ?",
        (session_id,)
    )
    conn.commit()
    conn.close()

def get_all_sessions():
    """ 获取所有会话ID """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM session_history")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sessions

