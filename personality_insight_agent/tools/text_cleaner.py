"""
文本清洗工具

在送入 Skill 1 (LLM) 之前进行本地预处理。
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """基础文本清洗"""
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[\s- -‏ - ]+$", "", text, flags=re.MULTILINE)
    return text.strip()


def remove_timestamps(text: str) -> str:
    """去除常见时间戳格式"""
    patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(:\d{2})?",
        r"\[\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}\]",
        r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?",
        r"(上午|下午|凌晨|早上|中午|晚上)\s*\d{1,2}:\d{2}",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text


def remove_system_messages(text: str) -> str:
    """去除常见系统消息"""
    patterns = [
        r".*加入了[群聊|群组|房间].*",
        r".*退出了[群聊|群组|房间].*",
        r".*撤回了一条消息.*",
        r".*修改群名.*",
        r".*邀请.*加入了.*",
        r".*拍了拍.*",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)
    return text


def replace_urls(text: str) -> str:
    """URL替换为占位符"""
    return re.sub(r"https?://\S+", "<URL>", text)


def remove_emoji_only_lines(text: str) -> str:
    """去除纯表情行"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r"^[\U0001f000-\U0001ffff☀-⟿⭐\s]+$", stripped):
            cleaned.append(line)
    return "\n".join(cleaned)


def extract_messages(text: str) -> list[dict]:
    """从聊天记录提取消息列表"""
    lines = text.strip().split("\n")
    messages = []
    current_speaker = None
    current_text: list[str] = []

    speaker_pattern = re.compile(r"^(.+?)[:：]\s*(.+)")

    for line in lines:
        m = speaker_pattern.match(line)
        if m:
            if current_speaker and current_text:
                messages.append({"speaker": current_speaker, "text": "\n".join(current_text)})
            current_speaker = m.group(1).strip()
            current_text = [m.group(2)]
        else:
            if current_speaker:
                current_text.append(line)

    if current_speaker and current_text:
        messages.append({"speaker": current_speaker, "text": "\n".join(current_text)})

    return messages


def count_words(text: str) -> int:
    """统计有效字数（中文按字符，英文按词）"""
    chinese = len(re.findall(r"[一-鿿]", text))
    english = len(re.findall(r"[a-zA-Z]+", text))
    return chinese + english


def full_clean(text: str) -> str:
    """完整清洗流程"""
    text = clean_text(text)
    text = remove_timestamps(text)
    text = remove_system_messages(text)
    text = replace_urls(text)
    text = remove_emoji_only_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
