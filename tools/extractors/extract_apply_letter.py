"""提取蔡岩峻的个人申请信"""
import os
from docx import Document


def extract_docx_text(docx_path: str) -> str:
    """从docx文件提取纯文本"""
    doc = Document(docx_path)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n".join(paragraphs)


# 提取蔡岩峻知守学堂申请信
input_file = "H:/蔡岩峻相关信息/第二学期/蔡岩峻 知守学堂申请信.docx"
text = extract_docx_text(input_file)

print("=" * 60)
print("蔡岩峻 知守学堂申请信")
print("=" * 60)
print(text)
print("=" * 60)
print(f"\n总字符数: {len(text)}")

# 保存
output_file = "H:/蔡岩峻相关信息/蔡岩峻_申请信.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)
print(f"✅ 已保存到: {output_file}")