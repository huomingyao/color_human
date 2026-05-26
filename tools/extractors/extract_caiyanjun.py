"""
提取蔡岩峻相关资料的文本内容
"""
import os
from pathlib import Path
from docx import Document


def extract_docx_text(docx_path: str) -> str:
    """从docx文件提取纯文本"""
    doc = Document(docx_path)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n".join(paragraphs)


def main():
    base_dir = Path("H:/蔡岩峻相关信息")

    all_text = []

    # 1. 清云宣传片
    print("处理: 清云宣传片...")
    qingyun_dir = base_dir / "清云宣传片"
    if qingyun_dir.exists():
        for f in qingyun_dir.glob("*.docx"):
            print(f"  - {f.name}")
            text = extract_docx_text(str(f))
            if text:
                all_text.append(f"# {f.stem}\n{text}\n")

    # 2. 活动设计
    print("处理: 活动设计...")
    activity_dir = base_dir / "活动参考" / "活动设计"
    if activity_dir.exists():
        for f in activity_dir.glob("*.docx"):
            print(f"  - {f.name}")
            text = extract_docx_text(str(f))
            if text:
                all_text.append(f"# {f.stem}\n{text}\n")

    # 3. 知守入职问卷（最有价值的个人资料）
    print("处理: 知守入职问卷...")
    third_semester = base_dir / "第三学期"
    if third_semester.exists():
        for f in third_semester.glob("蔡岩峻行知入职*.docx"):
            print(f"  - {f.name}")
            text = extract_docx_text(str(f))
            if text:
                all_text.append(f"# {f.stem}\n{text}\n")

    # 4. 第二学期 - 规则
    print("处理: 规则...")
    second_rules_dir = base_dir / "第二学期" / "规则"
    if second_rules_dir.exists():
        for f in second_rules_dir.glob("*.docx"):
            if f.name != "desktop.ini":
                print(f"  - {f.name}")
                text = extract_docx_text(str(f))
                if text:
                    all_text.append(f"# {f.stem}\n{text}\n")

    # 5. 第三学期 - 规则
    print("处理: 第三学期规则...")
    third_rules_dir = base_dir / "第三学期" / "规则"
    if third_rules_dir.exists():
        for f in third_rules_dir.glob("*.docx"):
            if f.name != "desktop.ini":
                print(f"  - {f.name}")
                text = extract_docx_text(str(f))
                if text:
                    all_text.append(f"# {f.stem}\n{text}\n")

    # 6. 第三学期 - 行知问卷汇总
    print("处理: 行知问卷...")
    questionnaire_files = [
        third_semester / "行知问卷",
        third_semester
    ]
    for q_dir in questionnaire_files:
        if q_dir.exists():
            for f in sorted(q_dir.glob("*.docx"))[:5]:  # 取前5个
                print(f"  - {f.name}")
                text = extract_docx_text(str(f))
                if text and len(text) > 100:
                    all_text.append(f"# {f.stem}\n{text}\n")

    # 合并所有文本
    combined_text = "\n".join(all_text)

    # 保存到文件
    output_file = "H:/蔡岩峻相关信息/蔡岩峻文本集.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"\n✅ 已提取文本到: {output_file}")
    print(f"   总字符数: {len(combined_text)}")


if __name__ == "__main__":
    main()