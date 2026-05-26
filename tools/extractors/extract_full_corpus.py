"""
提取蔡岩峻所有资料，完整版
"""
import os
from pathlib import Path
from docx import Document


def extract_docx_text(docx_path: str) -> str:
    """从docx文件提取纯文本"""
    try:
        doc = Document(docx_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"  ⚠️ 读取失败: {e}")
        return ""


def scan_directory(base_dir: Path, pattern: str = "*.docx") -> list[tuple[str, str]]:
    """递归扫描目录下所有文件"""
    files = []
    if base_dir.exists():
        for f in base_dir.rglob(pattern):
            # 跳过系统文件和临时文件
            if "desktop.ini" in f.name.lower() or "~" in f.name or f.stat().st_size < 100:
                continue
            files.append((f"docx:{f.relative_to(base_dir.parent)}", str(f)))
    return files


def main():
    base_dir = Path("H:/蔡岩峻相关信息")
    output_dir = base_dir / "corpus"
    output_dir.mkdir(exist_ok=True)

    all_files = []

    print("📁 扫描所有资料...")
    print("="*60)

    # 1. 清云宣传片
    print("[1/6] 清云宣传片...")
    for f in (base_dir / "清云宣传片").glob("*.docx"):
        text = extract_docx_text(str(f))
        if text and len(text) > 50:
            # 只取蔡岩峻写的部分
            if "蔡岩峻" in text[:500]:
                all_files.append(("宣传片-蔡岩峻", text[:3000]))
                print(f"  ✓ {f.name} ({len(text)}字符)")
            elif "何锐鹏" in f.name:
                all_files.append(("宣传片-案例", text[:2000]))
                print(f"  ✓ {f.name}")

    # 2. 活动设计
    print("[2/6] 活动设计...")
    activity_dir = base_dir / "活动参考" / "活动设计"
    if activity_dir.exists():
        for f in activity_dir.glob("*.docx"):
            text = extract_docx_text(str(f))
            if text and len(text) > 100:
                all_files.append((f"活动-{f.stem}", text))
                print(f"  ✓ {f.name} ({len(text)}字符)")

    # 3. 知守申请信（核心）
    print("[3/6] 知守申请信（核心材料）...")
    apply_letter = base_dir / "第二学期" / "蔡岩峻 知守学堂申请信.docx"
    if apply_letter.exists():
        text = extract_docx_text(str(apply_letter))
        if text:
            all_files.append(("申请信", text))
            print(f"  ✓ 知守学堂申请信 ({len(text)}字符)")

    # 4. 规则文件
    print("[4/6] 规则文件...")
    for sem in ["第一学期", "第二学期", "第三学期", "第四学期"]:
        rule_dir = base_dir / sem / "规则"
        if rule_dir.exists():
            for f in sorted(rule_dir.glob("*.docx"))[:3]:
                text = extract_docx_text(str(f))
                if text and 100 < len(text) < 5000:
                    all_files.append((f"{sem}-规则", text))
                    print(f"  ✓ {sem}/{f.name}")

    # 5. 第三学期-问卷
    print("[5/6] 问卷材料...")
    for f in list((base_dir / "第三学期").glob("*问卷*.docx"))[:3]:
        text = extract_docx_text(str(f))
        if text and len(text) > 200:
            all_files.append((f"问卷-{f.stem}", text[:1500]))
            print(f"  ✓ {f.name}")

    # 6. 书��/作业
    print("[6/6] 书籍作业...")
    for sem in ["第二学期", "第三学期"]:
        book_dir = base_dir / sem / "书籍"
        if book_dir.exists():
            for f in sorted(book_dir.glob("*.docx"))[:2]:
                text = extract_docx_text(str(f))
                if text and 200 < len(text) < 3000:
                    all_files.append((f"书籍-{f.stem[:20]}", text))
                    print(f"  ✓ {sem}/{f.name}")

    # 输出所有素材
    print("\n" + "="*60)
    print(f"📦 共提取 {len(all_files)} 个文件")
    total_chars = sum(len(t[1]) for t in all_files)
    print(f"📏 总字符数: {total_chars}")

    # 保存素材索引
    corpus_index = []
    for name, content in all_files:
        truncated = content[:2000] + "..." if len(content) > 2000 else content
        corpus_index.append({"source": name, "content": truncated})

    import json
    index_file = output_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(corpus_index, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 素材索引已保存: {index_file}")

    # 打印各文件预览
    print("\n" + "="*60)
    print("素材预览:")
    print("="*60)
    for i, (name, content) in enumerate(all_files[:5]):
        print(f"\n--- {name} ---")
        print(content[:300])
    print(f"\n... 共 {len(all_files)} 个文件")


if __name__ == "__main__":
    main()