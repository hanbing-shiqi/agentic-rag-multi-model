import os
import shutil
import config
import pymupdf.layout
from pathlib import Path
import tiktoken
import pymupdf
import glob

import pymupdf4llm



def clear_directory_contents(directory: Path) -> None:
    """Delete everything under directory but not the directory itself."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    
    # 建立一个统一的图片存放目录
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 开启 write_images，让 PyMuPDF4LLM 提取出图片并保存在 images_dir 下
    md = pymupdf4llm.to_markdown(
        doc, 
        header=False, 
        footer=False, 
        page_separators=True, 
        ignore_images=False,        # 必须设为 False 
        write_images=True,          # 必须设为 True 以保存图片
        image_path=str(images_dir)  # Markdown 中图片链接指向该路径
    )
    
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)