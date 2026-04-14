import sys
import os
import logging
import config

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Suppress OTel "Failed to detach context" warning caused by generator/context interaction.
# Tracing is unaffected.
# Known bug: https://github.com/open-telemetry/opentelemetry-python/issues/2606
class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    print("\n🔨 Creating RAG Assistant...")
    demo = create_gradio_ui()
    print("\n🚀 Launching RAG Assistant...")

    # # 获取图片存放目录的绝对路径
    # image_dir = os.path.abspath(os.path.join(config.MARKDOWN_DIR, "images"))
    
    # # 启动时添加 allowed_paths
    # demo.launch(
    #     # css=custom_css, 
    #     allowed_paths=[image_dir]  # 允许 Gradio 读取该目录下的图片并在聊天框渲染
    # )
    # demo.launch(css=custom_css)
    demo.launch()