"""
兼容入口：该文件保留旧名称，但逻辑已迁移到 similarity_pipeline.py。
"""

try:
    from .similarity_pipeline import main
except ImportError:  # 直接运行脚本时的相对导入
    from similarity_pipeline import main


if __name__ == "__main__":
    main()
