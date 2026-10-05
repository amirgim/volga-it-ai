"""
Setup script: creates the entire project structure.
Run: python setup_project.py
"""
import os
from pathlib import Path

ROOT = Path("solution")

DIRS = [
    "src",
    "train",
    "generator",
    "models",
    "docs",
    "dataset/images/real",
    "dataset/images/synthetic",
    "dataset/labels",
    "dataset/generator",
]

FILES = {
    "requirements.txt": """ultralytics==8.3.0
torch==2.4.0
torchvision==0.19.0
onnx==1.16.0
onnxruntime-gpu==1.19.0
opencv-python==4.10.0.84
numpy==1.26.4
pillow==10.4.0
pandas==2.2.2
albumentations==1.4.14
""",
    "generator/requirements.txt": """numpy==1.26.4
pillow==10.4.0
""",
    "dataset/data.yaml": """path: ../dataset
train: images/real
val: images/synthetic
names:
  0: type1
  1: type1a
  2: type1b
  3: other
""",
    "dataset/meta.csv": "image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions\n",
    "src/__init__.py": "",
    "models/.gitkeep": "",
    "dataset/generator/README.md": "Generator is shipped in `../../generator/`. See its README for usage.\n",
}

def main():
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for rel, content in FILES.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"created {p}")
    print("\nDone. Project created in:", ROOT.resolve())
    print("\nNext steps:")
    print("  cd solution")
    print("  pip install -r requirements.txt")
    print("  pip install -r generator/requirements.txt")

if __name__ == "__main__":
    main()
