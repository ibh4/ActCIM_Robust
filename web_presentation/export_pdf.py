#!/usr/bin/env python3
"""将演示网页导出为 PDF：每一页幻灯片对应一页 PDF。

原理：headless Chrome 按 #page-N 逐页渲染截图（2x 缩放，3840x2160），
再用 pymupdf 合成为 16:9（13.333x7.5 英寸）的 PDF。

用法：python3 export_pdf.py  （在 web_presentation/ 目录下运行）
输出：ActCIM_web_slides.pdf
"""
import pathlib
import shutil
import subprocess
import sys

import fitz  # pymupdf

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HERE = pathlib.Path(__file__).resolve().parent
INDEX = HERE / "index.html"
OUT_PDF = HERE / "ActCIM_web_slides.pdf"
TMP = HERE / ".tmp_export"
N_SLIDES = 14
# 16:9 页面，与幻灯片同比例（单位 pt，1in=72pt）
PAGE_W, PAGE_H = 13.333 * 72, 7.5 * 72


def shoot(page_no: int, out_png: pathlib.Path):
    url = INDEX.as_uri() + f"#page-{page_no}"
    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=1920,1080", "--force-device-scale-factor=2",
        "--virtual-time-budget=5000",  # 等待入场动画与图片加载完成
        f"--screenshot={out_png}", url,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    TMP.mkdir(exist_ok=True)
    pngs = []
    for i in range(1, N_SLIDES + 1):
        png = TMP / f"page_{i:02d}.png"
        shoot(i, png)
        if not png.exists():
            sys.exit(f"截图失败：第 {i} 页")
        pngs.append(png)
        print(f"  已渲染 {i}/{N_SLIDES}")

    doc = fitz.open()
    for png in pngs:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, filename=str(png))
    doc.save(OUT_PDF, deflate=True)
    doc.close()
    shutil.rmtree(TMP)
    print(f"完成：{OUT_PDF}（{N_SLIDES} 页，16:9）")


if __name__ == "__main__":
    main()
