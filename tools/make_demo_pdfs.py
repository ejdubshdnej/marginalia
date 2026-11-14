"""生成迁移演示用的两份样例 PDF（内容为虚构，不引用任何第三方文档）。

它们的作用是给「换一份从没见过的文档」这件事提供可复现的输入：

  data/demo/style_guide_en.pdf   英文，设计规范说明书，6 个配色方案
  data/demo/speaker_manual_zh.pdf 中文，产品说明书，含表格状规格清单

两份都是本仓库原创的虚构内容，可以随仓库一起分发。

用法：
  .venv/Scripts/python.exe -X utf8 tools/make_demo_pdfs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo"

CJK = "china-s"
LATIN = "helv"
LATIN_BOLD = "hebo"

# ---------------------------------------------------------------- 英文配色说明书

THEMES = [
    {
        "name": "Harbor Blue",
        "mood": "calm and professional, suited to analytic work",
        "colors": [("Deep Navy", "#16202e"), ("Harbor Teal", "#2f8f8f"), ("Mist", "#a9d6d8"), ("Paper", "#f4f9f8")],
        "font": "Liberation Sans Bold",
        "best_for": "Quarterly reviews, audit reports, policy briefings",
    },
    {
        "name": "Amber Field",
        "mood": "warm and energetic, draws the eye to calls to action",
        "colors": [("Burnt Sienna", "#e2703a"), ("Apricot", "#f2a65a"), ("Sand", "#e8c88a"), ("Slate", "#2b3a42")],
        "font": "Liberation Serif Bold",
        "best_for": "Campaign decks, event promotion, consumer product launches",
    },
    {
        "name": "Pine Hollow",
        "mood": "grounded and quiet, reads well in long sessions",
        "colors": [("Fir Green", "#274a2d"), ("Sage", "#7d8a6f"), ("Moss", "#a8b394"), ("Chalk", "#f7f6f1")],
        "font": "FreeSerif Bold",
        "best_for": "Field research, sustainability reporting, wellness material",
    },
    {
        "name": "Graph Paper",
        "mood": "neutral and precise, keeps attention on the data",
        "colors": [("Ink", "#2f3437"), ("Graphite", "#6b7280"), ("Rule", "#d1d5db"), ("White", "#ffffff")],
        "font": "Liberation Sans Bold",
        "best_for": "Engineering reviews, architecture diagrams, method sections",
    },
    {
        "name": "Copper Dusk",
        "mood": "rich and subdued, gives weight to conclusions",
        "colors": [("Ochre", "#b8860b"), ("Copper", "#a35c4a"), ("Wheat", "#d9c7a7"), ("Umber", "#4a3b32")],
        "font": "Liberation Serif Bold",
        "best_for": "Annual reports, hospitality, heritage brands",
    },
    {
        "name": "Night Signal",
        "mood": "high contrast, built for dark rooms and projectors",
        "colors": [("Void", "#12121a"), ("Signal Cyan", "#00d0ff"), ("Violet", "#8a7bd8"), ("Ash", "#e8e8ee")],
        "font": "Liberation Sans Bold",
        "best_for": "Demo days, conference talks, product announcements",
    },
]


def build_style_guide(path: Path) -> int:
    doc = pymupdf.open()
    for index, theme in enumerate(THEMES):
        page = doc.new_page(width=595, height=842)
        y = 80
        page.insert_text((60, y), theme["name"], fontsize=26, fontname=LATIN_BOLD)
        y += 26
        page.insert_text((60, y), theme["mood"], fontsize=11, fontname=LATIN)
        y += 42

        page.insert_text((60, y), "PALETTE", fontsize=9, fontname=LATIN_BOLD)
        y += 20
        for label, hexcode in theme["colors"]:
            rect = pymupdf.Rect(60, y - 11, 84, y + 7)
            rgb = tuple(int(hexcode[i: i + 2], 16) / 255 for i in (1, 3, 5))
            page.draw_rect(rect, color=None, fill=rgb)
            page.insert_text((96, y), f"{label}", fontsize=11, fontname=LATIN)
            page.insert_text((300, y), hexcode, fontsize=11, fontname=LATIN)
            y += 26

        y += 18
        page.insert_text((60, y), "TYPOGRAPHY", fontsize=9, fontname=LATIN_BOLD)
        y += 20
        page.insert_text((60, y), f"Headings use {theme['font']}.", fontsize=11, fontname=LATIN)
        y += 18
        page.insert_text((60, y), "Body text uses the default document serif.", fontsize=11, fontname=LATIN)
        y += 40

        page.insert_text((60, y), "RECOMMENDED FOR", fontsize=9, fontname=LATIN_BOLD)
        y += 20
        page.insert_text((60, y), theme["best_for"], fontsize=11, fontname=LATIN)

        page.insert_text((60, 800), f"{index + 1} of {len(THEMES)}", fontsize=9, fontname=LATIN)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return len(THEMES)


# ---------------------------------------------------------------- 中文产品说明

SPECS = [
    ("型号", "YUQUE X1"),
    ("颜色", "石墨灰 / 米白"),
    ("尺寸", "直径 98 毫米，高 156 毫米"),
    ("净重", "620 克"),
    ("电池容量", "5200 毫安时"),
    ("续航时间", "连续播放 14 小时，待机 10 天"),
    ("充电时间", "约 2.5 小时（15V 2A 适配器）"),
    ("扬声器", "3 英寸全频单元，额定功率 20 瓦"),
    ("频响范围", "55 赫兹至 20 千赫兹"),
    ("无线连接", "蓝牙 5.3、双频 Wi-Fi（2.4G / 5G）"),
    ("麦克风", "四麦克风环形阵列，支持 5 米远场唤醒"),
    ("防护等级", "IPX5，防溅水，不可浸水"),
    ("工作温度", "0 摄氏度至 40 摄氏度"),
    ("保修期", "整机 12 个月，电池 6 个月"),
]

SECTIONS = [
    (
        "一、开箱与首次开机",
        "取出音箱后，先接上电源适配器充电。机身顶部指示灯为红色时表示电量低于 20%，"
        "充满后转为绿色常亮。首次开机请长按顶部电源键 3 秒，听到提示音即表示启动完成。"
        "若指示灯快速闪烁橙色，说明未检测到配网状态，请按第四节重新配网。",
    ),
    (
        "二、配网与账号绑定",
        "音箱只支持 2.4G 与 5G 双频 Wi-Fi，不支持需要网页认证的公共网络。"
        "打开手机应用，选择「添加设备」，按提示输入 Wi-Fi 密码。配网过程中指示灯为蓝色呼吸状态，"
        "成功后蓝色常亮 2 秒并播报「网络已连接」。若连续三次配网失败，请将路由器加密方式改为 WPA2。",
    ),
    (
        "三、语音指令与远场唤醒",
        "唤醒词为「你好，云雀」。四麦克风环形阵列在安静环境下可覆盖 5 米半径，"
        "环境噪声超过 65 分贝时有效距离降到约 2 米。音箱内置回声消除，播放音乐时仍可正常唤醒，"
        "但音量高于 70% 时识别率会明显下降，建议降低音量后重试。",
    ),
    (
        "四、常见故障与处理",
        "无法开机：先用 15V 2A 适配器充电 30 分钟再试，普通 5V 充电器功率不足，"
        "可能无法唤醒深度放电的电池。无声音：检查是否处于静音状态，机身侧面的静音键按下后指示灯为红色常亮。"
        "频繁断连：确认音箱与路由器距离在 10 米以内，且中间没有承重墙阻隔。"
        "若问题依旧，长按电源键 10 秒恢复出厂设置，此操作会清除已绑定的账号信息。",
    ),
    (
        "五、保修与售后",
        "整机保修 12 个月，电池保修 6 个月，均自签收之日起计算。"
        "以下情况不在保修范围内：私自拆机、进水、跌落导致的屏幕或外壳破损、使用非原装适配器造成的损坏。"
        "申请售后时需提供订单号与设备底部的序列号。人为损坏的维修按实际更换部件收费。",
    ),
]


def build_speaker_manual(path: Path) -> int:
    doc = pymupdf.open()

    # 封面
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 200), "云雀智能音箱 X1", fontsize=28, fontname=CJK)
    page.insert_text((60, 240), "使用说明", fontsize=20, fontname=CJK)
    page.insert_text((60, 300), "本说明书适用于型号 YUQUE X1", fontsize=12, fontname=CJK)
    page.insert_text((60, 780), "文档版本 1.2", fontsize=10, fontname=CJK)

    # 规格页
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 80), "产品规格", fontsize=18, fontname=CJK)
    y = 120
    for label, value in SPECS:
        page.insert_text((60, y), f"{label}：", fontsize=11, fontname=CJK)
        page.insert_text((150, y), value, fontsize=11, fontname=CJK)
        y += 28

    # 正文页
    for heading, body in SECTIONS:
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 80), heading, fontsize=16, fontname=CJK)
        rect = pymupdf.Rect(60, 100, 535, 740)
        page.insert_textbox(rect, body, fontsize=11, fontname=CJK, lineheight=1.6)

    path.parent.mkdir(parents=True, exist_ok=True)
    pages = doc.page_count
    doc.save(path)
    doc.close()
    return pages


if __name__ == "__main__":
    en = OUT / "style_guide_en.pdf"
    zh = OUT / "speaker_manual_zh.pdf"
    n_en = build_style_guide(en)
    n_zh = build_speaker_manual(zh)
    print(f"已生成 {en.relative_to(ROOT)}：{n_en} 页，{en.stat().st_size} 字节")
    print(f"已生成 {zh.relative_to(ROOT)}：{n_zh} 页，{zh.stat().st_size} 字节")
