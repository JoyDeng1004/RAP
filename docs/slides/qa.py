"""几何 / 溢出 / 字体一致性检查（无 LibreOffice 环境下的替代 QA）。"""
import sys, unicodedata
from pptx import Presentation
from pptx.util import Emu

DECK = sys.argv[1] if len(sys.argv) > 1 else "RAP_DriveWeave_0826.pptx"
SW, SH = 13.333, 7.5
MARGIN = 0.35          # 允许的最小边距
GAP_MIN = 0.06         # 元素间最小间隙

def inches(v):
    return None if v is None else Emu(v).inches

def char_w(ch, pt):
    """近似字符宽度（英寸）。"""
    ea = unicodedata.east_asian_width(ch)
    if ea in ("W", "F"):
        return pt / 72.0
    if ch == " ":
        return pt / 72.0 * 0.30
    if ch.isdigit() or ch in ".,:%()[]/+-":
        return pt / 72.0 * 0.50
    return pt / 72.0 * 0.53

def est_height(tf, box_w):
    """估算 text frame 所需高度（英寸）。"""
    total = 0.0
    for p in tf.paragraphs:
        runs = [(r.text, (r.font.size.pt if r.font.size else 12)) for r in p.runs]
        if not runs:
            total += 12 / 72.0 * 1.25
            continue
        pt = max(s for _, s in runs)
        w = sum(char_w(c, s) for t, s in runs for c in t)
        avail = max(box_w - 0.06, 0.3)
        lines = max(1, -(-int(w * 1000) // int(avail * 1000)))
        # 显式换行
        lines += sum(t.count("\n") for t, _ in runs)
        total += lines * pt / 72.0 * 1.25
        total += 7 / 72.0        # paraSpaceAfter 近似
    return total

problems = []
prs = Presentation(DECK)
for i, slide in enumerate(prs.slides, start=1):
    boxes = []
    for sh in slide.shapes:
        x, y = inches(sh.left), inches(sh.top)
        w, h = inches(sh.width), inches(sh.height)
        if None in (x, y, w, h):
            continue
        kind = "text" if sh.has_text_frame and sh.text_frame.text.strip() else sh.shape_type
        label = (sh.text_frame.text.strip().replace("\n", " ")[:34] if sh.has_text_frame else str(sh.shape_type))

        # 出界 / 边距
        if x < MARGIN - 1e-6 or y < MARGIN - 1e-6 or x + w > SW - MARGIN + 1e-6 or y + h > SH - MARGIN + 1e-6:
            problems.append(f"p{i:02d} 越界/贴边  [{label}]  x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f} 右={x+w:.2f} 下={y+h:.2f}")

        # 文本溢出
        if sh.has_text_frame and sh.text_frame.text.strip():
            need = est_height(sh.text_frame, w)
            if need > h + 0.06:
                problems.append(f"p{i:02d} 文本可能溢出 [{label}]  盒高={h:.2f} 估算={need:.2f}")
            if not label.isdigit():          # 页码不参与重叠检测
                boxes.append((x, y, w, max(h, need), label))
        elif kind not in ("text",):
            boxes.append((x, y, w, h, label))

        # 字体 / 颜色一致性
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.name and r.font.name != "Segoe UI":
                        problems.append(f"p{i:02d} 字体非 Segoe UI [{label}] -> {r.font.name}")
                    try:
                        if r.font.color and r.font.color.rgb is not None and str(r.font.color.rgb) != "000000":
                            problems.append(f"p{i:02d} 文字非黑色 [{label}] -> {r.font.color.rgb}")
                    except Exception:
                        pass

    # 重叠检测（仅文本框之间）
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            ax, ay, aw, ah, al = boxes[a]
            bx, by, bw, bh, bl = boxes[b]
            ox = min(ax + aw, bx + bw) - max(ax, bx)
            oy = min(ay + ah, by + bh) - max(ay, by)
            if ox > GAP_MIN and oy > GAP_MIN:
                problems.append(f"p{i:02d} 文本框重叠  [{al}] x [{bl}]  重叠 {ox:.2f}x{oy:.2f} in")

print(f"共 {len(prs.slides._sldIdLst)} 页")
if problems:
    print(f"发现 {len(problems)} 处：")
    for p in problems:
        print("  " + p)
else:
    print("未发现几何/溢出/字体问题")
