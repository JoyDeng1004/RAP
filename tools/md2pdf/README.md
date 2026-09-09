# md2pdf

把带 LaTeX 公式和图片的 Markdown 导出成 PDF。离线,不需要 pandoc / LaTeX。

链路:`markdown-it`(转 HTML)→ 本地 MathJax(渲染公式)→ 图片转 base64 内嵌 → 无头 Chrome 打印 PDF。

## 用法

```bash
# 输出到同名 .pdf(推荐)
./tools/md2pdf/md2pdf.sh docs/xxx.md

# 或指定输出路径
./tools/md2pdf/md2pdf.sh docs/xxx.md /path/to/out.pdf
```

改一次 md,重跑同一条命令即可重导。首次运行会自动 `npm install`(markdown-it + mathjax),之后复用。

## 依赖

- Node.js(`node` / `npm`)
- Google Chrome 或 Chromium
  - 自动探测常见路径;找不到时用 `CHROME=/path/to/chrome ./tools/md2pdf/md2pdf.sh ...` 覆盖。

## 说明

- Markdown 里的 `$...$` / `$$...$$` 会被渲染成公式;图片路径按 md 文件所在目录解析,并内嵌进 PDF,所以 PDF 可单独分发。
- 公式里请用 `\text{中文}`,不要用 `\textbf{中文}`(MathJax 对后者 + 中文支持不佳)。
- `node_modules/` 和临时 `*.html` 已 gitignore。
