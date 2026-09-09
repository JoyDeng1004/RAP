const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');

const mdPath = process.argv[2];
const outHtml = process.argv[3];
const baseDir = path.dirname(mdPath);

let src = fs.readFileSync(mdPath, 'utf8');

// 1) Protect math BEFORE markdown so underscores/asterisks/backslashes survive.
//    Placeholder is pure alphanumeric so markdown-it never touches it and never
//    trims surrounding whitespace away.
const math = [];
src = src.replace(/\$\$([\s\S]+?)\$\$/g, (m, g) => {
  const i = math.length; math.push({ display: true, tex: g });
  return 'zzmathzz' + i + 'zz';
});
src = src.replace(/\$([^$\n]+?)\$/g, (m, g) => {
  const i = math.length; math.push({ display: false, tex: g });
  return 'zzmathzz' + i + 'zz';
});
// Also protect LaTeX-native delimiters: \[ ... \] (display) and \( ... \) (inline).
src = src.replace(/\\\[([\s\S]+?)\\\]/g, (m, g) => {
  const i = math.length; math.push({ display: true, tex: g });
  return 'zzmathzz' + i + 'zz';
});
src = src.replace(/\\\(([\s\S]+?)\\\)/g, (m, g) => {
  const i = math.length; math.push({ display: false, tex: g });
  return 'zzmathzz' + i + 'zz';
});

// 2) Markdown -> HTML
const md = new MarkdownIt({ html: true, linkify: false, typographer: false });
let html = md.render(src);

// 3) Restore math as \(...\) / \[...\]  (escape & < > so MathJax reads raw tex)
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
html = html.replace(/zzmathzz(\d+)zz/g, (m, i) => {
  const { display, tex } = math[+i];
  return display ? ('\\[' + esc(tex) + '\\]') : ('\\(' + esc(tex) + '\\)');
});

// 4) Inline images as base64 data URIs (so PDF is self-contained)
html = html.replace(/<img([^>]*?)src="([^"]+)"([^>]*)>/g, (m, a, srcAttr, b) => {
  const p = srcAttr.replace(/^\.\//, '');
  const abs = path.resolve(baseDir, p);
  if (fs.existsSync(abs)) {
    const ext = path.extname(abs).slice(1).toLowerCase();
    const mime = ext === 'png' ? 'image/png' : (ext === 'jpg' || ext === 'jpeg') ? 'image/jpeg' : 'image/' + ext;
    const data = fs.readFileSync(abs).toString('base64');
    return '<img' + a + 'src="data:' + mime + ';base64,' + data + '"' + b + '>';
  }
  console.warn('WARN: image not found:', abs);
  return m;
});

const mathjaxSrc = 'file://' + path.resolve(__dirname, 'node_modules/mathjax/es5/tex-mml-chtml.js');

const page = `<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<script>
window.MathJax = {
  tex: { inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] },
  svg: { fontCache: 'global' },
  startup: { typeset: true }
};
</script>
<script src="${mathjaxSrc}"></script>
<style>
  @page { size: A4; margin: 16mm 15mm; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body {
    font-family: "PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",-apple-system,sans-serif;
    font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
  }
  h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 0; }
  h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 20px; }
  h3 { font-size: 12pt; margin-top: 16px; }
  code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 9pt;
         font-family: "SF Mono",Menlo,Consolas,monospace; }
  pre { background: #f6f6f6; padding: 10px 12px; border-radius: 5px; overflow-x: auto;
        font-size: 8.5pt; line-height: 1.35; }
  pre code { background: none; padding: 0; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
  th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; vertical-align: top; }
  th { background: #efefef; font-weight: 600; }
  blockquote { border-left: 3px solid #d0d0d0; margin: 10px 0; padding: 2px 12px; color: #555; background:#fafafa; }
  img { max-width: 100%; height: auto; display: block; margin: 12px auto; }
  hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
  mjx-container { overflow-x: auto; overflow-y: hidden; }
  mjx-container[display="true"] { margin: 8px 0; }
  h2, h3 { break-after: avoid; }
  table, pre, img, mjx-container[display="true"] { break-inside: avoid; }
</style></head>
<body>
${html}
</body></html>`;

fs.writeFileSync(outHtml, page, 'utf8');
console.log('HTML written:', outHtml, '| math spans:', math.length);
