// Reading previews only: no notebook code is executed.
import fs from 'node:fs';
import path from 'node:path';
import { marked } from 'file:///C:/Users/thoma/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';

const root = process.cwd();
const destination = path.join(root, 'results/notebook_organization/20261004_subset_full_readability_v1/previews');
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const join = value => Array.isArray(value) ? value.join('') : value ?? '';
const linkTargets = html => html.replace(/href="([^"]+)\.(ipynb|md)(#[^"]*)?"/g, 'href="$1.html$3"');
const style = `body{font:16px/1.6 system-ui,sans-serif;color:#243041;margin:0 auto;padding:28px;max-width:1050px}h1{font-size:30px;line-height:1.25}h2{font-size:24px;margin-top:38px}h3{font-size:19px;margin-top:24px}a{color:#155eac}p,li{max-width:86ch}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}th,td{padding:8px 12px;border:1px solid #d5dce5;text-align:left}img{max-width:100%;height:auto}pre{font:13px/1.45 Consolas,monospace;overflow:auto;background:#f3f5f8;padding:14px}code{font-size:.9em}details{margin:18px 0}.output{margin:14px 0;overflow:auto}.banner{font-size:13px;color:#506075;border-bottom:1px solid #d5dce5;padding-bottom:12px}section{margin-bottom:20px}`;
function save(relative, body) {
  const target = path.join(destination, relative.replace(/\.(ipynb|md)$/, '.html'));
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `<!doctype html><meta charset="utf-8"><title>${escape(relative)}</title><style>${style}</style><div class="banner">Saved notebook reading preview · ${escape(relative)} · code was not executed</div>${linkTargets(body)}`);
}
let count = 0;
for (const directory of ['Subset', 'Full', 'archive']) {
  for (const filename of fs.readdirSync(path.join(root, 'notebooks', directory))) {
    if (!filename.endsWith('.ipynb')) continue;
    const relative = `notebooks/${directory}/${filename}`;
    const notebook = JSON.parse(fs.readFileSync(path.join(root, relative), 'utf8'));
    const cells = notebook.cells.map((cell, index) => {
      if (cell.cell_type === 'markdown') return `<section id="cell-${index}">${marked.parse(join(cell.source))}</section>`;
      let content = `<details><summary>Code cell ${index}${cell.execution_count ? ` · execution ${cell.execution_count}` : ''}</summary><pre>${escape(join(cell.source))}</pre></details>`;
      for (const output of cell.outputs ?? []) {
        const data = output.data ?? {};
        if (data['image/png']) content += `<div class="output"><img alt="Saved figure from cell ${index}" src="data:image/png;base64,${join(data['image/png'])}"></div>`;
        else if (data['text/html']) content += `<div class="output">${join(data['text/html'])}</div>`;
        else {
          const text = join(output.text ?? data['text/plain']);
          if (text) content += `<details class="output"><summary>Saved text output</summary><pre>${escape(text)}</pre></details>`;
        }
      }
      return `<section id="cell-${index}">${content}</section>`;
    });
    save(relative, cells.join('\n')); count++;
  }
}
for (const relative of ['README.md', 'notebooks/README.md', 'notebooks/concepts.md', 'notebooks/archive/README.md', 'docs/image_methodology.md', 'docs/metadata_methodology.md']) {
  save(relative, marked.parse(fs.readFileSync(path.join(root, relative), 'utf8')));
}
console.log(`Rendered ${count} saved notebooks and six guides without execution.`);
