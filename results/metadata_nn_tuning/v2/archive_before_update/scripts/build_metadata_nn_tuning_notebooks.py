"""Build four report-only notebooks without starting or modifying experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_nn_tuning_reporting import ARCHIVE, NAMES, build_notebook


def render_html(notebook, title):
    """Render saved outputs with bundled Markdown tooling; execute no cells."""
    try:
        from nbconvert import HTMLExporter
    except ImportError:
        runtime = Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
        node = runtime/"bin/node.exe"
        marked = runtime/"node_modules/marked/lib/marked.esm.js"
        if not node.is_file() or not marked.is_file():
            raise RuntimeError("HTML export needs nbconvert or the bundled Node/marked runtime")
        script = r'''
const {marked} = await import(process.argv[1]);
let input = ''; for await (const chunk of process.stdin) input += chunk;
const {notebook, title} = JSON.parse(input);
const join = x => Array.isArray(x) ? x.join('') : x ?? '';
const esc = x => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const sections = notebook.cells.map((cell,index) => {
  if(cell.cell_type === 'markdown') return `<section id="cell-${index}">${marked.parse(join(cell.source))}</section>`;
  let html = `<details><summary>Code cell ${index+1}${cell.execution_count ? ' · execution '+cell.execution_count : ''}</summary><pre>${esc(join(cell.source))}</pre></details>`;
  for(const output of cell.outputs ?? []) {
    const data = output.data ?? {};
    if(data['image/png']) html += `<div class="output"><img alt="Saved chart from cell ${index+1}" src="data:image/png;base64,${join(data['image/png'])}"></div>`;
    else if(data['text/html']) html += `<div class="output">${join(data['text/html'])}</div>`;
    else if(data['text/markdown']) html += `<div class="output">${marked.parse(join(data['text/markdown']))}</div>`;
    else if(output.output_type === 'error') html += `<pre class="error">${esc(join(output.traceback))}</pre>`;
    else if(output.text ?? data['text/plain']) html += `<pre class="output">${esc(join(output.text ?? data['text/plain']))}</pre>`;
  }
  return `<section id="cell-${index}">${html}</section>`;
});
const style = 'body{font:16px/1.6 system-ui,sans-serif;color:#243041;margin:0 auto;padding:28px;max-width:1050px}h1{font-size:30px;line-height:1.25}h2{font-size:24px;margin-top:38px}h3{font-size:19px;margin-top:24px}a{color:#155eac}p,li{max-width:86ch}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}th,td{padding:8px 12px;border:1px solid #d5dce5;text-align:left}img{max-width:100%;height:auto}pre{font:13px/1.45 Consolas,monospace;overflow:auto;background:#f3f5f8;padding:14px}code{font-size:.9em}details{margin:18px 0}.output{margin:14px 0;overflow:auto}.error{color:#9f1239}section{margin-bottom:20px}';
process.stdout.write(`<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)}</title><style>${style}</style></head><body><p>Saved notebook reading preview · ${esc(title)} · rendering executes no cells</p>${sections.join('\n')}</body></html>`);
'''
        return subprocess.run([str(node),"--input-type=module","-e",script,marked.as_uri()],
                              input=json.dumps({"notebook":notebook,"title":title}),text=True,
                              encoding="utf-8",check=True,capture_output=True).stdout
    else:
        body,_ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--run-id",default=None,help="Bind both conditions to this exact versioned run; default reads latest receipt")
    parser.add_argument("--condition",choices=("original","deduplicated","both"),default="both")
    parser.add_argument("--html",action="store_true",help="Render saved notebooks to new report-preview files without execution")
    parser.add_argument("--html-only",action="store_true",help="Render existing saved outputs without rebuilding notebooks")
    args = parser.parse_args()
    root = args.root.resolve()
    if not all((root/ARCHIVE/name).is_file() for name in NAMES.values()):
        raise RuntimeError("The original notebook archive must be complete before replacement")
    conditions = ("original","deduplicated") if args.condition == "both" else (args.condition,)
    for condition in conditions:
        directory = root/"notebooks"/("Subset" if condition == "original" else "Subset_dedup")
        for stage,name in NAMES.items():
            path = directory/name
            notebook = (nbformat.read(path,as_version=4) if args.html_only else
                        build_notebook(root,condition,stage,args.run_id))
            if not args.html_only:
                nbformat.write(notebook,path)
            print(path.relative_to(root))
            if args.html or args.html_only:
                html = render_html(notebook,f"{condition}: {path.name}")
                out = root/"results/metadata_nn_tuning/v1/report_previews"/condition/path.with_suffix(".html").name
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_text(html,encoding="utf-8")
                print(out.relative_to(root))


if __name__ == "__main__":
    main()
