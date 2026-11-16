"""Render executed notebooks for local reading-flow inspection (nbconvert required)."""
import argparse
import base64
from io import BytesIO
import re
from PIL import Image
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter


def main():
    parser=argparse.ArgumentParser();parser.add_argument('notebooks',nargs='+');parser.add_argument('--out',default='local/presentation-cleanup/rendered')
    args=parser.parse_args();root=Path.cwd().resolve();out=(root/args.out).resolve()
    if not out.is_relative_to(root/'local'):raise ValueError('Rendered reviews must remain under local/')
    out.mkdir(parents=True,exist_ok=True)
    for source in args.notebooks:
        path=(root/source).resolve()
        notebook=nbformat.read(path,as_version=4)
        # Basic exporter retains standard markdown/details and outputs, without browser widgets.
        exporter=HTMLExporter(template_name='basic',exclude_input_prompt=True,exclude_output_prompt=True)
        body,_=exporter.from_notebook_node(notebook)
        css='''body{font-family:var(--font-sans,system-ui);color:var(--foreground);line-height:1.5}
img{max-width:100%;height:auto}pre{overflow-x:auto;font-size:12px}table{border-collapse:collapse;font-size:13px}
th,td{padding:4px 8px;text-align:left;border-bottom:1px solid var(--border,#ddd)}
summary{cursor:pointer}details{margin:12px 0}h2{margin-top:32px}h3{margin-top:24px}'''
        html='<!doctype html><html><head><meta charset="utf-8"><title>Waveform Exploration</title><style>'+css+'</style></head><body>'+body+'</body></html>'
        target=out/(path.stem+'.html');target.write_text(html);print(target)
        def preview_image(match):
            with Image.open(BytesIO(base64.b64decode(match[1]))) as image:
                image.thumbnail((1400,1400))
                buffer=BytesIO();image.save(buffer,format='PNG')
                return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        preview=re.sub(r'data:image/png;base64,([A-Za-z0-9+/=]+)',preview_image,html)
        preview=preview.replace('<body>','<body><p>Reading preview · display images scaled; full-resolution exports unchanged.</p>',1)
        preview_path=out/(path.stem+'_preview.html');preview_path.write_text(preview);print(preview_path)


if __name__=='__main__':main()
