import fs from 'node:fs/promises';
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';

const filePath = '/Users/neo/Downloads/RepulsiveCurves.pdf';
const data = new Uint8Array(await fs.readFile(filePath));

const loadingTask = pdfjs.getDocument({ data, useWorkerFetch: false, isEvalSupported: false, disableFontFace: true });
const pdf = await loadingTask.promise;

let out = '';
for (let p = 1; p <= pdf.numPages; p += 1) {
  const page = await pdf.getPage(p);
  const content = await page.getTextContent();
  const items = content.items.map((it) => ('str' in it ? it.str : '')).filter(Boolean);
  out += `\n\n=== PAGE ${p} ===\n`;
  out += items.join(' ');
}

console.log(out);
