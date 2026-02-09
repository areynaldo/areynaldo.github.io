---
title: "Bit Cells"
date: 2026-01-31
summary: "A simple calculator for bitwise operations on grids—built for puzzle solvers."
description: "A simple calculator for bitwise operations on grids—built for puzzle solvers."
---

For the past few months, my friends and I have been solving [ThinkyGames' Thinky Dailies](https://thinkygames.com/dailies/). Many of these puzzles involve shading cells on a grid, and one particularly tricky puzzle had us performing bitwise operations across multiple grids. We ended up reaching for image editors just to XOR two patterns together (Spoiler: the solution had nothing to do with this).

That felt like overkill for such a simple task. So I built this: a dumb grid calculator that runs entirely in the browser. Create grids, name them, draw patterns, and combine them with expressions like `(A ^ B) & Mask`. The result updates live as you edit.

<style>
.grid-div { display: inline-block; margin: 1em; vertical-align: top; }
.grid { display: inline-grid; gap: 1px; background: black; border: 1px solid black; }
.grids { display: flex; flex-wrap: wrap; }
.calculator-panels { display: flex; gap: 2em; align-items: flex-start; }
.calculator-grids { flex: 1; min-width: 0; }
.calculator-output { flex-shrink: 0; }
.cell { width: 1em; height: 1em; background: white; cursor: pointer; }
.cell.active { background: black; }
@media (max-width: 800px) { .calculator-panels { flex-direction: column; } }
</style>

<div class="calculator-layout">
  <h4>Controls</h4>
  <div class="controls">
    <label>Size: <input type="number" id="sizeSlider" min="1" max="50" value="10" /></label>
    <label><input type="button" id="addGrid" value="Add Grid" /></label>
  </div>
  <div class="calculator-panels">
    <div class="calculator-grids">
      <h4>Grids</h4>
      <div class="grids" id="gridsDiv"></div>
    </div>
    <div class="calculator-output">
      <h4>Output</h4>
      <div class="grid-div">
        <div class="grid" id="outputGrid"></div>
        <div style="margin-top: 0.5em;">
          <input type="text" id="expressionInput" placeholder="e.g. A & B, ~A, rotL(A)" style="width: 12em;" />
          <button id="saveOutputBtn">Save as new</button>
        </div>
        <div id="expressionError" style="color: red; font-size: 0.8em;"></div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const gridsDiv = $("gridsDiv"), sizeSlider = $("sizeSlider"), addGrid = $("addGrid");
const outputGrid = $("outputGrid"), expressionInput = $("expressionInput");
const expressionError = $("expressionError"), saveOutputBtn = $("saveOutputBtn");
let currentLetter = "A", gridsMap = {}, lastResult = null;

const nextChar = c => String.fromCharCode(c.charCodeAt(0) + 1);
const mapGrid = (a, fn) => a.map((row, y) => row.map((cell, x) => fn(cell, x, y)));
const mapGrids = (a, b, fn) => a.map((row, y) => row.map((cell, x) => fn(cell, b[y][x])));
const gridAnd = (a, b) => mapGrids(a, b, (x, y) => x && y);
const gridOr = (a, b) => mapGrids(a, b, (x, y) => x || y);
const gridXor = (a, b) => mapGrids(a, b, (x, y) => x !== y);
const gridNot = a => mapGrid(a, x => !x);

function rotL(a) {
  const cols = a[0]?.length || 0, rows = a.length;
  return Array.from({length: cols}, (_, i) => Array.from({length: rows}, (_, j) => a[j][cols - 1 - i]));
}
function rotR(a) {
  const cols = a[0]?.length || 0, rows = a.length;
  return Array.from({length: cols}, (_, i) => Array.from({length: rows}, (_, j) => a[rows - 1 - j][i]));
}

function evaluateExpression(expr) {
  if (!expr.trim()) return null;
  const getGrid = name => {
    if (gridsMap[name]) return structuredClone(gridsMap[name]);
    throw new Error(`Unknown grid: ${name} (available: ${Object.keys(gridsMap).join(', ') || 'none'})`);
  };
  let tokens = [], i = 0;
  const ops = {'&':'AND', '|':'OR', '^':'XOR', '~':'NOT', '(':'LPAREN', ')':'RPAREN', ',':'COMMA'};
  while (i < expr.length) {
    if (expr[i] === ' ') { i++; continue; }
    if (ops[expr[i]]) { tokens.push({type: ops[expr[i]]}); i++; continue; }
    let start = i;
    while (i < expr.length && /[a-zA-Z0-9_]/.test(expr[i])) i++;
    if (i > start) tokens.push({type: 'ID', value: expr.slice(start, i)});
    else throw new Error('Unexpected character: ' + expr[i]);
  }
  let pos = 0;
  const peek = () => tokens[pos], consume = () => tokens[pos++];
  const parseOr = () => { let left = parseXor(); while (peek()?.type === 'OR') { consume(); left = gridOr(left, parseXor()); } return left; };
  const parseXor = () => { let left = parseAnd(); while (peek()?.type === 'XOR') { consume(); left = gridXor(left, parseAnd()); } return left; };
  const parseAnd = () => { let left = parseUnary(); while (peek()?.type === 'AND') { consume(); left = gridAnd(left, parseUnary()); } return left; };
  const parseUnary = () => { if (peek()?.type === 'NOT') { consume(); return gridNot(parseUnary()); } return parsePrimary(); };
  const parsePrimary = () => {
    let t = peek();
    if (!t) throw new Error('Unexpected end');
    if (t.type === 'LPAREN') { consume(); let val = parseOr(); if (peek()?.type !== 'RPAREN') throw new Error('Missing )'); consume(); return val; }
    if (t.type === 'ID') {
      consume();
      if (peek()?.type === 'LPAREN') { consume(); let arg = parseOr(); if (peek()?.type !== 'RPAREN') throw new Error('Missing )'); consume(); if (t.value === 'rotL') return rotL(arg); if (t.value === 'rotR') return rotR(arg); throw new Error('Unknown function: ' + t.value); }
      return getGrid(t.value);
    }
    throw new Error('Unexpected token');
  };
  return parseOr();
}

function renderGrid(container, data, size, onClick) {
  container.innerHTML = '';
  container.style.gridTemplateColumns = `repeat(${size}, 1em)`;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell' + (data?.[y]?.[x] ? ' active' : '');
      if (onClick) cell.onclick = () => onClick(x, y);
      container.appendChild(cell);
    }
  }
}

function renderOutputGrid() {
  const expr = expressionInput.value, size = Number(sizeSlider.value);
  if (!expr.trim()) { lastResult = null; expressionError.textContent = ''; renderGrid(outputGrid, null, size); return; }
  try { lastResult = evaluateExpression(expr); if (!lastResult) return; expressionError.textContent = ''; renderGrid(outputGrid, lastResult, lastResult[0]?.length || size); }
  catch (e) { expressionError.textContent = e.message; }
}

function setSize(size) {
  for (let name in gridsMap) {
    let g = gridsMap[name];
    while (g.length < size) g.push(Array(size).fill(false));
    g.length = size;
    g.forEach(row => { while (row.length < size) row.push(false); row.length = size; });
  }
}

function createGrid(size) {
  gridsMap[currentLetter] = Array.from({length: size}, () => Array(size).fill(false));
  currentLetter = nextChar(currentLetter);
  renderGrids();
}

function renderGrids() {
  gridsDiv.innerHTML = "";
  const size = Number(sizeSlider.value);
  for (let gridName in gridsMap) {
    let gridValues = gridsMap[gridName];
    let div = document.createElement("div");
    div.className = "grid-div";
    let nameInput = Object.assign(document.createElement("input"), { type: "text", value: gridName, style: "width:3em;display:inline;" });
    nameInput.onchange = () => { let n = nameInput.value.trim(); if (n && n !== gridName && !gridsMap[n]) { gridsMap[n] = gridsMap[gridName]; delete gridsMap[gridName]; renderGrids(); } else nameInput.value = gridName; };
    const btn = (l, fn) => { let b = document.createElement("button"); b.textContent = l; b.onclick = fn; return b; };
    let topRow = document.createElement("div");
    topRow.style = "margin-bottom:0.5em;";
    topRow.appendChild(nameInput);
    topRow.appendChild(btn("clone", () => { gridsMap[currentLetter] = structuredClone(gridValues); currentLetter = nextChar(currentLetter); renderGrids(); }));
    topRow.appendChild(btn("×", () => { delete gridsMap[gridName]; renderGrids(); }));
    div.appendChild(topRow);
    let grid = document.createElement("div");
    grid.className = "grid";
    renderGrid(grid, gridValues, size, (x, y) => { gridsMap[gridName][y][x] = !gridsMap[gridName][y][x]; renderGrids(); });
    div.appendChild(grid);
    let bottomRow = document.createElement("div");
    bottomRow.appendChild(btn("↺", () => { gridsMap[gridName] = rotL(gridValues); renderGrids(); }));
    bottomRow.appendChild(btn("↻", () => { gridsMap[gridName] = rotR(gridValues); renderGrids(); }));
    div.appendChild(bottomRow);
    gridsDiv.appendChild(div);
  }
  renderOutputGrid();
}

expressionInput.oninput = renderOutputGrid;
saveOutputBtn.onclick = () => { if (lastResult) { gridsMap[currentLetter] = structuredClone(lastResult); currentLetter = nextChar(currentLetter); renderGrids(); } };
sizeSlider.onchange = () => { setSize(Number(sizeSlider.value)); renderGrids(); };
addGrid.onclick = () => createGrid(Number(sizeSlider.value));
renderOutputGrid();
</script>