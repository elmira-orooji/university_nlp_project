const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Minimal DOM adapter: tests exercise asynchronous app behavior, not layout.
function setup(fetchImplementation) {
  class Element {
    constructor() {
      this.value = ''; this.textContent = ''; this.children = []; this.dataset = {};
      this.style = {}; this.disabled = false; this.hidden = false; this.attributes = {};
      this.classList = { toggle() {} };
    }
    addEventListener() {}
    setAttribute(key, value) { this.attributes[key] = value; }
    cloneNode() { return new Element(); }
    replaceChildren(...nodes) { this.children = nodes; }
    append(...nodes) { this.children.push(...nodes); }
    querySelector() { return new Element(); }
    focus() {}
  }
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, new Element());
    return nodes.get(selector);
  };
  const modes = ['process', 'classify', 'search'].map(mode => Object.assign(new Element(), { dataset: { mode } }));
  const views = ['tokens', 'frequency', 'json'].map(view => Object.assign(new Element(), { dataset: { view } }));
  const options = ['lower', 'stop', 'stem'].map(value => Object.assign(new Element(), { value }));
  const timers = new Map();
  const document = {
    documentElement: new Element(), querySelector: node,
    querySelectorAll(selector) {
      if (selector === '[data-mode]') return modes;
      if (selector === '[data-view]') return views;
      if (selector === '[data-mode],[data-view]') return [...modes, ...views];
      if (selector === '.option input:checked') return options.slice(0, 2);
      if (selector === '.option input') return options;
      if (selector === '.metric-icon') return Array.from({ length: 4 }, () => new Element());
      return [node('#clear'), node('#sample'), node('#upload'), node('#file'), node('#run'), ...options, ...modes];
    },
    createElement: () => new Element(), createElementNS: () => new Element()
  };
  const context = vm.createContext({ document, console, Intl, TextEncoder, AbortController,
    TypeError, Error, Blob, URL, navigator: {}, matchMedia: () => ({ matches: false }),
    localStorage: { getItem() { return null; }, setItem() {} },
    setTimeout(callback, delay) { const id = {}; timers.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timers.delete(id); }, fetch: fetchImplementation || (() => { throw Error('unexpected fetch'); })
  });
  node('#text').value = 'Sample text';
  vm.runInContext(readFileSync(path.join(__dirname, '../app.js'), 'utf8'), context);
  return { node, modes, timers, state: expression => vm.runInContext(expression, context) };
}
const response = data => ({ ok: true, json: async () => data });
const result = { items: ['sample'], label: 'done', tokens: 1, unique: 1, characters: 6, elapsed: 1 };

test('file read locks execution until input is ready', async () => {
  let resolveRead, calls = 0;
  const app = setup(() => { calls++; return Promise.resolve(response(result)); });
  app.node('#file').files = [{ size: 10, text: () => new Promise(resolve => { resolveRead = resolve; }) }];
  const loading = app.node('#file').onchange();
  assert.equal(app.node('#run').disabled, true);
  assert.equal(app.node('#text').readOnly, true);
  await app.node('#run').onclick();
  assert.equal(calls, 0);
  resolveRead('Loaded text');
  await loading;
  assert.equal(app.node('#text').value, 'Loaded text');
  assert.equal(app.node('#run').disabled, false);
});

test('failed file read preserves input and releases controls', async () => {
  const app = setup();
  app.node('#file').files = [{ size: 10, text: async () => { throw Error('Cannot read file'); } }];
  await app.node('#file').onchange();
  assert.equal(app.node('#text').value, 'Sample text');
  assert.equal(app.node('#run').disabled, false);
  assert.equal(app.node('#toast').textContent, 'Cannot read file');
});

test('new request clears previous result and failed requests clear metrics', async () => {
  let calls = 0, rejectRequest;
  const app = setup(() => ++calls === 1 ? Promise.resolve(response(result)) :
    new Promise((_resolve, reject) => { rejectRequest = reject; }));
  await app.node('#run').onclick();
  assert.equal(app.state('result.items[0]'), 'sample');
  const pending = app.node('#run').onclick();
  assert.equal(app.state('result'), null);
  assert.equal(app.node('#metric-tokens').textContent, '—');
  assert.equal(app.modes[1].disabled, true);
  rejectRequest(new TypeError('Failed to fetch'));
  await pending;
  assert.equal(app.node('#status').textContent, 'خطا در اجرا');
  assert.equal(app.node('#metric-tokens').textContent, '—');
  assert.equal(app.node('#download').disabled, true);
  assert.equal(app.node('#run').disabled, false);
});

test('request deadline unlocks interface and reports timeout', async () => {
  const app = setup((_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(Error('aborted')));
  }));
  const pending = app.node('#run').onclick();
  const timer = [...app.timers.values()].find(value => value.delay === 60000);
  assert.ok(timer);
  timer.callback();
  await pending;
  assert.match(app.node('#result').textContent, /زمان انتظار/);
  assert.equal(app.node('#run').disabled, false);
  assert.equal(app.state('result'), null);
});

test('malformed success response is not treated as completed processing', async () => {
  const app = setup(async () => response({ items: 'wrong' }));
  await app.node('#run').onclick();
  assert.equal(app.state('result'), null);
  assert.equal(app.node('#status').textContent, 'خطا در اجرا');
  assert.equal(app.node('#download').disabled, true);
});

test('large output limits DOM nodes but preserves full export data', async () => {
  const items = Array.from({ length: 2000 }, (_, i) => String(i));
  const app = setup(async () => response({ ...result, items }));
  await app.node('#run').onclick();
  assert.equal(app.node('#result').children[0].children.length, 1000);
  assert.equal(app.state('result.items.length'), 2000);
});

test('oversized UTF-8 input is rejected before network request', async () => {
  let calls = 0;
  const app = setup(() => { calls++; });
  app.node('#text').value = 'آ'.repeat(100001);
  await app.node('#run').onclick();
  assert.equal(calls, 0);
  assert.match(app.node('#toast').textContent, /۲۰۰/);
});
