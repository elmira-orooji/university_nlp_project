const $ = selector => document.querySelector(selector);
const fa = number => new Intl.NumberFormat('fa-IR').format(number);
const initial = $('#text').value;
const empty = $('.empty').cloneNode(true);
const MAX_TEXT_BYTES = 200_000;
const MAX_VISIBLE_ITEMS = 1000;
const REQUEST_TIMEOUT_MS = 60_000;
let mode = 'process', view = 'tokens', result = null, busy = false;
const labels = {
  process: ['هر متن، یک کشف تازه', 'متن را وارد کنید، مراحل پردازش را انتخاب کنید و نتیجه را ببینید.', 'پردازش متن', 'اجرای پردازش'],
  classify: ['متن به کدام دسته تعلق دارد؟', 'موضوع متن را با مدل Naive Bayes و داده‌های آموزشی پروژه بررسی کنید.', 'طبقه‌بندی متن', 'طبقه‌بندی متن'],
  search: ['در میان اسناد، پاسخ را پیدا کنید', 'عبارت انگلیسی را جست‌وجو کنید و اسناد مرتبط را بر اساس TF-IDF ببینید.', 'بازیابی اطلاعات', 'جست‌وجوی اسناد']
};
function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $('#toast').hidden = true; }, 3000);
}
function count() { $('#char-count').textContent = fa(Array.from($('#text').value).length) + ' نویسه'; }
function reset() {
  result = null;
  $('#result').replaceChildren(empty.cloneNode(true));
  $('#result-caption').textContent = 'خروجی واقعی، بر اساس متن ورودی';
  $('#status').textContent = 'در انتظار اجرا';
  $('#status').className = 'status';
  $('#copy').disabled = $('#download').disabled = true;
  for (const key of ['tokens', 'unique', 'characters', 'elapsed']) $('#metric-' + key).textContent = '—';
}
function setWorking(working, message = '') {
  busy = working;
  document.querySelectorAll('#clear,#sample,#upload,#file,#run,.option input,[data-mode]').forEach(el => { el.disabled = working; });
  $('#text').readOnly = working;
  $('#run').textContent = working ? message : labels[mode][3] + ' ←';
  $('.editor-panel').setAttribute('aria-busy', String(working));
  $('.results-panel').setAttribute('aria-busy', String(working));
}
function validResult(data) {
  return data && Array.isArray(data.items) && data.items.every(item => typeof item === 'string') &&
    typeof data.label === 'string' && ['tokens', 'unique', 'characters', 'elapsed'].every(key =>
      Number.isFinite(data[key]) && data[key] >= 0) &&
    (data.frequencies === undefined || (Array.isArray(data.frequencies) && data.frequencies.every(row =>
      Array.isArray(row) && typeof row[0] === 'string' && Number.isFinite(row[1]) && row[1] > 0)));
}
$('#text').addEventListener('input', () => { count(); reset(); });
count();
$('#clear').onclick = () => { if (busy) return; $('#text').value = ''; count(); reset(); $('#text').focus(); };
$('#sample').onclick = () => { if (busy) return; $('#text').value = mode === 'search' ? 'language science' : initial; count(); reset(); };
document.querySelectorAll('[data-mode]').forEach(button => {
  button.onclick = () => {
    if (busy) return;
    mode = button.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b === button));
    const label = labels[mode];
    $('#title').textContent = label[0];
    $('#subtitle').textContent = label[1];
    $('#breadcrumb').textContent = label[2];
    $('#run').textContent = label[3] + ' ←';
    $('.options').hidden = mode !== 'process';
    $('.settings-heading > span').hidden = mode !== 'process';
    $('#mode-note').textContent = mode === 'process' ? 'توکن‌سازی در تمام اجراها انجام می‌شود.' :
      mode === 'classify' ? 'مدل اصلی پروژه؛ آموزش روی حداکثر ۱۰۰ فایل از هر دسته.' :
        'پیش‌نمایش جست‌وجو روی حداکثر ۱۰۰ سند؛ رتبه‌بندی با TF-IDF.';
    $('.settings-heading h3').textContent = mode === 'process' ? 'مراحل پردازش' : 'روش اجرا';
    // Frequency belongs to text processing; do not carry it into other tools.
    view = 'tokens';
    document.querySelectorAll('[data-view]').forEach(b => {
      b.classList.toggle('selected', b.dataset.view === view);
      b.disabled = b.dataset.view === 'frequency' && mode !== 'process';
    });
    reset();
  };
});
$('#upload').onclick = () => { if (!busy) $('#file').click(); };
$('#text-tab').onclick = () => $('#text').focus();
$('#file').onchange = async () => {
  if (busy) return;
  const file = $('#file').files[0];
  if (!file) return;
  if (file.size > MAX_TEXT_BYTES) { toast('حجم فایل باید حداکثر ۲۰۰ کیلوبایت باشد.'); $('#file').value = ''; return; }
  setWorking(true, 'خواندن فایل…');
  try {
    const text = await file.text();
    if (new TextEncoder().encode(text).length > MAX_TEXT_BYTES) throw Error('حجم متن باید حداکثر ۲۰۰ کیلوبایت باشد.');
    $('#text').value = text;
    count();
    reset();
    toast('فایل بارگذاری شد.');
  } catch (error) {
    toast(error.message || 'خواندن فایل ناموفق بود.');
  } finally {
    $('#file').value = '';
    setWorking(false);
  }
};
function render() {
  if (!result) return;
  const container = $('#result');
  container.replaceChildren();
  if (view === 'json') {
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(result, null, 2);
    container.append(pre);
    return;
  }
  if (view === 'frequency') {
    if (!result.frequencies) { container.textContent = 'فراوانی فقط در بخش پردازش متن نمایش داده می‌شود.'; return; }
    const max = result.frequencies[0]?.[1] || 1;
    result.frequencies.forEach(([word, n]) => {
      const row = document.createElement('div');
      row.className = 'frequency-row';
      const label = document.createElement('span');
      label.textContent = word;
      const bar = document.createElement('div');
      bar.className = 'bar';
      const fill = document.createElement('i');
      fill.style.width = n / max * 100 + '%';
      bar.append(fill);
      const value = document.createElement('b');
      value.textContent = fa(n);
      row.append(label, bar, value);
      container.append(row);
    });
    return;
  }
  const wrap = document.createElement('div');
  wrap.className = mode === 'process' ? 'tokens' : 'rank-list';
  if (!result.items.length) wrap.textContent = 'نتیجه‌ای پیدا نشد.';
  result.items.slice(0, MAX_VISIBLE_ITEMS).forEach(item => {
    const token = document.createElement(mode === 'process' ? 'span' : 'div');
    token.className = mode === 'process' ? 'token' : 'rank-item';
    token.textContent = item;
    wrap.append(token);
  });
  container.append(wrap);
  if (result.items.length > MAX_VISIBLE_ITEMS) {
    const note = document.createElement('p');
    note.className = 'mode-note';
    note.textContent = 'فقط ' + fa(MAX_VISIBLE_ITEMS) + ' مورد نخست نمایش داده شده است. خروجی کامل در فایل JSON موجود است.';
    container.append(note);
  }
}
document.querySelectorAll('[data-view]').forEach(button => {
  button.onclick = () => {
    view = button.dataset.view;
    document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('selected', b === button));
    render();
  };
});
$('#run').onclick = async () => {
  if (busy) return;
  const text = $('#text').value;
  if (!text.trim()) { toast('ابتدا یک متن وارد کنید.'); $('#text').focus(); return; }
  if (new TextEncoder().encode(text).length > MAX_TEXT_BYTES) { toast('حجم متن باید حداکثر ۲۰۰ کیلوبایت باشد.'); return; }
  const payload = { mode, text, options: [...document.querySelectorAll('.option input:checked')].map(i => i.value) };
  reset();
  setWorking(true, 'در حال اجرا…');
  $('#status').textContent = 'در حال اجرا…';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch('/api/run', { method: 'POST', signal: controller.signal,
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    let data;
    try { data = await response.json(); } catch { throw Error('پاسخ سرور معتبر نیست.'); }
    if (!response.ok) throw Error(typeof data?.error === 'string' ? data.error : 'پردازش انجام نشد.');
    if (!validResult(data)) throw Error('ساختار خروجی سرور معتبر نیست.');
    result = data;
    render();
    $('#status').textContent = 'تکمیل شد';
    $('#status').className = 'status done';
    $('#result-caption').textContent = data.label;
    for (const key of ['tokens', 'unique', 'characters', 'elapsed']) $('#metric-' + key).textContent = fa(data[key]);
    $('#copy').disabled = $('#download').disabled = false;
  } catch (error) {
    reset();
    $('#result').textContent = controller.signal.aborted ? 'زمان انتظار به پایان رسید. دوباره تلاش کنید.' :
      error instanceof TypeError ? 'ارتباط با سرور برقرار نشد. اجرای سرور محلی را بررسی کنید.' : error.message;
    $('#status').textContent = 'خطا در اجرا';
  } finally {
    clearTimeout(timeout);
    setWorking(false);
  }
};
$('#copy').onclick = async () => {
  if (!result) return;
  try { await navigator.clipboard.writeText(JSON.stringify(result, null, 2)); toast('خروجی کپی شد.'); }
  catch { toast('کپی ممکن نیست؛ خروجی را دریافت کنید.'); }
};
$('#download').onclick = () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'nlp-' + mode + '.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('#guide').onclick = () => $('#help').showModal();
$('#close-help').onclick = () => $('#help').close();

document.querySelectorAll('.option input').forEach(input => input.addEventListener('change', reset));

// The interface follows Lemon's neutral glass language; all icons are local SVGs.
const iconPaths={
  process:['M4 5h16M4 12h10M4 19h13','M17 9l3 3-3 3'],
  classify:['M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z'],
  search:['M20 20l-5-5','M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0'],
  output:['M9 4H5v16h14V4h-4','M9 2h6v5H9zM8 12h8M8 16h5'],
  text:['M4 5h16M12 5v15M8 20h8'],
  spark:['M12 3l2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6z'],
  clock:['M12 8v5l3 2','M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0'],
  count:['M5 5h14M5 12h14M5 19h14'],
  unique:['M12 3l9 9-9 9-9-9z','M12 8l4 4-4 4-4-4z'],
  moon:['M20 15a8 8 0 0 1-11-11 8 8 0 1 0 11 11'],
  sun:['M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5','M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0']
};
function setIcon(element,name){if(!element)return;const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('class','ui-icon');svg.setAttribute('aria-hidden','true');for(const d of iconPaths[name]){const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',d);svg.append(path)}element.replaceChildren(svg)}
document.querySelectorAll('[data-mode]').forEach(button=>setIcon(button.querySelector('span'),button.dataset.mode));
setIcon($('.workspace-icon'),'classify');setIcon($('.note-symbol'),'spark');
setIcon($('.editor-panel .mini-icon'),'text');setIcon($('.results-panel .mini-icon'),'output');
document.querySelectorAll('.metric-icon').forEach((el,i)=>setIcon(el,['count','unique','text','clock'][i]));
const themeButton=$('#theme-toggle');
function applyTheme(theme){document.documentElement.dataset.theme=theme;const dark=theme==='dark';themeButton.setAttribute('aria-pressed',String(dark));themeButton.setAttribute('aria-label',dark?'فعال‌کردن حالت روشن':'فعال‌کردن حالت تیره');setIcon(themeButton,dark?'sun':'moon')}
let savedTheme;try{savedTheme=localStorage.getItem('vajeh-theme')}catch{}
applyTheme(savedTheme==='dark'||savedTheme==='light'?savedTheme:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
themeButton.onclick=()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';applyTheme(theme);try{localStorage.setItem('vajeh-theme',theme)}catch{}};

function syncSelections(){document.querySelectorAll("[data-mode]").forEach(el=>el.setAttribute("aria-pressed",String(el.dataset.mode===mode)));document.querySelectorAll("[data-view]").forEach(el=>el.setAttribute("aria-pressed",String(el.dataset.view===view)))}
syncSelections();
document.querySelectorAll("[data-mode],[data-view]").forEach(el=>el.addEventListener("click",syncSelections));
