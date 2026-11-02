"""Loopback-only web interface. Run: python ui/server.py."""
from collections import Counter
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
import importlib.util
import json
import logging
import math
import re
import socket
import time

from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 200_000
# JSON escaping may expand each source byte up to six times.
MAX_BODY_BYTES = 6 * MAX_TEXT_BYTES + 10_000
# Scan each candidate once before testing for email. A combined email/punctuation
# regex repeatedly scans the remaining suffix of long dot-only inputs.
CANDIDATE_PATTERN = re.compile(r"https?://\S+|[\w.+@’'#-]+|[^\w\s]")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]+")
WORD_PATTERN = re.compile(r"[#@]?\w+(?:[’'-]\w+)*|[^\w\s]")
STOP = frozenset('a an the is are was were be been being to of in on at for from with and or but it its this that as by not you your we our they their he she i have has had do does did can will would should'.split())
STATIC_FILES = {'/', '/index.html', '/style.css', '/lemon.css', '/app.js',
                '/fonts/Vazirmatn-Regular.ttf', '/fonts/Vazirmatn-Bold.ttf', '/fonts/OFL.txt'}
spec = importlib.util.spec_from_file_location('coursework', ROOT / 'Nlp-project.py')
coursework = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coursework)
_model_lock = Lock()
_index_lock = Lock()


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def _classifier():
    return coursework.train_naive_bayes(str(ROOT / 'Dataset/Classification-Train And Test'))


def classifier():
    # lru_cache alone allows duplicate training on concurrent cache misses.
    with _model_lock:
        return _classifier()


@lru_cache(maxsize=1)
def _search_index():
    files = sorted((ROOT / 'Dataset/IR/IR Documents to Index').glob('*.txt'))[:100]
    if not files:
        raise RequestError('مجموعه‌دادهٔ جست‌وجو موجود نیست.')
    docs, document_frequency = [], Counter()
    for path in files:
        counts = Counter(re.findall(r'\w+', path.read_text(encoding='utf-8', errors='ignore').lower()))
        docs.append((path.name, counts, max(1, sum(counts.values()))))
        document_frequency.update(counts.keys())
    return docs, document_frequency


def search_index():
    with _index_lock:
        return _search_index()


def validate_payload(data):
    if not isinstance(data, dict):
        raise RequestError('بدنهٔ درخواست باید یک شیء JSON باشد.')
    text = data.get('text')
    if not isinstance(text, str) or not text.strip():
        raise RequestError('ابتدا یک متن وارد کنید.')
    try:
        length = len(text.encode('utf-8'))
    except UnicodeEncodeError as exc:
        raise RequestError('کدگذاری متن معتبر نیست.') from exc
    if length > MAX_TEXT_BYTES:
        raise RequestError('حد مجاز متن ۲۰۰ کیلوبایت است.', 413)
    mode = data.get('mode', 'process')
    if not isinstance(mode, str) or mode not in {'process', 'classify', 'search'}:
        raise RequestError('نوع پردازش معتبر نیست.')
    options = data.get('options', [])
    if (not isinstance(options, list) or len(options) > 3 or
            any(not isinstance(option, str) or option not in {'lower', 'stop', 'stem'} for option in options)):
        raise RequestError('گزینه‌های پردازش معتبر نیستند.')
    return text, mode, set(options)


def tokenize(text):
    tokens = []
    for match in CANDIDATE_PATTERN.finditer(text):
        candidate = match.group()
        email = candidate.rstrip('.')
        if candidate.startswith(('http://', 'https://')):
            tokens.append(candidate)
        elif EMAIL_PATTERN.fullmatch(email):
            tokens.append(email)
            tokens.extend('.' * (len(candidate) - len(email)))
        else:
            tokens.extend(WORD_PATTERN.findall(candidate))
    return tokens


def run_analysis(data):
    text, mode, options = validate_payload(data)
    start = time.perf_counter()
    tokens = tokenize(text)
    if mode == 'classify':
        if not coursework.extract_words(text):
            raise RequestError('متن باید شامل واژه‌های انگلیسی با حداقل سه حرف باشد.')
        info, vocab = classifier()
        if not info or not vocab:
            raise RequestError('دادهٔ آموزشی موجود نیست.')
        scores, probabilities, prediction, confidence = coursework.classify_document(text, info, vocab)
        if prediction is None:
            raise RequestError('هیچ واژهٔ شناخته‌شده‌ای از داده‌های آموزشی در متن وجود ندارد.')
        output = sorted(probabilities.items(), key=lambda item: -item[1])
        payload = {'items': [f'{name} · {score:.2%}' for name, score in output],
                   'label': prediction, 'probabilities': probabilities,
                   'note': 'احتمال‌ها برآورد مدل هستند و تضمین درستی پیش‌بینی نیستند.'}
    elif mode == 'search':
        docs, document_frequency = search_index()
        terms = Counter(re.findall(r'\w+', text.lower()))
        ranked = []
        for name, counts, length in docs:
            score = sum(frequency * counts[term] / length *
                        (math.log((1 + len(docs)) / (1 + document_frequency[term])) + 1)
                        for term, frequency in terms.items())
            if score:
                ranked.append((name, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        payload = {'items': [f'{name} · {score:.5f}' for name, score in ranked[:10]],
                   'label': f'{len(ranked)} سند مرتبط از {len(docs)} سند'}
    else:
        values = tokens
        if 'lower' in options:
            values = [token.lower() for token in values]
        if 'stop' in options:
            values = [token for token in values if token.lower() not in STOP]
        if 'stem' in options:
            stemmer = PorterStemmer()
            values = [stemmer.stem(token) for token in values]
        payload = {'items': values, 'frequencies': Counter(values).most_common(6),
                   'label': 'پردازش تکمیل شد'}
    payload.update(tokens=len(tokens), unique=len(set(tokens)), characters=len(text),
                   elapsed=round((time.perf_counter() - start) * 1000))
    return payload


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'ui'), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def check_origin(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:
            raise RequestError('میزبان درخواست مجاز نیست.', 403)
        origin = self.headers.get('Origin')
        if origin is not None and origin not in {f'http://{host}' for host in hosts}:
            raise RequestError('مبدأ درخواست مجاز نیست.', 403)

    def send_json(self, payload, status):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_head(self):
        try:
            self.check_origin()
        except RequestError as exc:
            self.send_error(exc.status)
            return None
        if urlsplit(self.path).path not in STATIC_FILES:
            self.send_error(404)
            return None
        return super().send_head()

    def do_POST(self):
        try:
            self.check_origin()
            if self.path != '/api/run':
                raise RequestError('مسیر درخواست موجود نیست.', 404)
            if self.headers.get_content_type() != 'application/json':
                raise RequestError('نوع محتوا باید application/json باشد.', 415)
            if self.headers.get('Transfer-Encoding'):
                raise RequestError('ارسال قطعه‌ای درخواست پشتیبانی نمی‌شود.')
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise RequestError('طول درخواست معتبر نیست.')
            if len(lengths[0]) > 10:
                raise RequestError('حجم درخواست بیش از حد مجاز است.', 413)
            size = int(lengths[0])
            if size > MAX_BODY_BYTES:
                raise RequestError('حجم درخواست بیش از حد مجاز است.', 413)
            if size <= 0:
                raise RequestError('بدنهٔ درخواست خالی است.')
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise RequestError('دریافت درخواست کامل نشد.')
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise RequestError('ساختار JSON معتبر نیست.') from exc
            payload = run_analysis(data)
            status = 200
        except RequestError as exc:
            payload, status = {'error': str(exc)}, exc.status
        except (TimeoutError, socket.timeout):
            payload, status = {'error': 'مهلت دریافت درخواست به پایان رسید.'}, 408
        except Exception:
            logging.exception('NLP request failed')
            payload, status = {'error': 'اجرای پردازش ناموفق بود؛ گزارش سرور را بررسی کنید.'}, 500
        try:
            self.send_json(payload, status)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # A cancelled browser request must not become an application error.
            pass


def main():
    httpd = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
    print('NLP Studio: http://127.0.0.1:8765', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()
