"""Regression checks: python -m unittest discover -s ui/tests -v"""
import http.client
from collections import Counter
import json
import math
import threading
import tempfile
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from ui import server


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            'fruit': {'words': ['apple', 'apple', 'pear'], 'file_count': 1, 'total_words': 3},
            'tech': {'words': ['computer', 'code', 'code', 'code'], 'file_count': 1, 'total_words': 4},
        }
        self.vocab = ['apple', 'pear', 'computer', 'code']

    def test_empty_input_has_consistent_result_shape(self):
        self.assertEqual(server.coursework.classify_document('123 !', self.info, self.vocab), (None,) * 4)

    def test_unknown_words_do_not_create_prediction(self):
        self.assertEqual(server.coursework.classify_document('unseenword', self.info, self.vocab), (None,) * 4)

    def test_unknown_words_do_not_bias_known_words(self):
        a = server.coursework.classify_document('computer', self.info, self.vocab)
        b = server.coursework.classify_document('computer ' + 'unseenword ' * 100, self.info, self.vocab)
        self.assertEqual(a, b)

    def test_repeated_words_preserve_multinomial_scores(self):
        scores, probabilities, prediction, confidence = server.coursework.classify_document('apple apple pear', self.info, self.vocab)
        expected = math.log(.5) + 2 * math.log(3 / 7) + math.log(2 / 7)
        self.assertAlmostEqual(scores['fruit'], expected)
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertEqual(prediction, 'fruit')
        self.assertGreater(confidence, .5)


class QuietHandler(server.Handler):
    def log_message(self, *_args):
        pass


class IndexTests(unittest.TestCase):
    def test_tokenizer_preserves_emails_urls_and_apostrophes(self):
        text = "Mail a.b@example.com. Let's visit https://example.com #nlp @user"
        self.assertEqual(server.tokenize(text), ['Mail', 'a.b@example.com', '.', "Let's", 'visit', 'https://example.com', '#nlp', '@user'])

    def test_punctuation_only_input_is_processed_without_email_rescanning(self):
        self.assertEqual(len(server.tokenize('.' * 200000)), 200000)

    def test_tfidf_uses_requested_corpus_not_global_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            corpus = base / 'corpus'
            corpus.mkdir()
            (corpus / 'only.txt').write_text('zebras giraffes elephants', encoding='utf-8')
            with patch.object(server.coursework, 'ir_path', str(base / 'missing')):
                result = server.coursework.build_tfidf_index(str(corpus), str(base / 'output'))
            self.assertIsNotNone(result)
            self.assertEqual([doc['filename'] for doc in result['indexer'].documents], ['only.txt'])

    def test_model_cache_trains_once_for_concurrent_requests(self):
        server._classifier.cache_clear()
        try:
            with patch.object(server.coursework, 'train_naive_bayes', return_value=({'class': {}}, ['word'])) as train:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _: server.classifier(), range(8)))
                self.assertEqual(train.call_count, 1)
                self.assertTrue(all(result == results[0] for result in results))
        finally:
            server._classifier.cache_clear()

    def test_search_ranks_real_matches_and_handles_no_match(self):
        docs = [('apple.txt', Counter({'apple': 3}), 3), ('code.txt', Counter({'code': 4}), 4)]
        with patch.object(server, 'search_index', return_value=(docs, Counter({'apple': 1, 'code': 1}))):
            found = server.run_analysis({'mode': 'search', 'text': 'apple'})
            empty = server.run_analysis({'mode': 'search', 'text': 'missing'})
        self.assertTrue(found['items'][0].startswith('apple.txt'))
        self.assertEqual(empty['items'], [])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.port = cls.httpd.server_port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def request(self, data=None, raw=None, headers=None, path='/api/run', method='POST'):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=15)
        body = json.dumps(data).encode() if raw is None else raw
        try:
            conn.request(method, path, body=body, headers=headers or {'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_processing_and_stemming(self):
        status, body = self.request({'text': 'The cats are running and running.', 'options': ['lower', 'stop', 'stem']})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data['items'], ['cat', 'run', 'run', '.'])
        self.assertEqual(data['frequencies'][0], ['run', 2])

    def test_invalid_payloads_are_client_errors(self):
        for data in [[], None, 'hello', 1, {}, {'text': 5}, {'text': 'abc', 'mode': 'typo'},
                     {'text': 'abc', 'options': None}, {'text': 'abc', 'options': 'lower'},
                     {'text': 'abc', 'options': ['typo']}, {'text': 'abc', 'options': [{}]}]:
            with self.subTest(data=data):
                status, body = self.request(data)
                self.assertEqual(status, 400)
                self.assertIn('error', json.loads(body))

    def test_malformed_json(self):
        self.assertEqual(self.request(raw=b'{bad')[0], 400)

    def test_content_type_required(self):
        self.assertEqual(self.request({'text': 'hello'}, headers={'Content-Type': 'text/plain'})[0], 415)

    def test_cross_origin_rejected(self):
        self.assertEqual(self.request({'text': 'hello'}, headers={'Content-Type': 'application/json', 'Origin': 'https://example.com'})[0], 403)

    def test_same_origin_allowed(self):
        self.assertEqual(self.request({'text': 'hello'}, headers={'Content-Type': 'application/json', 'Origin': f'http://127.0.0.1:{self.port}'})[0], 200)

    def test_rebinding_host_rejected(self):
        self.assertEqual(self.request({'text': 'hello'}, headers={'Content-Type': 'application/json', 'Host': f'example.com:{self.port}'})[0], 403)

    def test_text_limit(self):
        self.assertEqual(self.request({'text': 'a' * 200001})[0], 413)

    def test_invalid_classification_skips_training(self):
        with patch.object(server, 'classifier', side_effect=AssertionError('must not train')):
            self.assertEqual(self.request({'text': '۱۲۳ !', 'mode': 'classify'})[0], 400)

    def test_internal_files_not_served(self):
        for path in ['/server.py', '/tests/', '/fonts/', '/../Nlp-project.py']:
            with self.subTest(path=path):
                self.assertEqual(self.request(path=path, method='GET')[0], 404)

    def test_static_files_work(self):
        self.assertEqual(self.request(path='/', method='GET')[0], 200)

    def test_parallel_processing_is_independent(self):
        def run(i):
            status, body = self.request({'text': f'word{i}', 'options': []})
            return status == 200 and json.loads(body)['items'] == [f'word{i}']
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(pool.map(run, range(8))))


if __name__ == '__main__':
    unittest.main()
