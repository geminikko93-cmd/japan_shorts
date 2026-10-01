import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import anthropic
import httpx2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shorts import director
from shorts.common import PipelineError


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'llm': {'base_url': 'https://codex.hungnguyen.codes',
                            'model': 'claude-opus-5-5', 'n_titles': 5, 'n_captions': 30}}
        self.env = patch.dict(os.environ, {'ANTHROPIC_AUTH_TOKEN': 'test-relay-token',
                                           'ANTHROPIC_API_KEY': 'test-official-key',
                                           'APPDATA': str(ROOT / 'work/test-appdata')}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.requests = []

    def call(self, content='{"ko":["번역"]}', status=200, stop='end_turn', mode='prompt', invoke=None):
        self.cfg['llm']['json_mode'] = mode
        def handler(request):
            self.requests.append(request)
            payload = {'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                       'model': 'claude-opus-5-5', 'stop_reason': stop, 'stop_sequence': None,
                       'content': [{'type': 'text', 'text': content}],
                       'usage': {'input_tokens': 10, 'output_tokens': 10}}
            if status != 200:
                payload = {'type': 'error', 'error': {'type': 'authentication_error',
                                                     'message': 'secret-error-detail'}}
            return httpx2.Response(status, json=payload)
        cls = anthropic.Anthropic
        def factory(**kwargs):
            return cls(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        with patch.object(director.anthropic, 'Anthropic', side_effect=factory):
            if invoke:
                return invoke()
            return director._ask_json('system', 'user', director.TRANSLATE_SCHEMA, self.cfg, '번역')[0]

    def test_wire_auth_url_model_no_beta(self):
        self.assertEqual(self.call(), {'ko': ['번역']})
        r = self.requests[0]
        self.assertEqual(str(r.url), 'https://codex.hungnguyen.codes/v1/messages')
        self.assertEqual(r.headers['authorization'], 'Bearer test-relay-token')
        self.assertNotIn('x-api-key', r.headers)
        self.assertNotIn('anthropic-beta', r.headers)
        body = json.loads(r.content)
        self.assertEqual(body['model'], 'claude-opus-5-5')
        self.assertNotIn('fallbacks', body)
        self.assertNotIn('output_config', body)
        self.assertIn('JSON Schema', body['system'])

    def test_schema_mode(self):
        self.call(mode='schema')
        self.assertEqual(json.loads(self.requests[0].content)['output_config']['format']['schema'],
                         director.TRANSLATE_SCHEMA)

    def test_environment_override(self):
        os.environ['ANTHROPIC_MODEL'] = 'gateway-opus'
        os.environ['ANTHROPIC_BASE_URL'] = 'https://example.test/anthropic/'
        self.call()
        self.assertEqual(str(self.requests[0].url), 'https://example.test/anthropic/v1/messages')
        self.assertEqual(json.loads(self.requests[0].content)['model'], 'gateway-opus')

    def test_missing_relay_token_does_not_use_official_key(self):
        os.environ.pop('ANTHROPIC_AUTH_TOKEN')
        with self.assertRaisesRegex(PipelineError, 'ANTHROPIC_AUTH_TOKEN'):
            self.call()
        self.assertEqual(self.requests, [])

    def test_error_safe_and_no_retry(self):
        with self.assertRaisesRegex(PipelineError, '401') as err:
            self.call(status=401)
        self.assertNotIn('secret-error-detail', str(err.exception))
        self.assertEqual(len(self.requests), 1)

    def test_refusal_and_truncation(self):
        for stop in ['refusal', 'max_tokens']:
            with self.subTest(stop=stop), self.assertRaises(PipelineError):
                self.call(stop=stop)

    def test_bad_json_and_schema(self):
        for content in ['not json', '{"ko":123}', '{}']:
            with self.subTest(content=content), self.assertRaises(PipelineError):
                self.call(content=content)

    def test_fenced_json(self):
        self.assertEqual(self.call(content='```json\n{"ko":["번역"]}\n```'), {'ko': ['번역']})

    def test_translation_entry(self):
        self.assertEqual(self.call(invoke=lambda: director.translate_ko(['翻訳'], self.cfg)), ['번역'])

    def test_script_entry(self):
        data = json.loads((ROOT / 'tests/fixture_script.json').read_text(encoding='utf-8'))
        # Stored scripts may include editor metadata. API schema intentionally omits it.
        data = {k: data[k] for k in director.SCHEMA['required']}
        result = self.call(content=json.dumps(data), invoke=lambda: director.generate_script('吉田仁人', self.cfg))
        self.assertEqual(result['name_ja'], data['name_ja'])

    def test_recommend_entry(self):
        data = {'people': [{'name_ja': '人物', 'name_ko': '인물', 'kind': '배우', 'reason_ko': '이유',
                            'known_for_ko': ['작품']},
                           {'name_ja': '既出 人物', 'name_ko': '기출', 'kind': '배우', 'reason_ko': '이유',
                            'known_for_ko': []}]}
        result = self.call(content=json.dumps(data), invoke=lambda: director.recommend_people(
            self.cfg, 5, ['여자 배우'], exclude=['既出人物']))
        self.assertEqual([p['name_ja'] for p in result], ['人物'])   # 제외 목록은 공백 무시하고 걸러냄
        user = json.loads(self.requests[0].content)['messages'][0]['content']
        self.assertIn('女優', user)
        self.assertIn('既出人物', user)

if __name__ == '__main__':
    unittest.main(verbosity=2)
