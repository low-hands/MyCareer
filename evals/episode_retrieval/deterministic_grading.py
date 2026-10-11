"""Frozen rule-based E2E grading; ambiguous cases need independent human review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

PROTOCOL_PATH = Path(__file__).parent / 'grading_v3/protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
GRADING_VERSION = PROTOCOL['version']
GRADING_DIGEST = hashlib.sha256(PROTOCOL_PATH.read_bytes() + Path(__file__).read_bytes()).hexdigest()


def _normalize(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())


def _matches(text, alternatives, *, regex=False):
    for alternative in alternatives:
        normalized = _normalize(alternative)
        pattern = normalized if regex else re.escape(normalized)
        if normalized[0].isdigit():
            pattern = r'(?<!\d)' + pattern
        if not regex and normalized[-1].isdigit():
            pattern += r'(?!\d)'
        if not regex and normalized.isascii() and normalized.isalpha():
            pattern = r'(?<![a-z0-9_])' + pattern + r'(?![a-z0-9_])'
        if re.search(pattern, text):
            return True
    return False


def _decision(status, reason, **evidence):
    return {'status': status, 'passed': status == 'passed', 'reason': reason,
            'grading_version': GRADING_VERSION, **evidence}


def grade_answer(case, data, answer):
    text = _normalize(answer)
    if not text:
        return _decision('failed', 'empty answer')
    # A denial/quotation/question is not an assertion that a record is absent.
    ambiguous = any(token in text for token in ('不是没找到', '并非没有', '不能说没找到',
        '不是没有', '并非没找到', '不能说没有', '不代表没有', '不意味着没有',
        '不等于没有', '没有遗漏', '没有否认', '没有删除', '假设', '如果', '可能', '不确定', '无法确定', '据说'))
    if not case['relevant']:
        absent = bool(re.search(
            r'(?:(?:没(?:有)?|未)(?:找到|查到|检索到|搜索到)|找不到|无法(?:找到|查到)|没有|无|不存在)[^。；;!?？]{0,40}'
            r'(?:记录|事件|反馈|拒信|结果|信息|报告)|(?:没有|无)[^。；;!?？]{0,30}(?:匹配|相关)(?:项|内容)', text))
        if not absent:
            return _decision('failed', 'explicit no-record assertion missing')
        # These identities are fixed in the grader, independent of retrieval aliases.
        organizations = {item['title'].split(' · ')[0] for item in data['corpus']
                         if ' · ' in item['title']}
        aliases = {'字节跳动': ('ByteDance', '字节'), '腾讯': ('Tencent', '鹅厂'),
                   '拼多多': ('Pinduoduo', 'PDD', '拼夕夕'), '阿里巴巴': ('Alibaba', '阿里')}
        question = _normalize(case['user_message'])
        other = []
        for company in sorted(organizations):
            names = (company, *aliases.get(company, ()))
            patterns = [r'(?<![a-z0-9_])' + re.escape(_normalize(n)) + r'(?![a-z0-9_])'
                        if n.isascii() else re.escape(_normalize(n)) for n in names]
            if not any(re.search(p, question) for p in patterns) and any(re.search(p, text) for p in patterns):
                other.append(company)
        if other:
            return _decision('failed', 'mentions another corpus organization', organizations=other)
        if '去年' in question and re.search(r'2026|今年', text):
            return _decision('failed', 'substitutes current-year evidence for last year')
        quoted_absence = bool(re.search(r'[“「"][^”」"]*(?:没有|未找到|没找到|不存在)[^”」"]*[”」"]', text))
        question_only = text.endswith(('?', '？'))
        if ambiguous or quoted_absence or question_only or re.search(r'但|不过|然而|实际|已完成|已投递|拿到offer|通过了|未通过|没通过|参加|经历|拒信原因是|[一二三]面.*(?:完成|结束|是)|\d{1,2}(?:月|/|-)\d{1,2}', text):
            return _decision('human_review', 'absence assertion has ambiguous or additional claims')
        return _decision('passed', 'explicit absence; no other corpus organization mentioned')
    rule = PROTOCOL['rules'].get(case['id'])
    if rule is None or set(rule['events']) != set(case['relevant']):
        return _decision('human_review', 'no frozen fact rubric for this case')
    missing = [name for name, alternatives in rule['required'].items() if not _matches(text, alternatives)]
    conflicts = [pattern for pattern in rule['forbidden'] if _matches(text, [pattern], regex=True)]
    if missing:
        return _decision('failed', 'required identity or key facts missing', missing=missing)
    organizations = {item['title'].split(' · ')[0] for item in data['corpus'] if ' · ' in item['title']}
    expected_titles = [item['title'] for item in data['corpus'] if item['id'] in case['relevant']]
    extras = [company for company in sorted(organizations)
              if _normalize(company) in text and not any(company in title for title in expected_titles)]
    if extras:
        return _decision('human_review', 'additional event claims need independent review', organizations=extras)
    if conflicts:
        return _decision('failed', 'conflicting event facts', conflicts=conflicts)
    if ambiguous:
        return _decision('human_review', 'key facts occur in ambiguous/quoted assertions')
    return _decision('passed', 'all frozen identity/time/round/outcome or practice fact groups matched')
