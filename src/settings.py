"""
자동매매 설정 읽기 — config/autotrade.yaml 위에 config/champion.yaml(진화 모델이 승격한 설정)을 덮어쓴다.

champion.yaml 은 src/evolve.py promote 가 만든다. 지우면 원래 설정으로 돌아간다.
판단 엔진·실행기·백테스트·화면이 모두 이 함수로 설정을 읽어야 '검증한 설정 = 주문하는 설정'이 유지된다.
"""
import os

import yaml

CFG = "config/autotrade.yaml"
CHAMPION = "config/champion.yaml"


def load(path=CFG, champion=CHAMPION):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    if champion and os.path.exists(champion):
        ch = yaml.safe_load(open(champion, encoding="utf-8")) or {}
        cfg["strategy"] = {**cfg.get("strategy", {}), **(ch.get("strategy") or {})}
        cfg["champion"] = {k: v for k, v in ch.items() if k != "strategy"}
    return cfg
