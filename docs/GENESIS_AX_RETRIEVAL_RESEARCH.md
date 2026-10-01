# GENESIS AI · Multi-view AX Retrieval Research Branch

이 브랜치는 기존 Dike 의사결정 POC를 바로 변경하지 않고, GENESIS AI 연구계획의 **추천·검색 모델 가설**을 별도 실험하기 위한 연구 브랜치다.

## 1. 문제 정의

기존 계획의 온라인 흐름은 다음에 가깝다.

```text
q
 -> Retrieve(q) = C
 -> Rashomon perspectives P1...Pk
 -> Rerank(Pk, C)
 -> Blind RCA
 -> Wald
 -> Evidence Graph / Dike
```

이 구조에서는 최초 `Retrieve(q)`에서 빠진 사례를 뒤의 Reranking이 복구할 수 없다. 따라서 다중관점 생성이 실제 **Recall 향상**으로 이어지지 않을 수 있다.

이 브랜치의 1차 가설은 다음과 같다.

> LLM으로 생성한 복수 문제관점을 검색 단계까지 Branch로 보존하면, 단일질의 후보군 또는 GPT-only 판단보다 AX 사례 Recall과 Strategy Coverage를 높일 수 있는가?

## 2. 제안 실행 구조

### Offline

```text
AX Cases
  ├─ Business Problem
  ├─ AX Target / Task
  └─ Business Context
        |
        +--> Multi-view embeddings
        |
Problem View
        |
        +--> Blind RCA / Factor Dictionary
               |
               +--> Root Cause Factor Anchors
```

Solution View는 이 단계와 분리해 Strategy Label을 구성한다.

### Online

```text
User business problem q
        |
        +--> original query branch q
        |
        +--> LLM perspectives P1...Pk
                  |
                  +--> Retrieve(P1)
                  +--> Retrieve(P2)
                  +--> ...
        |
Candidate Union
Retrieve(q) U Retrieve(P1) U ... U Retrieve(Pk)
        |
Multi-view + Root-Cause-aware reranking
        |
Branch results are preserved
        |
        +--> downstream Wald evidence
        +--> Evidence Graph
        +--> Dike / Agent
```

핵심은 **Rashomon을 Reranking 전용으로 쓰지 않고 Candidate Retrieval부터 사용**하는 것이다.

## 3. Multi-view 표현

사례 하나를 하나의 Dense Vector로만 표현하지 않는다.

```text
Case i
  ├─ z_problem
  ├─ z_task
  └─ z_context
```

현재 POC 기본 가중치는:

```text
problem = 0.45
task    = 0.35
context = 0.20
```

Perspective에 `focus=problem|task|context`를 주면 해당 View의 비중을 높인다.

이 구조의 목적은 "전체 문장이 얼마나 비슷한가" 외에 **어느 의미축에서 사례가 가까운가**를 보존하는 것이다.

## 4. Root Cause Anchor

Blind RCA의 결과를 단순 설명문으로 끝내지 않고, Pilot 단계에서 고정한 Factor Dictionary에 매핑해 공통 Anchor로 사용한다.

```text
Perspective
   -> Root Cause Factor
       -> related AX cases
           -> observed Strategies
```

현재 구현은 같은 Factor가 부여된 사례들의 Problem View embedding centroid를 Factor Anchor로 사용한다.

최종 reranking score:

```text
score =
    semantic_weight * multi_view_similarity
  + factor_weight   * root_cause_anchor_similarity
```

POC 기본값:

```text
semantic_weight = 0.85
factor_weight   = 0.15
```

이는 최종 논문 모델의 확정식이 아니라 실험 시작점이다.

## 5. Branch-preserving Candidate Expansion

기존:

```text
C = Retrieve(q)
Lk = Rerank(pk, C)
```

현재 실험:

```text
C* =
  Retrieve(q)
  U Retrieve(p1)
  U Retrieve(p2)
  ...
  U Retrieve(pk)

Lk = Rerank(pk, C*)
```

결과는 Branch별로 유지하고 중간 단계에서 평균 점수로 합치지 않는다.

## 6. Wald / Dike의 위치

이 연구 브랜치에서는 **Wald 자체를 neural loss로 정의하지 않는다.**

1차 단계:
- Wald = Root Cause Factor와 Strategy 관계의 독립적인 통계 Evidence
- Dike = 여러 Branch의 Evidence를 조기에 소거하지 않는 downstream decision layer

2차 실험에서만 다음을 검토한다.
- Wald 결과를 이용한 auxiliary constraint / regularization
- Branch collapse를 억제하는 diversity 또는 consistency loss
- Dike-inspired evidence aggregation

즉 통계검정과 최적화 목적함수를 구분한다.

## 7. 현재 구현

`src/ax_retrieval.py`

- `AXCase`
- `Perspective`
- `EmbeddingBackend` protocol
- `HashingEmbeddingBackend`: 외부 모델 없이 동작하는 smoke-test encoder
- `BranchPreservingAXRetriever`
  - Problem / Task / Context multi-view
  - Perspective별 independent retrieval
  - Candidate union
  - Root Cause Factor anchors
  - Branch별 reranking
- `recall_at_k`
- `strategy_coverage_at_k`

중요: `HashingEmbeddingBackend`는 최종 연구 모델이 아니다. 구조 검증용 fallback이며, 이후 실제 실험에서는 pretrained embedding / reranker backend를 주입한다.

## 8. 1차 검증

최소 Baseline:

| ID | 방법 |
|---|---|
| B0 | Single-query lexical/BM25 |
| B1 | Single-query dense retrieval |
| B2 | GPT/LLM + single RAG |
| B3 | LLM perspectives + same candidate pool reranking |
| Ours-1 | Perspective-wise candidate expansion |
| Ours-2 | Ours-1 + Multi-view |
| Ours-3 | Ours-2 + Root Cause Anchor |

핵심 지표:

- Recall@K
- nDCG@K
- Perspective Coverage
- Strategy Coverage
- Root Cause Coverage
- Branch Diversity

가장 먼저 확인할 것은 **Ours-1이 B3보다 최초 후보군에서 놓친 relevant case를 실제로 복구하는지**다.

## 9. Smoke Test

```bash
python -m unittest tests.test_ax_retrieval
```

테스트는 원질의 Top-1 후보군에서 누락된 사례가 별도 Perspective branch 검색을 통해 복구되고, Branch union 기준 Recall/Strategy Coverage가 증가하는 구조를 검증한다.

## 10. 다음 구현 순서

1. NIPA/NIA/KOSA AX Case schema loader
2. LLM Perspective JSON contract 및 fidelity/dedup guard
3. pretrained embedding backend
4. Cross-encoder reranker backend
5. gold relevance / Strategy labels 기반 evaluation runner
6. Blind RCA Factor Dictionary loader
7. Wald Factor-Strategy evidence matrix 연결
8. Evidence Graph / Dike downstream adapter

현재 브랜치에서는 1차적으로 **"기존 후보군을 그대로 재랭킹하는 것보다 관점별로 후보를 확장하면 Recall을 복구할 수 있는가"**를 코드 수준에서 분리해 검증할 수 있도록 만들었다.
