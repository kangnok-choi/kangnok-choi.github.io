---
title: Induction Heads
---

# Induction Heads

##### - Elhage et al. *A Mathematical Framework for Transformer Circuits* (2021) 
##### - Olsson et al. *In-context Learning and Induction Heads* (2022)

Induction head는 Transformer가 in-context learning을 수행하는 가장 단순하면서도 가장 자주 인용되는 회로다.
두 개의 attention head가 협력하여 `[A][B] ... [A] → [B]` 라는 **패턴 복사** 규칙을 구현한다.
2-layer attention-only 모델에서도 또렷이 관찰되며, 학습 손실 곡선에서 *phase change*로 등장한다.

> "Induction heads are a circuit that copies, and copies in a way that depends on context.
> They are arguably the first nontrivial in-context learning algorithm a transformer learns."
> — Olsson et al., 2022

---

## 1 무엇을 하는가

토큰 시퀀스가 `... A B ... A ?` 로 끝날 때, induction head는 마지막 위치의 logit에 `B`를 끌어올린다.
즉, **앞서 등장했던 bigram을 복사**한다. 이는 단순한 `n`-gram 통계로는 설명되지 않는데,
회로가 토큰 자체가 아니라 **상대적 위치 패턴**에 반응하기 때문이다.

좀 더 구체적으로, induction head는 두 단계로 구성된다.

1. **Previous-token head (Layer $\ell$)**: 각 위치 $t$의 representation에 위치 $t-1$의 정보를 복사한다.
2. **Induction head (Layer $\ell+1$)**: 현재 토큰 `A`를 query로 사용해, 직전에 `A`가 등장했던 위치를 attend하고, 그 위치의 *다음 토큰* `B`를 value로 가져온다.

핵심은 두 번째 head가 `prev_token` 정보를 K-side에서 받아 ==content-based matching== 을 한다는 점이다.

## 2 수식으로 본 메커니즘

Attention-only 2-layer 모델을 가정하자. Residual stream $x_t \in \mathbb{R}^d$에 대해 layer 1 head는

$$
x_t^{(1)} = x_t + \sum_{s \le t} \alpha_{ts}^{(1)} \, W_{OV}^{(1)} \, x_s,
$$

이때 $\alpha_{ts}^{(1)}$가 $s = t-1$에서 거의 1이 되도록 학습되면 (previous-token head),
$x_t^{(1)} \approx x_t + W_{OV}^{(1)} x_{t-1}$ 이 된다.

Layer 2의 induction head는

$$
\alpha_{ts}^{(2)} \propto \exp\!\bigl( (W_Q^{(2)} x_t)^\top (W_K^{(2)} x_s^{(1)}) \bigr).
$$

여기서 $x_s^{(1)}$이 $x_{s-1}$의 정보를 품고 있으므로, $W_Q^{(2)} W_K^{(2) \top}$가
$x_t$와 $x_{s-1}$ 사이의 유사도를 측정하도록 학습되면 `A`-매칭이 일어난다.

## 3 · 학습 동역학에서의 phase change

| Stage | Loss behavior | What's happening |
|---|---|---|
| **A** | 빠른 감소 (unigram → bigram) | embedding이 토큰 빈도를 학습 |
| **B** | 정체 (loss plateau) | 표면 통계 한계에 도달 |
| **C** | 급격한 감소 (induction bump) | induction head가 형성 — *phase change* |
| **D** | 완만한 개선 | longer-range circuit이 induction 위에 쌓임 |

이 **induction bump**는 모델 크기와 거의 무관하게 동일한 정성적 모양을 보인다.
Olsson et al.은 이를 in-context learning의 등장 시점으로 해석한다.

## 4 · 어떻게 확인하는가

가장 단순한 진단은 **prefix-matching score**다.

```python
import torch

def prefix_matching_score(attn_pattern, tokens):
    """
    attn_pattern: [seq, seq] — head's attention weights
    tokens: [seq] — input token ids
    returns: average attention from position t to position s,
             where tokens[s-1] == tokens[t] and s > 0.
    """
    seq = tokens.shape[0]
    score = 0.0
    count = 0
    for t in range(1, seq):
        for s in range(1, t):
            if tokens[s - 1] == tokens[t]:
                score += attn_pattern[t, s].item()
                count += 1
    return score / max(count, 1)
```

Induction head는 random tokens에서도 prefix-matching score ≈ 1.0에 가깝게 나온다.
일반 attention head는 같은 입력에서 0.01 수준에 머문다.

!!! note "추가 진단"
    - **Copying score**: induction head의 OV circuit이 입력 토큰을 그대로 logit에 흘려보내는가
      ($W_U W_{OV}$의 대각이 양수인가)
    - **Activation patching**: prompt의 `B` 위치 representation을 노이즈로 바꿨을 때 logit이 무너지는가

## 5 · 함의

- 학습된 회로를 **회로 수준에서** 식별·검증할 수 있다는 첫 번째 강한 증거였다.
- "in-context learning" 같은 행동적 현상이 사실은 한두 개의 작은 회로 위에 얹혀 있다는 가설을 뒷받침한다.
- 이후 작업(IOI circuit, S-inhibition, name-mover head 등)은 induction의 *구성 요소* 어휘를 그대로 빌려 쓴다.

## 6 · 읽으며 든 의문

- Induction head가 두 개 이상 등장하면 서로 redundant한가, 아니면 specialize되는가? *e.g.* one for code, one for natural language.
- Phase change의 시점은 어떤 학습률·dataset distribution에 가장 민감한가?
- SAE로 induction head의 OV-direction을 feature 단위로 쪼개면 어떤 모양인가?

---

##### Further reading

- Elhage, Nanda, Olsson et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Olsson, Elhage, Nanda et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Nanda. *TransformerLens* — `transformer_lens.HookedTransformer` 로 직접 attention pattern을 떠서 score를 재현해볼 수 있다.
