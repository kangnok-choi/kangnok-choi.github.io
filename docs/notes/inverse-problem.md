# Inverse Problem

역문제(Inverse Problem)에 대한 노트입니다.

## 정의

역문제는 관측된 결과로부터 원인이나 모델의 파라미터를 추정하는 문제를 말한다.

!!! note "정의"
    **Forward Problem**: 모델 → 결과  
    **Inverse Problem**: 결과 → 모델

## 예시

일반적인 역문제의 예시:

- **의료 영상**: CT, MRI 스캔 데이터로부터 내부 구조 복원
- **지진파 분석**: 지진파 데이터로부터 지구 내부 구조 추정
- **이미지 복원**: 흐릿한 이미지로부터 원본 이미지 복원

## 특성

역문제는 다음과 같은 특성을 가진다:

1. **ill-posed**: 해가 유일하지 않거나, 해가 존재하지 않거나, 해가 불안정할 수 있음
2. **정규화 필요**: ill-posed 문제를 해결하기 위해 정규화(regularization) 기법 필요

## 관련 기법

- Tikhonov regularization
- Total Variation (TV) regularization
- Bayesian inference

---

!!! tip "참고"
    이 노트는 지속적으로 업데이트됩니다.
