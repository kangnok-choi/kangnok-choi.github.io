# Inverse problem

왜 inverse problem을 공부하냐?

Machine unlearning을 공부하다가, unlearning이 결국에는 inverse problem이랑 개념적으로 동일하다고 생각했음.

---
어느 부분이 유사하냐? 

Machine unlearning과 inverse problem 모두 이미 결과에 반영된 정보를 역추적하여 데이터 단위로 분리하는 과정 

- 이를 통해 unlearning에서는 특정 data point의 정보를 제거할 수 있음.
- 즉, data point $x$가 모델 학습에 미치는 영향을 조사할 수 있음. 
---


특히 나는 ill-posed discrete inverse problem에 관심 있다. 딥러닝 모델은 ill-posed 성질을 띄고 있으며, computation이 가능해야 하기 때문. 

여기에는 PC Hansen의 discrete inverse problems: insight and algorithms를 읽고 정리한다