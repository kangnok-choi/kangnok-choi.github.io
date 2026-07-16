---
name: paper-analysis
description: >
  이 블로그(kangnok-choi.github.io, MkDocs)에 논문 분석 글을 올리는 워크플로우.
  "새 논문 정리/분석 시작", "템플릿 불러와줘", "페이지 생성", "논문이랑 내가 쓴 거 맞는지 봐줘" 같은 요청에 사용.
  세 단계(① 새 글 세팅 ② 검토③ PR)로 구성. 각 유저는 자기 브랜치에서 작업하고,
  문서를 쓰면 반드시 PR로 올린다.
---

# 논문 분석 글 워크플로우

이 저장소는 MkDocs(Material) 기반 개인/그룹 연구 블로그다. 논문 분석 글 한 편을 올리는
전 과정을 3단계로 돕는다. **단계는 유저가 어디까지 했는지 보고 판단**한다 — 빈 템플릿을
원하면 ①, 초안을 다 썼으면 ②, 확인 끝났으면 ③.

## 대원칙 (항상 지킬 것)

- **브랜치**: 각자 자기 유저 브랜치에서 작업한다. `main`에서 직접
  문서 작업/커밋 금지. 브랜치가 없으면 만들고 (`git switch -c <branch>`), 있으면 그걸 쓴다.
- **PR 필수**: 문서를 쓰면 반드시 PR로 올린다. `git push -u origin <branch>` 후
  `gh pr create --base main`. 저장소의 `push.sh`(add/commit/push to current)는 main 직push
  용이라 이 워크플로우에선 쓰지 않는다.
- **사실 우선**: 논문 내용은 기억에 의존하지 말고 **항상 원문을 열어 확인**한다.
- **유저 목소리 보존**: 보강할 때 유저가 쓴 문장·톤·구조를 살리고, 정확성/디테일만
  더한다. 통째로 다시 쓰지 않는다.

---

## 1. 새 글 세팅 — 새 논문 시작

1. **정보 확인**: 논문(제목·저자·연도·학회·arXiv 링크), 대상 위치, 유저(=브랜치)를 파악.
   - 대상 위치: 개인이면 `docs/<user>/<category>/`, 그룹이면 `docs/<group>/`.
     (그룹 예: `docs/multilingual_research_group/`, `docs/world_model_group/`)
   - 없는 정보만 짧게 묻는다.
2. **브랜치 확인/생성** (대원칙대로).
3. **페이지 생성**: `docs/<대상>/<슬러그>.md`를 만든다. 슬러그는 소문자 kebab (`<paper-slug>.md`).
   - 헤더 형식: `##### *저자 et al.* 논문 풀 타이틀 *(연도 학회)*`

4. **목록/네비 등록**:
   - 카테고리(또는 그룹) `index.md`의 `<div class="post-grid">` 안에 카드 추가:
     ```html
     <a class="post-card" href="<슬러그>/">
       <div class="post-card__cover" style="background-image: url('cover_images/<커버>.svg')"></div>
       <div class="post-card__body">
         <div class="post-card__meta">저자 et al. 연도 학회</div>
         <h3 class="post-card__title">짧은 제목</h3>
         <p class="post-card__desc">한 줄 설명</p>
       </div>
     </a>
     ```
     (전용 커버가 없으면 그 폴더의 `cover_images/cover.svg` 재사용)
   - `mkdocs.yml`의 `nav:` 해당 섹션에 `- 짧은 제목: <경로>.md` 추가. 섹션에 항목이
     처음 생기면 `- <index.md>` + 하위 항목 형태로 펼친다.
5. 유저에게 "이제 초안 채우세요"라고 알리고 ②를 대기.

## ② 검토·보강 — 유저가 초안을 다 썼을 때

목표: **플레이스홀더 제거 → 원문 대조 사실검증**.

1. **플레이스홀더/지침 제거**: 템플릿에서 남은 안내 문구를 전부 지운다.
   - 이탤릭 지침(예: `*지침: ...*`, `*저자의 주장(Storyline)을 해체...*`,
     `*리뷰어의 시각으로...*`), 그리고 유저가 답을 안 채운 프롬프트 불릿
     (예: "기존 연구의 한계점(Pain Point)은 무엇인가?")을 삭제.
   - 유저가 실제로 쓴 내용은 남긴다.
2. **원문 대조 사실검증**: arXiv를 연다.
   - 개요/주장: `https://arxiv.org/abs/<id>`
   - 표·부록·수식 등 디테일: `https://ar5iv.labs.arxiv.org/abs/<id>` (본문에 없는
     appendix 표까지 있음 — 리다이렉트 뜨면 그 URL로 다시 fetch)
   - 유저의 각 주장을 원문과 대조:
     - **맞음** → 그대로 두되 정확한 수치/이름/인용문으로 보강.
     - **틀림/부정확** → 정정하고, 근거(정확한 숫자나 원문 인용)를 함께 남긴다.
     - **원문에서 못 찾음** → 단정하지 말고 ⚠️로 "확인 필요" 표시 후 유저에게 물음.
   - 그림이 표/figure라면 이미지를 직접 읽어 숫자까지 교차검증.
3. **이미지 로컬화**: 외부 URL(특히 `github.com/user-attachments/assets/...`)은
   깨질 수 있으니 로컬로 내려받아 `docs/<대상>/post_images/<슬러그>/`에 저장하고
   `<figure markdown="span"> ![alt](post_images/<슬러그>/<파일>){ width="90%" }</figure>`
   로 교체.
   - user-attachments URL은 인증이 필요해 curl이 404 난다. 유저 터미널의 gh 로그인
     토큰으로 받는다:
     `curl -L -H "Authorization: token $(gh auth token)" -o <경로> "<url>"`
   - 받은 파일이 진짜 이미지인지(용량·매직바이트/Read로 미리보기) 확인.
4. **수식 렌더 확인** (아래 규칙 참고).

## ③ PR — 확인 끝났을 때

1. **로컬 미리보기**: `./serve.sh -a 127.0.0.1:8000` (백그라운드). live-reload.
   원격(SSH) 환경이면 유저에게 맥에서 `ssh -N -L 8000:localhost:8000 <user>@<host>`로
   터널 열라고 안내.
2. 유저 확인 후: `git add . && git commit -m "docs: add <논문> analysis"`
   → `git push -u origin <branch>` → `gh pr create --base main`.

---

## 저장소 규칙 (참고)

### 디렉터리 구조
- `docs/<user>/index.md` — 개인 프로필(카테고리 배너)
- `docs/<user>/<category>/index.md` — 카테고리 목록(`post-grid`)
- `docs/<user>/<category>/<slug>.md` — 논문 글
- `docs/<group>/index.md` — 그룹은 카테고리 없이 글이 바로 붙음
- 그림: `.../post_images/<slug>/*.png`, 카드 커버: `.../cover_images/*.svg`
- 새 개인 카테고리를 만들면 `docs/<user>/index.md` 배너에도 추가.

### 페이지 프론트매터
```markdown
---
title: 짧은 제목
---

# 짧은 제목

##### *저자 et al.* 논문 풀 타이틀 *(연도 학회)*
```

### 수식 (KaTeX + pymdownx.arithmatex)
- **인라인**: `$...$` — 문장 어디든 OK.
- **디스플레이(크게)**: `$$` 는 반드시
  1. 여는 `$$`, 내용, 닫는 `$$` 를 각각 **자기 줄**에,
  2. 앞뒤로 **빈 줄**,
  3. `$$` 뒤에 마침표 등 **트레일링 문자 금지**,
  4. **리스트 안에 들여쓰기하면 깨진다** → 들여쓰기 0(top-level)으로 뺄 것.
  ```markdown
  앞 문장:

  $$
  \mathcal{L} = ...
  $$

  뒤 문장
  ```
- 확인법: 빌드 HTML에 `<div class="arithmatex">\[` (블록) 이 뜨면 성공.
  `<p>$$` 로 남아 있으면 실패.

### 커밋/배포
- 커밋 메시지: `docs: ...` 형태.
- PR로만 main에 반영. `push.sh`는 이 워크플로우에서 사용 금지.
