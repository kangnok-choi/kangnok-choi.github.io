---
name: thumbnail_maker
description: >
  이 블로그(kangnok-choi.github.io, MkDocs)의 포스트/논문/카테고리에 어울리는 SVG 커버(썸네일)를
  블로그 비주얼 아이덴티티에 맞춰 생성하는 스킬. "썸네일 만들어줘", "커버 이미지", "cover.svg",
  "카드 이미지", "프로필 배경" 같은 요청은 물론, 새 논문 분석 글이나 새 카테고리/그룹 페이지를
  만들면서 카드에 넣을 이미지가 필요한 상황이면 명시적으로 SVG를 언급하지 않아도 이 스킬을 사용.
---

# 블로그 SVG 커버(썸네일) 생성

이 저장소의 카드형 UI(post-card, cat-banner, profile-card)는 각각 `cover_images/*.svg`를
배경으로 쓴다. 이 스킬은 글의 핵심 개념을 시각적 메타포로 표현한 SVG를 사이트 스타일에
맞춰 만드는 방법을 담는다.

## 커버 종류와 규격

| 용도 | 파일 위치 | viewBox | 표시 방식 (extra.css) |
|---|---|---|---|
| **포스트 커버** (post-card) | `docs/<대상>/cover_images/<슬러그>.svg` | `0 0 600 360` (5:3) | 카드도 `aspect-ratio: 5/3` → 거의 크롭 없음 |
| **카테고리 배너** (cat-banner) | `docs/<user>/<category>/cover_images/cover.svg` | `0 0 1200 240` | 높이 195px 고정, `background-position: center right` → **왼쪽이 잘림**. 왼쪽엔 제목 텍스트가 오버레이됨 |
| **프로필 배경** (profile-card) | `docs/<user|group>/cover_images/profile.svg` | `0 0 480 480` | `opacity: 0.26`으로 텍스트 뒤에 깔림, 세로형 카드(5/6)라 좌우가 잘림 |

- 포스트 커버는 슬러그 이름으로 만든다(예: `induction-heads.svg`). 그 섹션의 `cover.svg`는
  전용 커버가 없을 때의 폴백.
- 모든 커버는 `preserveAspectRatio="xMidYMid slice"` — 카드 크기에 따라 가장자리가
  잘릴 수 있으니 **핵심 도형은 안전 영역 안에** 배치한다:
  - 포스트 커버: 가장자리 ~30px 여백
  - 배너: 그림은 **오른쪽 절반(x ≥ 600)** 에, 왼쪽은 배경/패턴만 (제목이 덮으므로)
  - 프로필: 좌우가 잘려도 무방한 균일한 패턴/질감 위주, 텍스트 가독성을 해치는 진한 도형 금지

## 비주얼 아이덴티티 (기존 커버들과 통일)

기존 커버(`docs/*/cover_images/*.svg`)를 1–2개 열어보고 톤을 맞춘다. 공통 문법:

- **배경**: 연한 블루 그라디언트. `#f3f6fc → #dbe6f7` 또는 `#fafbfd → #e8eef9`.
  다크 모드에서도 커버는 밝게 유지하는 것이 기존 관례(카드 대비가 오히려 좋음).
- **단일 액센트 컬러**: `#1e4ea3` (사이트 `--accent`). 색을 늘리는 대신
  `fill-opacity`(0.05 ~ 0.85)로 위계를 만든다. 강조 하나만 진하게, 나머지는 흐리게.
- **질감**: 은은한 도트 패턴 (`r=1`, `fill-opacity="0.08"`, 14~16px 간격) — 선택 사항.
- **글자**: 꼭 필요할 때만, `font-family="JetBrains Mono, monospace"`로 토큰/라벨 수준만.
  **포스트 제목을 커버에 넣지 않는다** — 카드 본문에 이미 제목이 있다.
- **ID 충돌 방지**: gradient/pattern id에 슬러그 접두사를 붙인다 (예: `id="qurating-bg"`).

## 디자인 프로세스

1. **글을 읽고 핵심 메커니즘 하나를 뽑는다.** 논문/포스트의 주제가 아니라
   *가장 그림이 되는 아이디어 하나*. 커버는 다이어그램이지 일러스트가 아니다.
   - 예: induction heads → `A B C D A ?` 토큰열과 back-reference 화살표
   - 예: attention → 하삼각 어텐션 매트릭스 타일
   - 예: 데이터 선별(QuRating) → 문서 뭉치에서 일부만 진하게 골라내는 그리드
2. **도형 문법으로 표현한다**: rect 타일, 원 노드 + 곡선 엣지(`Q` 베지어), 모노스페이스
   토큰, 점선(`stroke-dasharray`) 정도면 대부분 표현된다. 화살촉은 작은 `polygon`.
3. **검증한다**:
   - XML 파싱: `python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse(sys.argv[1])" <파일>`
   - 렌더 확인: `./serve.sh`가 떠 있으면 해당 페이지에서, 아니면 브라우저로 SVG 직접 열기.
     원격(SSH)이면 유저에게 터널(`ssh -N -L 8000:localhost:8000 ...`) 안내.
4. **카드에 연결한다**: 해당 `index.md`의 카드에서 커버 경로를 교체/추가.
   ```html
   <div class="post-card__cover" style="background-image: url('cover_images/<슬러그>.svg')"></div>
   ```
   (새 글 등록 자체는 paper-analysis 스킬의 몫 — 이 스킬은 SVG 생성과 경로 연결까지만.)

## 포스트 커버 골격 예시

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 360" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="myslug-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fafbfd"/>
      <stop offset="1" stop-color="#e8eef9"/>
    </linearGradient>
    <pattern id="myslug-dots" width="14" height="14" patternUnits="userSpaceOnUse">
      <circle cx="1" cy="1" r="1" fill="#1e4ea3" fill-opacity="0.08"/>
    </pattern>
  </defs>
  <rect width="600" height="360" fill="url(#myslug-bg)"/>
  <rect width="600" height="360" fill="url(#myslug-dots)"/>

  <!-- 핵심 메타포: 중앙 안전 영역(30px 여백) 안에 -->
  <g transform="translate(60, 110)" fill="#1e4ea3">
    <!-- ... -->
  </g>
</svg>
```

## 주의

- 파일은 순수 SVG로 — 외부 이미지/폰트 링크, `<script>`, CSS `@import` 금지
  (GitHub Pages에서 background-image로 로드되므로 외부 참조는 렌더가 깨진다).
- 시안을 여러 개 만들 때는 같은 폴더에 이름을 달리해 저장하고 유저가 고르게 한다.
- 기존 커버를 덮어쓰기 전에는 반드시 유저에게 확인한다.
