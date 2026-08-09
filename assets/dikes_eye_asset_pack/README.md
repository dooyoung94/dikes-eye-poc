# Dike's Eye Asset Pack

직접 GitHub 레포의 `assets/` 폴더에 업로드해서 사용하는 프론트 디자인 asset 세트입니다.

## 포함 파일

- `assets/background/agora_bg_desktop.webp` : 데스크톱 전체 배경
- `assets/background/agora_bg_left.webp` / `center.webp` / `right.webp` : 반응형/좌우 배치용 배경
- `assets/hero/hero_frame.svg` : Hero 금박 프레임
- `assets/hero/scales_gold.svg` : 디케 저울
- `assets/hero/laurel_gold.svg` : 월계관
- `assets/hero/divider_gold.svg` : 금박 구분선
- `assets/icons/*_icon.svg` : 6단계 설명/분석 아이콘
- `assets/cards/parchment_card_texture.webp` : 판정 카드 질감
- `assets/cards/input_box_texture.webp` : 입력 박스 질감
- `assets/cards/card_corner.svg` : 카드 금박 모서리
- `assets/texture/marble_texture.webp` : 마블 질감
- `dike_assets.css` : 적용 예시 CSS

## 권장 UI 매핑

1. 시민의 목소리 → `consensus_icon.svg`
2. 당신의 조건 → `condition_icon.svg`
3. Rashomon → `rashomon_icon.svg`
4. Wald → `wald_icon.svg`
5. Dike의 저울 → `balance_icon.svg`
6. Solomon Choice → `solomon_icon.svg`

## 중요

배경/장식 asset에는 점수나 분석 문장을 넣지 않았습니다.
실제 점수, Evidence, 설명은 Streamlit HTML에서 동적으로 출력하세요.
