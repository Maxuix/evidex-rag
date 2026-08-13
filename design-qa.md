# Design QA — 回答过程

## Comparison target

- Source visual truth: `/private/tmp/rag-execution-trace-audit/03-answer-process-reference.png`
  (Figma file: `https://www.figma.com/design/4HLtVif7Nt0mJ0SyVuoRoe`)
- Browser-rendered implementation:
  - Desktop: `/private/tmp/rag-execution-trace-audit/09-implementation-desktop-final.png`
  - Mobile: `/private/tmp/rag-execution-trace-audit/10-implementation-mobile-final.png`
  - Mobile technical details: `/private/tmp/rag-execution-trace-audit/11-implementation-mobile-technical-final.png`
- Route: `http://127.0.0.1:3000/`
- State: a completed historical answer with “回答过程” expanded; technical details collapsed
  for the primary comparison and expanded for the secondary mobile comparison.
- Data state: 3 retrieval calls, 23 candidate references, and 1 final citation. The source mock uses
  3, 19, and 2 respectively; this difference is expected because the implementation renders real
  run data rather than fixture copy.

## Viewport and normalization

- Source pixels: 1320 × 1432. CSS design size: 1320 × 1432. Density: 1× Figma capture.
- Desktop implementation pixels: 1280 × 1100. CSS viewport: 1280 × 1100. Density: 1×
  (captured pixels match CSS pixels).
- Mobile implementation pixels: 390 × 844. CSS viewport: 390 × 844. Density: 1×.
- The source is a standalone component board and includes explanatory design annotations below the
  component. The implementation is evaluated in its real chat canvas. Comparison therefore uses
  the answer-process card as the focused surface and treats surrounding chat/sidebar content and
  the source annotation cards as out of scope.

## Full-view comparison evidence

The source and desktop implementation captures were opened together in one comparison pass. The
implementation preserves the intended hierarchy: title/status header, plain-language outcome,
three summary metrics, four semantic stages, collapsed technical details, and a privacy note. It
adapts the standalone reference to the existing chat width without changing the information order.

The desktop browser check reported document width 1280/1280 with no horizontal overflow. The
mobile check reported document width 390/390 and answer-process width 326/326. Five completed
historical traces were collapsed by default.

## Focused region comparison evidence

- Mobile primary state: `/private/tmp/rag-execution-trace-audit/10-implementation-mobile-final.png`
  verifies wrapped summary copy, stacked metric pills, readable stage cards, and preserved hierarchy
  at 390 px.
- Mobile technical state:
  `/private/tmp/rag-execution-trace-audit/11-implementation-mobile-technical-final.png` verifies the
  one-column diagnostic grid. Its measured width was 302/302 with no overflow.
- Browser DOM checks found zero visible `ev_*` references and zero raw `ok`, `rejected`, or `failed`
  status tokens. Both disclosure controls changed `aria-expanded` on click. Browser console warning
  and error count was zero.

## Required fidelity surfaces

- Fonts and typography: uses the product's existing font stack and text tokens. Heading/body/meta
  hierarchy is visibly distinct, wraps without clipping on desktop and mobile, and avoids the old
  machine-log density.
- Spacing and layout rhythm: card padding, section gaps, radii, borders, and vertical rhythm follow
  the reference while fitting the narrower production chat column. Desktop and 390 px layouts have
  no horizontal overflow.
- Colors and visual tokens: existing neutral, green-complete, amber-candidate, and blue-focus tokens
  map to the source semantics with readable contrast and no new palette.
- Image quality and asset fidelity: the target component contains no product imagery, logos, or
  raster assets. No image asset was substituted with CSS or inline SVG. Numbered stage markers are
  semantic text, not a replacement for a required image asset.
- Copy and content: all counts come from the real run contract. Candidate material and final citations
  are explicitly differentiated; tool IDs, event references, protocol statuses, and provider payloads
  are not exposed as process content.
- States and accessibility: completed runs are collapsed by default; active runs remain visible and
  use the latest confirmed snapshot. Disclosure controls are native buttons with `aria-expanded` and
  visible focus styles. Direct low-level keyboard dispatch could not be exercised because the local
  browser security policy rejected that automation path; native button keyboard semantics remain in
  place.

## Findings

- No actionable P0, P1, or P2 differences remain.
- [P3] Completion markers use ordered numbers rather than the reference's check icons. This is an
  intentional product deviation: it makes the sequence explicit and avoids introducing an icon
  dependency for a decorative-only difference. If an existing icon set is adopted later, the final
  state could use its standard completion glyph without changing layout.

## Open questions

- None blocking. Live in-progress copy is bounded by the current progress snapshot contract; richer
  per-tool user-facing narration would require a separate backend contract change.

## Comparison history

1. Original baseline — P1: the execution card exposed six synthetic stages plus `ev_*` references
   and raw protocol statuses. Fix: replaced it with four user-goal stages, real aggregate counts,
   and a separately collapsed diagnostic layer. Evidence:
   `/var/folders/6c/h_0cyj2x7b9383tdk5tvvn180000gn/T/codex-clipboard-b4299704-a095-400c-a811-40a5dd0c2415.png`
   versus the three final implementation captures above.
2. First implementation pass — P2: every completed historical answer was expanded, creating excessive
   conversation height. Fix: terminal traces now initialize collapsed while active traces remain open.
   Post-fix DOM evidence: five historical controls all reported `aria-expanded="false"` on load.
3. Interaction pass — P2 accessibility risk: native `details` behavior was unreliable under the local
   browser's interaction harness. Fix: replaced both disclosures with explicit `button` controls,
   React state, `aria-expanded`, and `:focus-visible` styling. Post-fix click checks changed both
   controls to expanded and exposed the intended content only.

## Implementation checklist

- [x] Replace machine-readable trace rows with four user-facing stages.
- [x] Render retrieval, candidate, and final-citation counts from real run data.
- [x] Keep terminal traces and technical details collapsed by default.
- [x] Hide internal references and raw protocol statuses from visible process content.
- [x] Verify desktop and 390 px responsive layouts with real historical data.
- [x] Verify disclosure interactions, semantic state, console output, and production frontend build.

## Follow-up polish

- P3 only: adopt the product's future standard completion icon if an icon library becomes part of the
  existing design system.

final result: passed
