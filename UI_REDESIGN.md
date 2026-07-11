# GemmaClip UI Redesign Brief

## Codex execution document

This document is the implementation brief for a full frontend visual redesign of GemmaClip.

The goal is not to add more features. The goal is to make the existing product feel more futuristic, minimal, cohesive, premium, and technically trustworthy while preserving all current behavior.

Codex should read this entire file before changing code.

---

## 1. Required starting context

Before coding, read:

1. `AGENTS.md`
2. `docs/WEB_APP_SPEC.md`
3. `docs/WEB_DEVELOPMENT.md`
4. `web/src/styles.css`
5. `web/src/routes/index.tsx`
6. `web/src/routes/quick.tsx`
7. `web/src/routes/lab.tsx`
8. `web/src/routes/lab.$runId.tsx`
9. every file under `web/src/routes/lab.$runId.*.tsx`
10. the shared components under `web/src/components/`
11. `web/src/lib/types.ts`
12. the existing frontend tests

Create a new implementation branch from the commit containing this document:

```text
feature/gemmaclip-futuristic-ui
```

Do not implement the redesign directly on a backend or routing-fix branch.

---

## 2. Product positioning

GemmaClip has two product surfaces:

### Quick Caption

A simple path for a normal user:

```text
Upload video
→ wait while GemmaClip processes it
→ receive grounded captions
```

This surface must feel calm, simple, and almost effortless.

### Gemma Lab

A transparent inspection surface for builders and judges:

```text
Video
→ Frames
→ Audio
→ Evidence
→ Captions
→ Compare
```

This surface must feel like a precise AI instrument panel rather than a normal dashboard.

The product promise remains:

> Simple for everyone. Transparent for builders.

The redesign should make that promise visible through hierarchy, progressive disclosure, truthful status communication, and clear separation between the simple path and the technical path.

---

## 3. Design concept

### Concept name

**Signal Glass**

### Product metaphor

GemmaClip should feel like a calm glass-box AI instrument:

- dark precision surfaces
- sparse typography
- clean signal lines
- structured evidence
- visible model decisions
- restrained motion
- clear operational status

The product must not look like:

- a generic SaaS dashboard
- a gaming interface
- a cryptocurrency website
- a neon cyberpunk poster
- a terminal emulator
- a collection of unrelated shadcn cards
- a landing page full of marketing sections

“Futuristic” should come from precision, depth, spacing, typography, data presentation, and responsive motion—not from excessive glow, gradients, particles, or neon colors.

---

## 4. Problems in the current interface

The current interface is functional but visually inconsistent with the intended direction.

### Current issues to solve

1. The landing and Quick Caption surfaces use a warm paper/editorial visual language while Gemma Lab uses a darker technical language.
2. The serif display typography feels editorial rather than futuristic.
3. The landing page contains too many marketing sections for a focused prototype.
4. Repeated rounded cards create visual noise and make every element look equally important.
5. The horizontal Lab stepper feels like a website tab bar, not a persistent pipeline instrument.
6. Processing states are truthful but visually passive.
7. Technical metadata is present, but the hierarchy between user-facing results and builder-facing diagnostics is weak.
8. Caption cards devote too much space to chrome and too little to the caption text.
9. Model routing and evidence are important differentiators but do not yet feel like the center of the Lab experience.
10. The current design switches between light and dark modes by route instead of presenting one cohesive product world.

The redesign should solve these issues without changing the API, route behavior, stored run contract, or backend pipeline.

---

## 5. Non-negotiable functional constraints

Do not change any of the following unless a separate task explicitly requests it:

- backend APIs
- request or response schemas
- Python pipeline behavior
- provider routing behavior
- run storage
- upload limits
- supported file types
- stage status semantics
- error handling semantics
- polling behavior
- generation outcome semantics
- the leaderboard contract
- `/input/tasks.json` to `/output/results.json`

Do not modify:

- `lovable/`
- secret handling
- provider credentials
- deployment configuration
- generated run data
- media artifacts

Do not introduce mock results.

Do not enable controls whose backend actions are still intentionally disabled.

Do not make disabled controls look interactive.

Do not expose raw chain-of-thought, credentials, private endpoints, authorization headers, base64 media, or full provider response bodies.

---

## 6. Technical constraints

Use the existing frontend stack:

- React
- TypeScript
- Vite
- TanStack Router
- TanStack Query
- Tailwind CSS v4
- existing shadcn-style primitives
- Lucide icons

Avoid adding a new component library.

Avoid adding a heavyweight animation library. CSS transitions are sufficient.

Avoid adding external runtime font dependencies. Use the existing font stack or system-safe fallbacks.

New reusable components are allowed when they reduce repetition and improve consistency.

Prefer semantic components and CSS variables over route-specific one-off styling.

---

## 7. Global visual system

### 7.1 Default theme

Use one cohesive dark theme across Landing, Quick Caption, and Gemma Lab.

The default visual environment should be near-black, not pure black.

Suggested values:

```css
:root {
  --background: #07090d;
  --foreground: #f4f6f8;

  --card: #0c1017;
  --card-foreground: #f4f6f8;

  --popover: #10151e;
  --popover-foreground: #f4f6f8;

  --primary: #f4f6f8;
  --primary-foreground: #080a0e;

  --secondary: #131923;
  --secondary-foreground: #e9edf2;

  --muted: #111720;
  --muted-foreground: #8994a5;

  --accent: #171e29;
  --accent-foreground: #f4f6f8;

  --border: rgba(255, 255, 255, 0.085);
  --input: rgba(255, 255, 255, 0.11);
  --ring: #ff6b3d;

  --ember: #ff6b3d;
  --ember-soft: rgba(255, 107, 61, 0.12);

  --lab: #72a7ff;
  --lab-soft: rgba(114, 167, 255, 0.11);

  --success: #54d39a;
  --warning: #f1b65a;
  --danger: #ff6f78;
}
```

Equivalent OKLCH values are acceptable and may be preferable because the project already uses OKLCH.

### 7.2 Color discipline

Use color sparingly.

Primary accent:

```text
Ember orange
```

Use it for:

- primary actions
- active pipeline stage
- selected frame
- important route decision
- processing signal
- focus rings

Use blue only for technical information and neutral model/data states.

Use green only for success.

Use yellow only for warnings or degraded states.

Use red only for errors or destructive actions.

Do not decorate ordinary cards with colored borders.

Do not use multiple bright colors in one component unless the colors communicate distinct states.

### 7.3 Typography

Remove the editorial serif feeling.

Update the display stack to a modern geometric sans stack using available fonts:

```css
--font-display: "Inter", "Geist", ui-sans-serif, system-ui, sans-serif;
--font-sans: "Inter", "Geist", ui-sans-serif, system-ui, sans-serif;
--font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
```

Display typography should use:

- font weight 550–700
- tight tracking
- compact line height
- no italics for emphasis

Suggested headline behavior:

```css
font-weight: 650;
letter-spacing: -0.045em;
line-height: 0.95;
```

Use monospace only for:

- stage numbers
- timestamps
- route metadata
- model names
- status labels
- counts
- configuration values

Do not use monospace for normal paragraphs.

### 7.4 Radius system

Reduce the feeling that every object is a pill or rounded card.

Suggested system:

```text
small controls: 6px
buttons: 8px or full pill only for primary CTA
panels: 12px
large hero surface: 16px
```

Avoid stacking many rounded containers inside each other.

### 7.5 Borders and depth

Use thin, low-contrast borders.

Default panel:

```css
background: rgba(13, 18, 26, 0.82);
border: 1px solid rgba(255, 255, 255, 0.08);
box-shadow:
  inset 0 1px 0 rgba(255, 255, 255, 0.025),
  0 20px 60px rgba(0, 0, 0, 0.22);
backdrop-filter: blur(18px);
```

Do not add a shadow to every component.

Use shadow only for:

- major upload surface
- floating header
- sticky Lab rail
- modal/popover surfaces

### 7.6 Background treatment

Use a restrained grid and one radial light source.

Suggested utilities:

```css
.signal-grid {
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  background-size: 48px 48px;
}

.signal-glow {
  background:
    radial-gradient(circle at 50% 0%, rgba(255,107,61,.11), transparent 38%),
    radial-gradient(circle at 86% 28%, rgba(80,130,255,.06), transparent 30%);
}
```

The grid must be subtle. It should disappear behind content and never reduce text contrast.

Do not add particles, stars, moving blobs, or looping decorative animations.

---

## 8. Spacing and layout

Use a consistent page shell:

```text
maximum width: 1440px
main horizontal padding: 24px mobile, 32px tablet, 40px desktop
header height: approximately 64px
content top spacing: 40–64px
section spacing: 72–120px only on Landing
```

Avoid full-width text blocks.

Suggested readable widths:

```text
hero heading: 850px maximum
body copy: 560px maximum
technical description: 680px maximum
```

Use large areas of negative space.

The product should feel intentionally sparse rather than unfinished.

---

## 9. Header redesign

Target file:

```text
web/src/components/AppHeader.tsx
```

### Requirements

- one header style across all routes
- sticky at the top
- translucent dark surface
- subtle bottom border
- compact height
- visible active navigation
- responsive mobile navigation
- no route-specific switch to a separate visual world

Suggested structure:

```text
[GemmaClip mark] [GemmaClip]

Quick Caption   Gemma Lab

PURE GEMMA · SYSTEM ONLINE
```

The right-side status should be small and optional on narrow screens.

Use a tiny status dot with a text label, but do not imply real provider health unless the backend actually exposes it. A safe label is:

```text
PURE GEMMA PIPELINE
```

The logo should be geometric and simple.

Do not use a serif lowercase “g”.

Possible mark:

- six small frame bars forming a compact symbol
- a rounded square with a cut corner
- a two-line waveform/frame hybrid

Implement with CSS and text/icons only. Do not add an image dependency.

Navigation active state should use a thin inner fill or underline, not a large solid tab.

---

## 10. Landing page redesign

Target file:

```text
web/src/routes/index.tsx
```

The current page is too long for the desired minimal direction.

### New structure

Use only three major regions:

1. focused hero and upload
2. compact pipeline explanation
3. minimal footer

Remove or heavily reduce:

- the large promise strip
- the oversized “Are you a nerd?” section
- repeated stat cards
- decorative demo content that competes with the real upload action

### Hero layout

Desktop:

```text
┌──────────────────────────────────────────────────────────┐
│ small eyebrow                                            │
│                                                          │
│ Build grounded captions.                                 │
│ Inspect every decision.                                  │
│                                                          │
│ Drop a video. GemmaClip selects moments, checks audio,   │
│ builds evidence, and writes captions.                    │
│                                                          │
│ ┌──────────────────────────┐  ┌────────────────────────┐ │
│ │ upload surface           │  │ live pipeline preview  │ │
│ │ selected-file state      │  │ 01 Frames             │ │
│ │ primary actions          │  │ 02 Audio              │ │
│ └──────────────────────────┘  │ 03 Evidence           │ │
│                               │ 04 Captions           │ │
│                               └────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

Mobile:

```text
headline
copy
upload
primary action
secondary action
compact pipeline list
```

### Copy

Primary headline:

```text
Build grounded captions.
Inspect every decision.
```

Supporting copy:

```text
Drop a video. GemmaClip selects important moments, checks useful audio, builds structured evidence, and writes captions with Gemma.
```

Eyebrow:

```text
VIDEO CAPTIONING · PURE GEMMA
```

### Actions

When no file is selected:

```text
Choose a video
```

When a file is selected:

```text
Generate captions
Open in Gemma Lab
```

Both actions should use the same selected file.

Do not force the user to reselect the file to choose a product mode.

Primary action:

```text
Generate captions
```

Secondary action:

```text
Open in Gemma Lab
```

### Pipeline preview

The pipeline preview should be clean and factual:

```text
01  FRAME SELECTION
    Anchor + high-change moments

02  AUDIO CHECK
    Optional bounded evidence window

03  GROUNDED EVIDENCE
    Model route and structured facts

04  CAPTION WRITING
    Four grounded styles
```

Do not present fake live values before a run exists.

---

## 11. Upload surface redesign

Target file:

```text
web/src/components/UploadDropzone.tsx
```

The dropzone should feel like the main command surface, not a generic dashed form control.

### Default state

- one-pixel solid or segmented border instead of a thick dashed border
- subtle grid inside the surface
- large amount of empty space
- small upload icon
- strong short instruction
- supported formats below

Suggested copy:

```text
Drop video to begin
MP4 · WEBM · MOV
```

### Selected state

Show:

- filename
- file size
- file type
- “Ready” status
- replace action

Do not hide the action required to replace the file.

### Interaction

- keyboard accessible
- visible focus state
- drag-active state with ember border
- no constant glow
- no bouncing icon
- minimum touch target of 44px

---

## 12. Quick Caption redesign

Target file:

```text
web/src/routes/quick.tsx
```

Quick Caption should remain simple and should not expose technical controls.

### Empty state

Use a centered command surface with:

```text
Quick Caption
One video in. Grounded captions out.
```

Upload should be the primary visible action.

### Processing state

Use a two-column layout on desktop:

```text
left: video/file summary
right: stage progress
```

If a playable video preview is not yet available, show factual file metadata instead of a fake preview.

The processing component should display:

```text
01  Inspecting video          complete
02  Selecting moments         active
03  Checking audio            waiting
04  Building evidence         waiting
05  Writing captions          waiting
```

Use the backend stage message. Do not fabricate a numeric percentage.

The active stage may use:

- a subtle scan-line motion
- ember status marker
- low-intensity pulse

Completed stages should become quieter, not brighter.

### Result state

Desktop layout:

```text
left column, sticky:
- video player
- run summary
- route summary
- download and restart actions

right column:
- outcome notice
- caption cards
- open in Lab CTA
```

The caption text should be the dominant visual element.

Keep the “Open this run in Gemma Lab” action visible but visually secondary to the captions.

### Error state

Display:

- clear error title
- sanitized backend message
- choose another video action
- no technical stack trace
- no endless processing animation

---

## 13. Processing component redesign

Target file:

```text
web/src/components/ProcessingStatus.tsx
```

Remove support for visually prominent fake percentage progress unless a real percentage is supplied by the backend.

The default should be stage-based.

Suggested component anatomy:

```text
PROCESSING RUN

01  VIDEO       complete
02  FRAMES      active
03  AUDIO       waiting
04  EVIDENCE    waiting
05  CAPTIONS    waiting
```

Use:

- left stage number
- center label
- right state text or icon
- thin connector line
- visible `aria-live` status for the active stage

Do not use “•••” as the only indicator of activity.

Respect `prefers-reduced-motion`.

---

## 14. Gemma Lab entry redesign

Target file:

```text
web/src/routes/lab.tsx
```

The Lab entry should visually connect to the main product rather than appearing as a different website.

Suggested headline:

```text
Open the glass-box pipeline.
```

Supporting copy:

```text
Inspect frame selection, audio routing, structured evidence, and every generated caption from one stored run.
```

Use the same upload component as Landing and Quick Caption.

Add a compact capability row:

```text
FRAMES  ·  AUDIO  ·  EVIDENCE  ·  CAPTIONS
```

Do not add decorative technical claims that the app cannot prove.

---

## 15. Gemma Lab shell redesign

Target files:

```text
web/src/routes/lab.$runId.tsx
web/src/components/PipelineStepper.tsx
```

### Desktop shell

Replace the horizontal top stepper with a persistent vertical rail.

Suggested layout:

```text
┌───────────────┬──────────────────────────────────────────┐
│ pipeline rail │ stage content                            │
│               │                                          │
│ 01 Video      │                                          │
│ 02 Frames     │                                          │
│ 03 Audio      │                                          │
│ 04 Evidence   │                                          │
│ 05 Captions   │                                          │
│ 06 Compare    │                                          │
│               │                                          │
│ run metadata  │                                          │
└───────────────┴──────────────────────────────────────────┘
```

Suggested rail width:

```text
220–260px
```

The rail should:

- remain visible on desktop
- show stage number
- show stage label
- show stage state
- highlight the active route
- allow navigation to completed and available stages
- retain truthful invalidated/error states

### Mobile shell

Use a compact horizontally scrollable stage strip below the header.

Do not force the desktop rail into a narrow viewport.

### Run identity

Display a shortened run ID and current status in the rail footer.

Example:

```text
RUN nEAd…TGGb9
MODEL GENERATED
```

Do not expose private internal paths.

### Error behavior

Preserve `LabRunContent` behavior.

A failed run must display the saved pipeline error, not a processing state.

---

## 16. Standard Lab stage layout

All stage pages should use one shared visual grammar.

Recommended anatomy:

```text
eyebrow / stage number
stage title
short explanation

┌────────────────┬─────────────────────────────────┐
│ configuration  │ artifact / evidence / result    │
│ 320–360px      │ flexible                        │
└────────────────┴─────────────────────────────────┘

previous / next navigation
```

On mobile, stack configuration above artifact content.

Configuration should look quieter than the generated artifact.

Do not use a separate card around every field.

Group related controls with separators and concise labels.

Disabled actions must include a truthful explanation.

---

## 17. Video stage

Target file:

```text
web/src/routes/lab.$runId.video.tsx
```

The video player should be the dominant artifact.

Show metadata in a compact strip:

```text
18.25s   3840×2160   24 FPS   H.264   NO AUDIO
```

Use monospace labels with normal readable values.

Avoid six separate metadata cards.

Show preset/configuration in the quieter control column.

Do not imply the video can be reprocessed if the relevant action is disabled.

---

## 18. Frames stage

Target file:

```text
web/src/routes/lab.$runId.frames.tsx
```

This stage should be one of the strongest visuals in the product.

### Timeline

Show a horizontal timeline above the frame grid:

```text
0s ──●────────●────●────────●──────●── 18.25s
     anchor    change       anchor
```

Use actual frame timestamps.

Do not invent unsampled points.

### Frame grid

Desktop:

```text
3 columns × 2 rows
```

Each frame card should show:

```text
FRAME 03
09.672s
HIGH CHANGE
22.52 score
```

Image should dominate.

Metadata should be compact.

Included/selected frames should use a one-pixel ember outline.

Do not use a large glow.

### Configuration

Group:

- method
- total frames
- anchor count
- high-change count
- minimum spacing
- change sensitivity

Any rerun action that remains disabled must look disabled and explain why.

---

## 19. Audio stage

Target file:

```text
web/src/routes/lab.$runId.audio.tsx
```

The design must clearly distinguish:

- no audio stream
- audio available but no useful candidate
- selected audio candidate
- audio fallback occurred
- audio unavailable after safe fallback

Suggested artifact panel:

```text
AUDIO STATUS
Unavailable

No audio stream was detected in this video.
```

When waveform data exists, render it as a simple signal plot using existing data and CSS/SVG.

Do not invent waveform samples.

Show:

- selected time range
- RMS
- route explanation
- artifact availability

Do not present RMS as proof of speech.

---

## 20. Evidence stage

Target files:

```text
web/src/routes/lab.$runId.evidence.tsx
web/src/components/RouteDecision.tsx
web/src/components/EvidenceCard.tsx
web/src/components/RawJsonViewer.tsx
```

This is the product’s key differentiation and should feel highly intentional.

### Route summary

Place the route decision at the top.

Suggested structure:

```text
ROUTE DECISION

Google
Gemma 4 31B
VISUAL ONLY

Fireworks audio-visual inference was unavailable.
Audio was dropped safely and the run continued with six frames.
```

Use safe public labels only.

The route panel should show:

- provider
- safe model label
- modality
- fallback badge when applicable
- route reason
- audio status

### Evidence hierarchy

Use three major groups:

```text
VERIFIED
- subjects
- actions
- setting
- visible objects
- temporal progression

CAPTION DIRECTION
- caption focus
- mood
- style hooks

DO NOT CLAIM
- possible misreads
- unsupported claims
```

“Verified” should be visually strongest.

“Do not claim” should use warning styling but remain readable and calm.

### Raw JSON

Keep raw JSON behind an explicit disclosure control.

Do not show raw JSON as the default main artifact.

Do not expose prompts or hidden reasoning.

---

## 21. Caption stage and caption cards

Target files:

```text
web/src/routes/lab.$runId.captions.tsx
web/src/components/CaptionCard.tsx
web/src/components/GenerationOutcomeNotice.tsx
```

### Caption hierarchy

Caption cards should emphasize:

1. style
2. caption text
3. grounding availability
4. count metadata
5. actions

Suggested card:

```text
FORMAL                                      VALID

A person walks through an indoor room while the camera
captures the movement across six selected frames.

24 WORDS  ·  VISUAL EVIDENCE

Copy                         Inspect grounding
```

Use larger caption text:

```text
17–20px
line-height 1.55–1.7
```

Avoid heavy card headers and footers.

Use separators rather than nested background surfaces.

### Actions

- Copy remains available.
- Regenerate must be hidden or disabled when no callback/action exists.
- Disabled rerun actions must not appear clickable.
- Keep accessible button labels.

### Outcome notice

`model_generated` should be quiet and positive.

`evidence_fallback` should be clearly visible as degraded but still successful.

`deterministic_fallback` should never be presented as a successful ready result.

---

## 22. Compare stage

Target file:

```text
web/src/routes/lab.$runId.compare.tsx
```

The current implementation truthfully shows that experiment creation is deferred.

Preserve that honesty.

Redesign the empty state as a deliberate future-work panel rather than a generic blank card.

Suggested copy:

```text
No experiments stored for this run.

Comparison becomes available after interactive stage reruns are enabled.
```

Do not render mock charts or invented comparisons.

---

## 23. Shared component language

Prefer a small set of reusable patterns.

Potential components:

```text
GlassPanel
SectionLabel
StatusBadge
DataRow
MetricStrip
StageRail
StageRailItem
ArtifactHeader
InlineNotice
```

Do not create abstractions solely to reduce one line of Tailwind classes.

Create shared components only when at least two real surfaces benefit.

### Status badge style

Badges should resemble instrument labels:

```text
GOOGLE
VISUAL ONLY
MODEL GENERATED
AUDIO DROPPED
```

Use:

- uppercase
- monospace
- 10–11px
- moderate tracking
- compact padding
- subtle border

Do not make every metadata value a badge.

---

## 24. Buttons

### Primary

- off-white or ember fill
- high contrast
- compact label
- 44px minimum height for major actions
- no oversized shadow

### Secondary

- transparent or dark surface
- thin border
- clear hover state

### Ghost

- only for low-priority local actions
- must still have visible keyboard focus

### Disabled

- no hover effect
- reduced contrast while remaining legible
- `cursor-not-allowed`
- explanatory title or nearby note where useful

Do not use rounded-full for every button.

Use pill buttons only for primary hero actions or very compact mode switches.

---

## 25. Motion system

Motion should communicate state.

Allowed:

- 140–220ms opacity transitions
- subtle vertical reveal of 4–8px
- border/color interpolation
- processing scan line
- active stage pulse
- route panel reveal

Avoid:

- floating particles
- continuous background animation
- large parallax
- bouncing controls
- rotating decorative icons
- excessive staggered entrance animation

Suggested timing:

```css
--motion-fast: 140ms;
--motion-normal: 190ms;
--motion-slow: 260ms;
--motion-ease: cubic-bezier(.2,.8,.2,1);
```

Always respect:

```css
@media (prefers-reduced-motion: reduce)
```

Processing state must remain understandable with animation disabled.

---

## 26. Accessibility requirements

The redesign is incomplete unless it remains accessible.

Required:

- WCAG AA text contrast
- visible keyboard focus
- semantic headings
- keyboard-operable upload control
- 44px touch targets for important actions
- no color-only state communication
- `aria-live` for processing updates
- correct `role="alert"` for errors
- correct `role="status"` for processing
- labels for icon-only buttons
- reduced-motion support
- video player remains keyboard accessible
- horizontal mobile stage navigation remains keyboard and touch accessible

Do not remove existing accessible behavior during visual refactoring.

---

## 27. Responsive behavior

### Mobile: below 768px

- single-column layouts
- header navigation remains usable
- Lab pipeline becomes horizontal scroll strip
- no sticky side rail
- stage controls stack above artifacts
- metadata wraps cleanly
- caption cards remain full width
- large headlines reduce without clipping

### Tablet: 768–1100px

- landing hero may stack
- Quick result may stack video above captions
- Lab rail may remain compact if space allows

### Desktop: above 1100px

- persistent Lab rail
- two-column stage layout
- sticky video/result summary where useful
- wider artifact workspace

Test at approximately:

```text
390×844
768×1024
1440×900
```

---

## 28. Copy and tone

Use short, confident, factual copy.

Good:

```text
Building grounded evidence
Audio dropped safely
Six frames selected
Model-generated captions
```

Avoid:

```text
Unleash the power of AI
Revolutionary intelligence
Magical captions
Next-generation synergy
```

Do not overuse “futuristic” language inside the product.

The interface should look futuristic without saying that it is futuristic.

---

## 29. Implementation sequence

Implement in phases so visual changes remain reviewable.

### Phase 1: foundations

Modify:

```text
web/src/styles.css
```

Tasks:

- replace warm paper theme with Signal Glass dark tokens
- update typography
- add grid/glow utilities
- add motion tokens
- add shared focus behavior
- retain existing semantic Tailwind color mappings

### Phase 2: shell and navigation

Modify:

```text
web/src/components/AppHeader.tsx
web/src/components/PipelineStepper.tsx
web/src/routes/lab.$runId.tsx
```

Tasks:

- unify header
- implement desktop Lab rail
- implement mobile stage strip
- preserve stage state and routing
- preserve Lab error wrapper

### Phase 3: entry surfaces

Modify:

```text
web/src/routes/index.tsx
web/src/routes/quick.tsx
web/src/routes/lab.tsx
web/src/components/UploadDropzone.tsx
web/src/components/ProcessingStatus.tsx
web/src/components/StateViews.tsx
```

Tasks:

- simplify Landing
- improve selected-file actions
- redesign Quick empty, processing, ready, and error states
- redesign Lab entry

### Phase 4: Lab artifacts

Modify:

```text
web/src/routes/lab.$runId.video.tsx
web/src/routes/lab.$runId.frames.tsx
web/src/routes/lab.$runId.audio.tsx
web/src/routes/lab.$runId.evidence.tsx
web/src/routes/lab.$runId.captions.tsx
web/src/routes/lab.$runId.compare.tsx
```

Tasks:

- apply standard stage layout
- improve artifact hierarchy
- preserve all existing disabled-action truthfulness

### Phase 5: shared result components

Modify as needed:

```text
web/src/components/CaptionCard.tsx
web/src/components/RouteDecision.tsx
web/src/components/EvidenceCard.tsx
web/src/components/GenerationOutcomeNotice.tsx
web/src/components/ConfigSection.tsx
web/src/components/PrevNext.tsx
web/src/components/RawJsonViewer.tsx
```

Tasks:

- reduce nested card chrome
- standardize labels and status badges
- improve focus and hover states
- hide nonfunctional actions

### Phase 6: tests and polish

- update tests for changed accessible labels only when necessary
- add targeted tests for new navigation behavior
- add targeted tests for upload selected state
- preserve Lab error-state regression test
- verify mobile layouts manually
- verify reduced-motion behavior

---

## 30. State coverage checklist

Every redesigned surface must account for all real states.

### Upload

- empty
- drag active
- selected file
- upload busy
- upload error

### Run

- pending
- processing
- ready
- error

### Stage

- waiting
- active
- complete
- invalidated
- error

### Generation

- model generated
- evidence fallback / degraded
- deterministic fallback / error

### Audio

- usable
- uncertain
- silent
- unavailable
- failed
- audio fallback occurred

### Route

- Fireworks visual
- Fireworks audio-visual
- Google visual
- provider fallback
- visual-only fallback after audio removal

Do not design only the successful screenshot state.

---

## 31. Testing requirements

Run all existing checks:

```bash
python -m compileall src tests
pytest

git diff --check

cd web
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npm audit --audit-level=high
```

Frontend tests must not make live provider calls.

Add or update tests for:

1. header navigation remains present and accessible
2. upload selected state displays filename and readiness
3. Quick processing state exposes current stage accessibly
4. Lab desktop/mobile stage navigation preserves route targets
5. failed Lab run displays stored error instead of processing
6. CaptionCard does not show an active regenerate action when no handler exists
7. disabled stage actions remain disabled
8. route fallback badges remain visible when `audioFallbackOccurred` is true

Do not weaken tests merely to accommodate a new layout.

---

## 32. Performance requirements

The redesign should not materially slow the demo.

Avoid:

- large image assets
- background videos
- WebGL
- canvas decoration
- large animation libraries
- unnecessary rerenders
- expensive continuous blur over the entire viewport

Use `backdrop-filter` selectively.

Prefer CSS gradients and static utilities.

Maintain the existing frontend build process.

---

## 33. Definition of done

The redesign is complete when:

1. Landing, Quick Caption, and Gemma Lab feel like one cohesive product.
2. The visual language is futuristic and minimal without becoming neon or noisy.
3. Landing has one obvious upload path and two clear post-selection actions.
4. Quick Caption remains understandable to non-technical users.
5. Gemma Lab feels like a persistent pipeline inspection workspace.
6. Model routing and structured evidence are visually prominent.
7. Frame selection is presented as a timeline plus actual sampled frames.
8. Captions are easier to read than their surrounding metadata.
9. All processing, fallback, degraded, and error states remain truthful.
10. No backend behavior or API contract changes.
11. No mock data is introduced.
12. No secrets or hidden reasoning are exposed.
13. Mobile and desktop layouts are both usable.
14. Existing tests pass, and new critical layout/state tests are added.
15. Typecheck, lint, tests, build, and audit pass.
16. `lovable/` remains untouched.

---

## 34. Codex final report

After implementation, report:

1. branch name
2. commit SHA
3. visual-system changes
4. Landing changes
5. Quick Caption changes
6. Gemma Lab shell changes
7. each stage-page change
8. accessibility improvements
9. responsive behavior
10. tests added or updated
11. exact validation results
12. screenshots or a concise manual QA description for mobile and desktop
13. known limitations
14. confirmation that backend/API behavior was unchanged
15. confirmation that `lovable/` was untouched
16. clean worktree status

Do not claim the redesign is complete if any main route still uses the old warm paper/editorial style.
