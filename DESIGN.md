---
name: GemmaClip
description: Grounded video captions with an inspectable Gemma 4 pipeline.
colors:
  background: "#07090D"
  foreground: "#F4F6F8"
  ink: "#080A0E"
  card: "#0C1017"
  popover: "#10151E"
  secondary: "#131923"
  muted: "#111720"
  accent: "#171E29"
  muted-foreground: "#8994A5"
  secondary-foreground: "#E9EDF2"
  rule: "#FFFFFF16"
  input: "#FFFFFF1C"
  ember: "#FF6B3D"
  ember-soft: "#FF6B3D1F"
  lab: "#72A7FF"
  lab-soft: "#72A7FF1C"
  success: "#54D39A"
  warning: "#F1B65A"
  danger: "#FF6F78"
typography:
  display:
    fontFamily: "Inter, Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(3rem, 7vw, 6rem)"
    fontWeight: 600
    lineHeight: 0.96
    letterSpacing: "-0.055em"
  headline:
    fontFamily: "Inter, Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.04em"
  title:
    fontFamily: "Inter, Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Inter, Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "JetBrains Mono, SFMono-Regular, Consolas, ui-monospace, monospace"
    fontSize: "11px"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0.18em"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  2xl: "16px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  2xl: "32px"
  3xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.foreground}"
    textColor: "{colors.ink}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "36px"
  button-outline:
    backgroundColor: "{colors.background}"
    textColor: "{colors.foreground}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "36px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.muted-foreground}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "0 12px"
    height: "36px"
  input:
    backgroundColor: "transparent"
    textColor: "{colors.foreground}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "0 12px"
    height: "36px"
  card:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.xl}"
    padding: "{spacing.xl}"
  navigation:
    backgroundColor: "{colors.background}"
    textColor: "{colors.foreground}"
    rounded: "0"
    padding: "12px 24px"
    height: "64px"
  upload-dropzone:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.xl}"
    padding: "{spacing.2xl}"
---

# Design System: GemmaClip

## 1. Overview

**Creative North Star: "Signal Glass"**

GemmaClip is a calm glass-box AI instrument: near-black precision surfaces, sparse sans typography, signal lines, structured evidence, and visible model decisions. The visual system serves two modes in one product world. Quick Caption removes configuration and keeps the path plain; Gemma Lab exposes the same pipeline as a legible sequence of artifacts rather than a dashboard of opaque controls.

Depth comes from tonal layering, restrained borders, a single ember action accent, and occasional cool blue for Lab-specific state. Motion is short and state-driven. Technical metadata is present but subordinate to the artifact it explains. The system rejects generic SaaS dashboards, gaming interfaces, cryptocurrency aesthetics, neon cyberpunk posters, terminal emulators, unrelated card collections, and marketing-page excess.

**Key Characteristics:**

- Near-black surfaces with high-contrast foreground text.
- Ember marks action, focus, and active pipeline state; Lab blue marks technical context.
- Inter/Geist carries the interface; JetBrains Mono carries labels and operational metadata.
- Evidence artifacts, timeline structure, and truthful status carry the visual interest.
- Corners stay compact (6–12px); softness comes from tonal layering, not oversized rounding.

## 2. Colors

The palette is restrained and operational: one warm action signal over cool charcoal surfaces, with semantic colors reserved for state.

### Primary

- **Ember action** (`{colors.ember}`): The rare, warm signal for primary actions, keyboard focus, selected pipeline stages, and attention-worthy state changes.
- **Lab blue** (`{colors.lab}`): Technical context for Gemma Lab and audio/evidence metadata; never a competing call to action.

### Secondary

- **Signal lift** (`{colors.accent}`): A raised interactive surface used for active navigation and hover states.
- **Mint success** (`{colors.success}`), **amber warning** (`{colors.warning}`), and **coral danger** (`{colors.danger}`): Semantic status only; each state must also have text or an icon.

### Neutral

- **Signal black** (`{colors.background}`): The application canvas.
- **Carbon panel** (`{colors.card}`) and **deep panel** (`{colors.popover}`): Tonal layers for content and transient surfaces.
- **Quiet layer** (`{colors.muted}`) and **slate layer** (`{colors.secondary}`): Secondary panels and controls.
- **Ice foreground** (`{colors.foreground}`) and **steel muted text** (`{colors.muted-foreground}`): Primary and supporting text. Muted text is for metadata, never essential instructions.
- **Hairline rule** (`{colors.rule}`) and **input stroke** (`{colors.input}`): Low-contrast structure and field boundaries.

### Named Rules

**The One Signal Rule.** Ember is scarce. Use it to tell the user what can be acted on or what needs attention, never to decorate a passive surface.

**The State-with-Text Rule.** Color can reinforce a state, but labels, icons, or shape must carry the meaning too.

## 3. Typography

**Display Font:** Inter, Geist, ui-sans-serif, system-ui, sans-serif
**Body Font:** Inter, Geist, ui-sans-serif, system-ui, sans-serif
**Label/Mono Font:** JetBrains Mono, SFMono-Regular, Consolas, ui-monospace, monospace

**Character:** A single clean sans keeps the product familiar and quiet. Mono is a deliberate instrument readout for route labels, stage names, run IDs, timestamps, and safe model metadata; it is not a fake terminal voice.

### Hierarchy

- **Display** (600, `clamp(3rem, 7vw, 6rem)`, 0.96): Landing headline only; the current landing implementation is unusually tight and should not be copied into new surfaces without review.
- **Headline** (600, 30px, 1.2): Section titles and result framing.
- **Title** (600, 18px, 1.3): Card titles, control groups, and important UI labels.
- **Body** (400, 16px, 1.6): Explanatory copy, capped around 65–75ch when prose is long.
- **Label** (400, 11px, 0.18em, uppercase where used): Pipeline stages, metadata, statuses, and compact operational annotations.

### Named Rules

**The Single-Family Rule.** Do not introduce a second display or body family; hierarchy comes from weight, scale, and spacing.

**The Readable-Tight Rule.** Keep the interface dense enough for inspection, but never sacrifice legibility for compressed tracking. New display styles must not go tighter than `-0.04em`.

## 4. Elevation

GemmaClip uses tonal layering first and shadows second. The default card is a dark surface with a quiet border and a small structural shadow. The `glass-panel` treatment is a focal exception for the upload surface and other places where the user is looking through a layer into an active workflow; it combines a translucent panel, a hairline, restrained blur, and ambient shadow. Do not make every container glassy.

### Shadow Vocabulary

- **Structural card shadow** (`shadow`): Small separation for standard cards and controls; it should not become a floating decoration.
- **Focal glass ambient** (`inset 0 1px 0 rgb(255 255 255 / 0.025), 0 20px 60px rgb(0 0 0 / 0.22)`): Reserved for the `glass-panel` upload and focal inspection surfaces.
- **Focus ring** (`2px solid {colors.ember}`, 3px offset): Keyboard orientation, not elevation.

### Named Rules

**The Layer-First Rule.** Establish hierarchy with background, card, and popover tones before reaching for shadow or blur.

**The Focal-Glass Rule.** Blur is a purposeful container treatment, never the default style for every panel.

## 5. Components

### Buttons

- **Shape:** Compact, gently rounded controls (6px radius); large actions are 40px high and default actions are 36px high.
- **Primary:** Ice foreground fill with ink text; use for the next clear action such as Generate captions.
- **Hover / Focus:** Hover moves to a tonal change; keyboard focus uses the ember ring with a 3px offset. Disabled controls use reduced opacity and no pointer interaction.
- **Secondary / Ghost / Tertiary:** Outline uses the input stroke over the background; ghost is transparent and gains the signal-lift surface on hover.

### Cards / Containers

- **Corner Style:** 12px for standard cards; 16px is the outer limit for a focal grouping.
- **Background:** Carbon panel for content, deep panel for popovers, and signal lift for active controls.
- **Shadow Strategy:** Use the structural shadow vocabulary for standard cards; reserve `glass-panel` for focal workflows.
- **Border:** Quiet full borders only; no colored side stripes.
- **Internal Padding:** 24px is the standard card rhythm, with 16px for compact controls and 32px–48px for the upload surface.

### Inputs / Fields

- **Style:** 36px tall, transparent or card-toned background, input stroke, 6px radius, 12px horizontal padding.
- **Focus:** A visible 1px ember ring with the shared 3px outline offset.
- **Error / Disabled:** Error uses danger plus an explanatory message; disabled fields reduce opacity and keep the not-allowed cursor.

### Navigation

- **Style:** A 64px sticky header with a dark translucent background, bottom rule, and selective backdrop blur. Labels are familiar sans text; compact mono metadata sits below or beside the product name.
- **Default / Hover / Active:** Quiet muted text by default, signal-lift hover, and foreground text on an active surface. Do not rely on hover color alone.
- **Mobile:** Collapse navigation to the existing menu control. Gemma Lab's stage navigation becomes a horizontally scrollable strip; on desktop it becomes a sticky vertical rail.

### Upload Dropzone

The primary landing action is a bordered, card-toned dropzone with a restrained signal grid, a centered film/upload mark, clear file-type guidance, and a visible selected-file state. It is an action surface, not an illustration; the file name and readiness state become the focus after selection.

### Pipeline Stepper

The Lab stepper is a persistent instrument: six labeled stages, explicit status icons, chronological connectors, and an ember active indicator. It stays legible as a compact mobile strip and expands into a desktop rail without changing stage meaning.

## 6. Do's and Don'ts

### Do:

- **Do** use near-black precision surfaces and high-contrast foreground text.
- **Do** reserve ember for actions, focus, selected stages, and attention-worthy status.
- **Do** make frames, audio status, route decisions, evidence, and captions the primary visuals.
- **Do** distinguish processing, stale, failed, degraded, model-generated, and deterministic fallback states with text and icons.
- **Do** keep corners between 6px and 16px and use the existing Inter/Geist plus JetBrains Mono pairing.
- **Do** keep motion short, state-driven, and disabled or reduced under `prefers-reduced-motion`.

### Don't:

- **Don't** make the product look like a generic SaaS dashboard, a gaming interface, a cryptocurrency website, a neon cyberpunk poster, a terminal emulator, or a collection of unrelated cards.
- **Don't** add marketing-page excess, opaque AI claims, decorative noise, or controls that imply capabilities the backend does not provide.
- **Don't** use gradient text, repeating stripe backgrounds, or decorative grid overlays as the main visual structure.
- **Don't** use colored side-stripe borders, oversized 32px+ card radii, or a border paired with a wide decorative shadow on new components.
- **Don't** treat blur as default glassmorphism; the current `glass-panel` is a focal exception.
- **Don't** copy the current landing H1's `-0.055em` tracking into new display treatments; new display type must stay at or above `-0.04em`.
- **Don't** rely on color alone for state, and never expose credentials, raw media payloads, private endpoints, or hidden reasoning.
