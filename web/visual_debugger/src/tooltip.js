/**
 * @file Own semantic help records, tooltip registration, rendering and placement.
 * Use one delegated controller per tooltip surface. Registered owners hold
 * frozen descriptions in weak maps; prose is rendered as text, never HTML.
 * This module validates display structure, not scientific authorization.
 * Callers supply facts permitted for their audience and destroy controllers
 * when their view is removed.
 */
const TOOLTIP_OWNER_ATTRIBUTE = "data-tooltip-owner";
const TOOLTIP_KIND_ATTRIBUTE = "data-tooltip-kind";
const TOOLTIP_TONE_ATTRIBUTE = "data-tooltip-tone";
const TOOLTIP_ACCENT_ATTRIBUTE = "data-tooltip-accent";
const VIEWPORT_GUTTER = 8;
const TOOLTIP_GAP = 12;

const descriptorByOwner = new WeakMap();
const surfaceByOwner = new WeakMap();
const inspectableByOwner = new WeakMap();
const controllerByTooltip = new WeakMap();
const semanticDescriptors = new WeakSet();

export const SEMANTIC_TONES = Object.freeze([
  "neutral",
  "information",
  "positive",
  "warning",
  "danger",
]);
const SEMANTIC_TONE_SET = new Set(SEMANTIC_TONES);
export const SEMANTIC_ACCENTS = Object.freeze([
  "none",
  "mage",
  "warrior",
  "hunter",
  "rogue",
  "priest",
]);
const SEMANTIC_ACCENT_SET = new Set(SEMANTIC_ACCENTS);

const KIND_PRIORITY = Object.freeze({
  status: 0,
  modifier: 0,
  overflow: 0,
  "status-overflow": 0,
  "modifier-overflow": 0,
  legality: 0,
  cooldown: 0,
  visibility: 0,
  control: 0,
  agent: 1,
  activation: 2,
  impact: 2,
  event: 2,
  "accepted-route": 2,
  "combat-route": 2,
  "pending-route": 3,
  obstacle: 4,
  range: 5,
  "range-ultimate": 5,
  "range-basic": 5,
  "range-observation": 5,
  aura: 6,
  map: 7,
  composite: 8,
});

/** @typedef {"element" | "pointer"} TooltipAnchor */
/** @typedef {"compact" | "full"} SemanticSurface */
/** @typedef {{compact: boolean, full: boolean}} SemanticMetadata */
/**
 * @typedef {{
 *   label: string,
 *   value: string,
 *   metadata: Readonly<SemanticMetadata>,
 * }} SemanticRow
 */
/**
 * @typedef {{
 *   title: string,
 *   summary: string | null,
 *   rows: ReadonlyArray<SemanticRow>,
 *   metadata: Readonly<SemanticMetadata>,
 * }} SemanticSection
 */
/**
 * @typedef {{
 *   kind: string,
 *   id: string,
 *   title: string,
 *   tone: "neutral" | "information" | "positive" | "warning" | "danger",
 *   accent: "none" | "mage" | "warrior" | "hunter" | "rogue" | "priest",
 *   summary: string,
 *   rows: ReadonlyArray<SemanticRow>,
 *   sections: ReadonlyArray<SemanticSection>,
 *   metadata: Readonly<SemanticMetadata>,
 *   anchor: TooltipAnchor,
 * }} TooltipDescriptor
 */
/**
 * @typedef {{
 *   kind: string,
 *   id: string,
 *   title: string,
 *   tone: "neutral" | "information" | "positive" | "warning" | "danger",
 *   accent: "none" | "mage" | "warrior" | "hunter" | "rogue" | "priest",
 *   summary: null,
 *   rows: ReadonlyArray<SemanticRow>,
 *   sections: ReadonlyArray<SemanticSection>,
 *   metadata: Readonly<SemanticMetadata>,
 *   anchor: TooltipAnchor,
 * }} RowOnlyTooltipDescriptor
 */
/** @typedef {TooltipDescriptor | RowOnlyTooltipDescriptor} AnyTooltipDescriptor */
/**
 * @typedef {{
 *   descriptor: AnyTooltipDescriptor,
 *   paintOrder?: number,
 *   element?: Element,
 *   trigger?: Element,
 * }} TooltipCandidate
 */
/** @typedef {{x: number, y: number}} TooltipPoint */
/**
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 * }} TooltipRectangle
 */
/** @typedef {{width: number, height: number}} TooltipSize */
/** @typedef {"right-below" | "left-below" | "right-above" | "left-above"} TooltipPlacementName */
/**
 * @typedef {{
 *   anchorRect: TooltipRectangle,
 *   pointer: TooltipPoint,
 *   tooltipSize: TooltipSize,
 *   viewport: TooltipRectangle,
 *   protectedRects?: ReadonlyArray<TooltipRectangle>,
 * }} TooltipPlacementInput
 */
/**
 * @typedef {{
 *   left: number,
 *   top: number,
 *   placement: TooltipPlacementName,
 * }} TooltipPlacement
 */
/**
 * @typedef {{
 *   root: Document | ShadowRoot | HTMLElement,
 *   tooltip: HTMLElement,
 *   title: HTMLElement,
 *   details: HTMLElement,
 *   onInspect?: (descriptor: AnyTooltipDescriptor, context: Readonly<{owner: Element, trigger: Element | null}>) => void,
 * }} TooltipControllerOptions
 */
/**
 * @typedef {{
 *   refresh: () => void,
 *   hide: () => void,
 *   inspect: (target?: unknown) => boolean,
 *   destroy: () => void,
 * }} TooltipController
 */

/**
 * Construct and recursively freeze the only semantic explanation shape that
 * leaves the browser fact layer. Payload prose remains text; it cannot select
 * a tag name, class name, data attribute, or tone outside the fixed allowlist.
 *
 * @overload
 * @param {{summary: null} & Record<string, unknown>} rawDescriptor
 * @returns {RowOnlyTooltipDescriptor}
 */
/**
 * @overload
 * @param {unknown} rawDescriptor
 * @returns {TooltipDescriptor}
 */
/**
 * Validate and freeze a semantic help record, or reuse an existing branded
 * record unchanged. Require exactly kind, id, title, tone, accent, summary,
 * rows, sections, metadata and anchor. Text fields are trimmed; unknown tone/
 * accent become neutral/none, and anchor must be element or pointer. A null
 * summary requires at least one row visible on compact or full surfaces.
 * Normalize/freeze nested rows, sections and visibility metadata. Invalid
 * structure throws TypeError. This is a display contract: accessors may be
 * read, and the brand does not certify scientific information rights.
 *
 * @param {unknown} rawDescriptor
 * @returns {AnyTooltipDescriptor}
 */
export function createSemanticDescriptor(rawDescriptor) {
  if (!isRecord(rawDescriptor)) {
    throw new TypeError("semantic descriptor must be an object.");
  }
  if (semanticDescriptors.has(rawDescriptor)) {
    return /** @type {AnyTooltipDescriptor} */ (rawDescriptor);
  }
  assertExactKeys(
    rawDescriptor,
    [
      "kind",
      "id",
      "title",
      "tone",
      "accent",
      "summary",
      "rows",
      "sections",
      "metadata",
      "anchor",
    ],
    "semantic descriptor",
  );
  const kind = nonEmptyString(rawDescriptor.kind, "descriptor.kind");
  const id = nonEmptyString(rawDescriptor.id, "descriptor.id");
  const title = nonEmptyString(rawDescriptor.title, "descriptor.title");
  const summary =
    rawDescriptor.summary === null
      ? null
      : nonEmptyString(rawDescriptor.summary, "descriptor.summary");
  const tone =
    typeof rawDescriptor.tone === "string" && SEMANTIC_TONE_SET.has(rawDescriptor.tone)
      ? /** @type {TooltipDescriptor["tone"]} */ (rawDescriptor.tone)
      : "neutral";
  const accent =
    typeof rawDescriptor.accent === "string" &&
    SEMANTIC_ACCENT_SET.has(rawDescriptor.accent)
      ? /** @type {TooltipDescriptor["accent"]} */ (rawDescriptor.accent)
      : "none";
  const anchor = rawDescriptor.anchor;
  if (anchor !== "element" && anchor !== "pointer") {
    throw new TypeError('descriptor.anchor must be "element" or "pointer".');
  }
  const metadata = normalizeMetadata(rawDescriptor.metadata, "descriptor.metadata");
  const rows = Object.freeze(
    normalizeArray(rawDescriptor.rows, "descriptor.rows").map((row, index) =>
      normalizeSemanticRow(row, `descriptor.rows[${index}]`),
    ),
  );
  const sections = Object.freeze(
    normalizeArray(rawDescriptor.sections, "descriptor.sections").map(
      (section, index) =>
        normalizeSemanticSection(section, `descriptor.sections[${index}]`),
    ),
  );
  if (
    summary === null &&
    !["compact", "full"].some((surface) =>
      hasVisibleSemanticRow(rows, sections, /** @type {SemanticSurface} */ (surface)),
    )
  ) {
    throw new TypeError(
      "descriptor.summary may be null only when at least one semantic row is visible.",
    );
  }
  const descriptor = Object.freeze({
    kind,
    id,
    title,
    tone,
    accent,
    summary,
    rows,
    sections,
    metadata,
    anchor,
  });
  semanticDescriptors.add(descriptor);
  return descriptor;
}

/**
 * Return a frozen compact/full projection with rows and sections filtered
 * by their visibility metadata, preserving order. Normalize rawDescriptor
 * first; an existing branded descriptor is reused. Reject another surface
 * or a null-summary result with no visible row using TypeError. Share
 * already frozen row values and leave the original descriptor unchanged.
 *
 * @param {unknown} rawDescriptor
 * @param {SemanticSurface} surface
 */
export function projectSemanticDescriptor(rawDescriptor, surface) {
  const descriptor = normalizeDescriptor(rawDescriptor);
  if (surface !== "compact" && surface !== "full") {
    throw new TypeError('semantic surface must be "compact" or "full".');
  }
  const rows = Object.freeze(descriptor.rows.filter((row) => row.metadata[surface]));
  const sections = Object.freeze(
    descriptor.sections
      .filter((section) => section.metadata[surface])
      .map((section) =>
        Object.freeze({
          ...section,
          rows: Object.freeze(section.rows.filter((row) => row.metadata[surface])),
        }),
      ),
  );
  if (descriptor.summary === null && !hasVisibleSemanticRow(rows, sections, surface)) {
    throw new TypeError(
      `descriptor.summary may be null only when the ${surface} projection contains at least one semantic row.`,
    );
  }
  return Object.freeze({
    kind: descriptor.kind,
    id: descriptor.id,
    title: descriptor.title,
    tone: descriptor.tone,
    accent: descriptor.accent,
    summary: descriptor.summary,
    rows,
    sections,
    metadata: descriptor.metadata,
    anchor: descriptor.anchor,
    surface,
  });
}

/**
 * Render options.descriptor into title and details and return its
 * projection. surface defaults to compact and also accepts full. Require
 * compatible text elements or throw TypeError. Replace details with fixed
 * DOM structure and textContent only; minimal DOM stand-ins receive flattened
 * text instead. This changes supplied nodes but not the descriptor, and
 * performs no visibility, focus or tooltip-registration work.
 *
 * @param {{
 *   descriptor: unknown,
 *   title: HTMLElement,
 *   details: HTMLElement,
 *   surface?: SemanticSurface,
 * }} options
 */
export function renderSemanticDescriptor(options) {
  if (!isRecord(options)) {
    throw new TypeError("semantic render options must be an object.");
  }
  const surface = options.surface ?? "compact";
  const projection = projectSemanticDescriptor(options.descriptor, surface);
  const title = options.title;
  const details = options.details;
  if (!isTextSurface(title) || !isTextSurface(details)) {
    throw new TypeError("semantic title and details must be text elements.");
  }
  title.textContent = projection.title;
  const ownerDocument = details.ownerDocument;
  if (
    !ownerDocument ||
    typeof ownerDocument.createElement !== "function" ||
    typeof details.replaceChildren !== "function"
  ) {
    details.textContent = semanticDescriptorText(projection).join("\n");
    return projection;
  }

  const nodes = [];
  if (projection.summary !== null) {
    const summary = ownerDocument.createElement("p");
    summary.className = "semantic-explanation__summary";
    summary.textContent = projection.summary;
    nodes.push(summary);
  }
  if (projection.rows.length > 0) {
    nodes.push(renderRows(ownerDocument, projection.rows));
  }
  for (const section of projection.sections) {
    const sectionNode = ownerDocument.createElement("section");
    sectionNode.className = "semantic-explanation__section";
    const heading = ownerDocument.createElement("h3");
    heading.className = "semantic-explanation__heading";
    heading.textContent = section.title;
    sectionNode.append(heading);
    if (section.summary !== null) {
      const sectionSummary = ownerDocument.createElement("p");
      sectionSummary.className = "semantic-explanation__section-summary";
      sectionSummary.textContent = section.summary;
      sectionNode.append(sectionSummary);
    }
    if (section.rows.length > 0) {
      sectionNode.append(renderRows(ownerDocument, section.rows));
    }
    nodes.push(sectionNode);
  }
  details.replaceChildren(...nodes);
  return projection;
}

/**
 * Flatten an existing projection into ordered text strings for accessible
 * help or diagnostics. Start with summary, or title when summary is null;
 * then emit rows and sections in order. Return a new array without mutation
 * or another validation pass. It does not include title when summary exists.
 *
 * @param {ReturnType<typeof projectSemanticDescriptor>} projection
 * @returns {string[]}
 */
export function semanticDescriptorText(projection) {
  const text = projection.summary === null ? [projection.title] : [projection.summary];
  for (const row of projection.rows) {
    text.push(`${row.label}: ${row.value}`);
  }
  for (const section of projection.sections) {
    text.push(section.title);
    if (section.summary !== null) {
      text.push(section.summary);
    }
    for (const row of section.rows) {
      text.push(`${row.label}: ${row.value}`);
    }
  }
  return text;
}

/**
 * Associate element with a normalized descriptor in the shared weak map.
 * options defaults to {}; surface may override placement bounds and
 * inspectable=false disables full-card inspection (default true). Add the
 * owner attribute and accessible description for interactive owners. No
 * per-owner event listener is installed. Invalid inputs throw TypeError;
 * validation is not transactional, so call with valid options before showing
 * the view. Re-registering replaces its descriptor and placement settings.
 *
 * @param {Element} element
 * @param {unknown} descriptor
 * @param {{surface?: Element, inspectable?: boolean}} [options]
 * @returns {void}
 */
export function registerTooltipOwner(element, descriptor, options = {}) {
  if (!isTooltipOwner(element)) {
    throw new TypeError("tooltip owner must be an element.");
  }
  if (!isRecord(options)) {
    throw new TypeError("tooltip owner options must be an object.");
  }
  const normalized = normalizeDescriptor(descriptor);
  descriptorByOwner.set(element, normalized);
  const tagName =
    typeof element.tagName === "string" ? element.tagName.toLowerCase() : "";
  if (
    normalized.kind === "control" ||
    normalized.kind === "composite" ||
    ["a", "button", "input", "select", "textarea", "summary"].includes(tagName) ||
    (element.getAttribute("tabindex") !== null &&
      element.getAttribute("tabindex") !== "-1")
  ) {
    const ariaDescription =
      normalized.summary === null
        ? semanticDescriptorText(projectSemanticDescriptor(normalized, "compact")).join(
            " · ",
          )
        : normalized.summary;
    element.setAttribute("aria-description", ariaDescription);
  }
  if (options.surface !== undefined) {
    if (!isTooltipOwner(options.surface)) {
      throw new TypeError("tooltip owner surface must be an element.");
    }
    surfaceByOwner.set(element, options.surface);
  } else {
    surfaceByOwner.delete(element);
  }
  inspectableByOwner.set(element, options.inspectable !== false);
  element.setAttribute(TOOLTIP_OWNER_ATTRIBUTE, "");
}

/**
 * Return the best candidate reference, or null for an empty array. Rank
 * by semantic kind, smaller paintOrder, descriptor ID and input index.
 * Omitted paintOrder uses that index. Invalid candidates/kind/ID throw
 * TypeError; negative paintOrder throws RangeError. Input is not sorted or
 * mutated. This checks ranking fields only, not the complete descriptor.
 *
 * @template {TooltipCandidate} Candidate
 * @param {ReadonlyArray<Candidate>} candidates
 * @returns {Candidate | null}
 */
export function chooseTooltipCandidate(candidates) {
  if (!Array.isArray(candidates)) {
    throw new TypeError("tooltip candidates must be an array.");
  }

  /** @type {Candidate | null} */
  let best = null;
  /** @type {[number, number, string, number] | null} */
  let bestRank = null;

  for (const [index, candidate] of candidates.entries()) {
    const uncheckedCandidate = /** @type {unknown} */ (candidate);
    if (!isRecord(uncheckedCandidate) || !isRecord(uncheckedCandidate.descriptor)) {
      throw new TypeError("each tooltip candidate must contain a descriptor.");
    }
    const kind = nonEmptyString(candidate.descriptor.kind, "descriptor.kind");
    const id = nonEmptyString(candidate.descriptor.id, "descriptor.id");
    const paintOrder =
      candidate.paintOrder === undefined
        ? index
        : nonNegativeFinite(candidate.paintOrder, "candidate.paintOrder");
    /** @type {[number, number, string, number]} */
    const rank = [kindPriority(kind), paintOrder, id, index];

    if (bestRank === null || compareRanks(rank, bestRank) < 0) {
      best = candidate;
      bestRank = rank;
    }
  }

  return best;
}

/**
 * Choose fixed-position coordinates in CSS pixels for one measured card.
 * input supplies anchorRect, pointer, tooltipSize and viewport; optional
 * protectedRects defaults to [] (non-arrays also become []). Validate finite
 * ordered rectangles and nonnegative sizes; a viewport must have positive
 * area. Type/range failures throw TypeError/RangeError.
 *
 * Try four corners with an 8px viewport gutter and 12px gap. Minimize
 * protected-area overlap first, then owner overlap, overflow, pointer
 * distance and clamping movement. Ties favor right-below. Return frozen
 * left/top/placement; oversized cards can still overflow or overlap when
 * no clear placement exists. No DOM reads or writes occur here.
 *
 * @param {TooltipPlacementInput} input
 * @returns {TooltipPlacement}
 */
export function placeTooltip(input) {
  if (!isRecord(input)) {
    throw new TypeError("tooltip placement input must be an object.");
  }
  const anchorRect = normalizeRectangle(input.anchorRect, "anchorRect");
  const pointer = normalizePoint(input.pointer, "pointer");
  const tooltipSize = normalizeSize(input.tooltipSize, "tooltipSize");
  const viewport = normalizeRectangle(input.viewport, "viewport");
  const protectedRects = Array.isArray(input.protectedRects)
    ? input.protectedRects.map((rectangle, index) =>
        normalizeRectangle(rectangle, `protectedRects[${index}]`),
      )
    : [];
  if (viewport.right <= viewport.left || viewport.bottom <= viewport.top) {
    throw new RangeError("viewport must have positive width and height.");
  }

  const horizontalGutter = Math.min(
    VIEWPORT_GUTTER,
    (viewport.right - viewport.left) / 2,
  );
  const verticalGutter = Math.min(
    VIEWPORT_GUTTER,
    (viewport.bottom - viewport.top) / 2,
  );
  const usableViewport = Object.freeze({
    left: viewport.left + horizontalGutter,
    top: viewport.top + verticalGutter,
    right: viewport.right - horizontalGutter,
    bottom: viewport.bottom - verticalGutter,
  });
  const placementEnvelope = protectedRects.reduce(
    (envelope, protectedRect) => ({
      left: Math.min(envelope.left, protectedRect.left),
      top: Math.min(envelope.top, protectedRect.top),
      right: Math.max(envelope.right, protectedRect.right),
      bottom: Math.max(envelope.bottom, protectedRect.bottom),
    }),
    anchorRect,
  );
  const right = Math.max(placementEnvelope.right, pointer.x) + TOOLTIP_GAP;
  const left =
    Math.min(placementEnvelope.left, pointer.x) - TOOLTIP_GAP - tooltipSize.width;
  const below = Math.max(placementEnvelope.bottom, pointer.y) + TOOLTIP_GAP;
  const above =
    Math.min(placementEnvelope.top, pointer.y) - TOOLTIP_GAP - tooltipSize.height;

  /** @type {ReadonlyArray<{placement: TooltipPlacementName, left: number, top: number}>} */
  const candidates = Object.freeze([
    Object.freeze({ placement: "right-below", left: right, top: below }),
    Object.freeze({ placement: "left-below", left, top: below }),
    Object.freeze({ placement: "right-above", left: right, top: above }),
    Object.freeze({ placement: "left-above", left, top: above }),
  ]);

  /** @type {TooltipPlacement | null} */
  let best = null;
  /** @type {number[] | null} */
  let bestScore = null;

  for (const [index, candidate] of candidates.entries()) {
    const rawBounds = rectangleFromPosition(candidate.left, candidate.top, tooltipSize);
    const overflow = rectangleOverflow(rawBounds, usableViewport);
    const clamped = clampPosition(
      candidate.left,
      candidate.top,
      tooltipSize,
      usableViewport,
    );
    const clampedBounds = rectangleFromPosition(clamped.left, clamped.top, tooltipSize);
    const ownerClearance = TOOLTIP_GAP / 2;
    const paddedAnchor = {
      left: anchorRect.left - ownerClearance,
      top: anchorRect.top - ownerClearance,
      right: anchorRect.right + ownerClearance,
      bottom: anchorRect.bottom + ownerClearance,
    };
    const inspectedOverlap = rectangleOverlapArea(clampedBounds, paddedAnchor);
    const protectedOverlap = protectedRects.reduce(
      (total, protectedRect) =>
        total +
        rectangleOverlapArea(clampedBounds, {
          left: protectedRect.left - ownerClearance,
          top: protectedRect.top - ownerClearance,
          right: protectedRect.right + ownerClearance,
          bottom: protectedRect.bottom + ownerClearance,
        }),
      0,
    );
    const pointerDistance = pointToRectangleDistance(pointer, clampedBounds);
    const displacement =
      Math.abs(clamped.left - candidate.left) + Math.abs(clamped.top - candidate.top);
    const score = [
      protectedOverlap,
      inspectedOverlap,
      overflow,
      pointerDistance,
      displacement,
      index,
    ];

    if (bestScore === null || compareNumericRanks(score, bestScore) < 0) {
      bestScore = score;
      best = Object.freeze({
        left: clamped.left,
        top: clamped.top,
        placement: candidate.placement,
      });
    }
  }

  if (best === null) {
    throw new Error("tooltip placement produced no candidates.");
  }
  return best;
}

/**
 * Set tooltip's maximum CSS width/height from the owning viewport and
 * measure it. If neither free axis fits, constrain the axis retaining the
 * larger card area, then measure again. Inputs are validated CSS-pixel
 * rectangles/point; return the resulting DOMRect. This mutates tooltip
 * styles and causes layout reads, but does not choose its final position.
 *
 * @param {{
 *   tooltip: HTMLElement,
 *   anchorRect: TooltipRectangle,
 *   pointer: TooltipPoint,
 *   viewport: TooltipRectangle,
 *   protectedRects: ReadonlyArray<TooltipRectangle>,
 * }} input
 * @returns {DOMRect}
 */
function measureTooltipForPlacement(input) {
  const { tooltip, anchorRect, pointer, viewport, protectedRects } = input;
  const viewportWidth = viewport.right - viewport.left;
  const viewportHeight = viewport.bottom - viewport.top;
  const horizontalGutter = Math.min(VIEWPORT_GUTTER, viewportWidth / 2);
  const verticalGutter = Math.min(VIEWPORT_GUTTER, viewportHeight / 2);
  const maxWidth = Math.max(viewportWidth - 2 * horizontalGutter, 0);
  const maxHeight = Math.max(viewportHeight - 2 * verticalGutter, 0);
  tooltip.style.maxWidth = `${maxWidth}px`;
  tooltip.style.maxHeight = `${maxHeight}px`;

  let measured = tooltip.getBoundingClientRect();
  const placementEnvelope = protectedRects.reduce(
    (envelope, protectedRect) => ({
      left: Math.min(envelope.left, protectedRect.left),
      top: Math.min(envelope.top, protectedRect.top),
      right: Math.max(envelope.right, protectedRect.right),
      bottom: Math.max(envelope.bottom, protectedRect.bottom),
    }),
    anchorRect,
  );
  const usableViewport = {
    left: viewport.left + horizontalGutter,
    top: viewport.top + verticalGutter,
    right: viewport.right - horizontalGutter,
    bottom: viewport.bottom - verticalGutter,
  };
  const horizontalClearance = Math.min(
    maxWidth,
    Math.max(
      Math.min(placementEnvelope.left, pointer.x) - TOOLTIP_GAP - usableViewport.left,
      usableViewport.right - Math.max(placementEnvelope.right, pointer.x) - TOOLTIP_GAP,
      0,
    ),
  );
  const verticalClearance = Math.min(
    maxHeight,
    Math.max(
      Math.min(placementEnvelope.top, pointer.y) - TOOLTIP_GAP - usableViewport.top,
      usableViewport.bottom -
        Math.max(placementEnvelope.bottom, pointer.y) -
        TOOLTIP_GAP,
      0,
    ),
  );

  if (measured.width > horizontalClearance && measured.height > verticalClearance) {
    const widthLimitedArea = horizontalClearance * measured.height;
    const heightLimitedArea = measured.width * verticalClearance;
    if (heightLimitedArea >= widthLimitedArea) {
      tooltip.style.maxHeight = `${verticalClearance}px`;
    } else {
      tooltip.style.maxWidth = `${horizontalClearance}px`;
    }
    measured = tooltip.getBoundingClientRect();
  }

  return measured;
}

/**
 * Create the sole delegated controller for options.tooltip. Supply root,
 * tooltip with a nonempty ID, title and details; optional onInspect receives
 * a descriptor plus owner/trigger context. Validate compatible DOM surfaces
 * and reject a second active controller with Error. Other invalid options
 * throw TypeError. Install pointer/focus/click/keyboard/scroll/resize handlers
 * and a removal observer when available, then hide the tooltip.
 *
 * Return frozen refresh, hide, inspect and destroy methods. hide only clears
 * the current card; later interaction/refresh may show it again. inspect
 * returns whether onInspect was called. destroy removes listeners, observer
 * and controller ownership and is safe to repeat. Owner registrations remain
 * weakly held. The caller owns persistent inspector rendering and disposal.
 *
 * @param {TooltipControllerOptions} options
 * @returns {TooltipController}
 */
export function createTooltipController(options) {
  if (!isRecord(options)) {
    throw new TypeError("tooltip controller options must be an object.");
  }
  const { root, tooltip, title, details } = options;
  const onInspect = options.onInspect;
  if (!isEventRoot(root)) {
    throw new TypeError("tooltip root must support delegated events.");
  }
  if (!isTooltipSurface(tooltip)) {
    throw new TypeError("tooltip surface must be an HTML element.");
  }
  if (!isTextSurface(title) || !isTextSurface(details)) {
    throw new TypeError("tooltip title and details must be text elements.");
  }
  if (onInspect !== undefined && typeof onInspect !== "function") {
    throw new TypeError("tooltip onInspect must be a function when provided.");
  }
  const tooltipId = nonEmptyString(tooltip.id, "tooltip.id");
  if (controllerByTooltip.has(tooltip)) {
    throw new Error("tooltip surface already has an active controller.");
  }

  let destroyed = false;
  /** @type {Element | null} */
  let activeOwner = null;
  /** @type {Element | null} */
  let focusedOwner = null;
  /** @type {Element | null} */
  let focusedTrigger = null;
  /** @type {TooltipPoint | null} */
  let lastPointer = null;
  /** @type {Element | null} */
  let describedByTrigger = null;
  let describedByAdded = false;
  let dismissedUntilInteraction = false;

  tooltip.style.pointerEvents = "none";

  /**
   * Attach this tooltip ID to trigger's aria-describedby token list.
   * First remove only the token previously added by this controller. null
   * clears the current link; an unchanged trigger is a no-op. Preserve other
   * ARIA tokens and remember whether this controller added the ID.
   *
   * @param {Element | null} trigger
   */
  function setDescribedByTrigger(trigger) {
    if (describedByTrigger === trigger) {
      return;
    }
    clearDescribedByTrigger();
    if (trigger === null) {
      return;
    }
    const tokens = attributeTokens(trigger, "aria-describedby");
    describedByAdded = !tokens.includes(tooltipId);
    if (describedByAdded) {
      tokens.push(tooltipId);
      trigger.setAttribute("aria-describedby", tokens.join(" "));
    }
    describedByTrigger = trigger;
  }

  /**
   * Remove this controller's added aria-describedby token and clear its
   * link state. Preserve a preexisting identical token and every unrelated
   * token. Return nothing; repeat calls are harmless.
   */
  function clearDescribedByTrigger() {
    if (describedByTrigger !== null && describedByAdded) {
      const remaining = attributeTokens(describedByTrigger, "aria-describedby").filter(
        (token) => token !== tooltipId,
      );
      if (remaining.length === 0) {
        describedByTrigger.removeAttribute("aria-describedby");
      } else {
        describedByTrigger.setAttribute("aria-describedby", remaining.join(" "));
      }
    }
    describedByTrigger = null;
    describedByAdded = false;
  }

  /**
   * Clear active-owner state, owned ARIA link, tooltip content and visual
   * attributes/size limits. Hide the card without dropping remembered pointer
   * or focused-owner state, so refresh can show it again. Does not destroy
   * listeners or unregister owners.
   */
  function hide() {
    activeOwner = null;
    clearDescribedByTrigger();
    tooltip.hidden = true;
    tooltip.style.visibility = "";
    tooltip.removeAttribute(TOOLTIP_KIND_ATTRIBUTE);
    tooltip.removeAttribute(TOOLTIP_TONE_ATTRIBUTE);
    tooltip.removeAttribute(TOOLTIP_ACCENT_ATTRIBUTE);
    tooltip.removeAttribute("data-tooltip-placement");
    tooltip.style.maxWidth = "";
    tooltip.style.maxHeight = "";
    title.textContent = "";
    if (typeof details.replaceChildren === "function") {
      details.replaceChildren();
    } else {
      details.textContent = "";
    }
  }

  /**
   * Render candidate's compact descriptor and place its card near pointer
   * or the owner center. Hide instead if owner left root. fromFocus controls
   * ARIA linkage; pointer anchoring uses a point rectangle. Read DOM geometry,
   * measure with protected regions and update tooltip styles/attributes.
   * Candidate must be registered and its descriptor valid; errors propagate.
   *
   * @param {{element: Element, descriptor: AnyTooltipDescriptor, trigger?: Element}} candidate
   * @param {TooltipPoint | null} pointer
   * @param {boolean} fromFocus
   */
  function show(candidate, pointer, fromFocus) {
    if (!ownerWithinRoot(root, candidate.element)) {
      hide();
      return;
    }
    const descriptor = candidate.descriptor;
    activeOwner = candidate.element;
    setDescribedByTrigger(fromFocus ? (candidate.trigger ?? candidate.element) : null);
    renderSemanticDescriptor({ descriptor, title, details, surface: "compact" });
    tooltip.setAttribute(TOOLTIP_KIND_ATTRIBUTE, descriptor.kind);
    tooltip.setAttribute(TOOLTIP_TONE_ATTRIBUTE, descriptor.tone);
    tooltip.setAttribute(TOOLTIP_ACCENT_ATTRIBUTE, descriptor.accent);
    tooltip.style.visibility = "hidden";
    tooltip.hidden = false;

    const ownerRect = normalizeRectangle(
      candidate.element.getBoundingClientRect(),
      "tooltip owner rectangle",
    );
    const anchorPoint =
      pointer === null
        ? Object.freeze({
            x: (ownerRect.left + ownerRect.right) / 2,
            y: (ownerRect.top + ownerRect.bottom) / 2,
          })
        : pointer;
    const anchorRect =
      descriptor.anchor === "pointer"
        ? Object.freeze({
            left: anchorPoint.x,
            top: anchorPoint.y,
            right: anchorPoint.x,
            bottom: anchorPoint.y,
          })
        : ownerRect;
    const viewport = ownerSurfaceRectangle(candidate.element, tooltip.ownerDocument);
    const protectedRects = localProtectedRects(candidate.element);
    const measured = measureTooltipForPlacement({
      tooltip,
      anchorRect,
      pointer: anchorPoint,
      viewport,
      protectedRects,
    });
    const placement = placeTooltip({
      anchorRect,
      pointer: anchorPoint,
      tooltipSize: {
        width: measured.width,
        height: measured.height,
      },
      viewport,
      protectedRects,
    });
    tooltip.style.left = `${placement.left}px`;
    tooltip.style.top = `${placement.top}px`;
    tooltip.setAttribute("data-tooltip-placement", placement.placement);
    tooltip.style.visibility = "";
  }

  /**
   * Return the highest-ranked registered owner under pointer, or null.
   * Use elementsFromPoint paint order, ignore duplicate/disconnected/outside
   * owners and reuse stored descriptors. pointer uses viewport CSS pixels.
   * This reads DOM hit testing without changing registration or focus.
   *
   * @param {TooltipPoint} pointer
   * @returns {{element: Element, descriptor: AnyTooltipDescriptor} | null}
   */
  function candidateAtPointer(pointer) {
    const elements = elementsFromPoint(root, pointer.x, pointer.y);
    /** @type {Array<{element: Element, descriptor: AnyTooltipDescriptor, paintOrder: number}>} */
    const candidates = [];
    const visited = new Set();

    for (const [paintOrder, element] of elements.entries()) {
      const owner = closestRegisteredOwner(element);
      if (owner === null || visited.has(owner) || !ownerWithinRoot(root, owner)) {
        continue;
      }
      const descriptor = descriptorByOwner.get(owner);
      if (descriptor === undefined) {
        continue;
      }
      visited.add(owner);
      candidates.push({ element: owner, descriptor, paintOrder });
    }

    return chooseTooltipCandidate(candidates);
  }

  /**
   * Resolve target or its registered ancestor inside root. Return owner,
   * stored descriptor and original element trigger, or null for invalid,
   * unregistered or disconnected targets. No nearest-by-distance fallback occurs.
   *
   * @param {unknown} target
   * @returns {{element: Element, descriptor: AnyTooltipDescriptor, trigger: Element} | null}
   */
  function candidateForTarget(target) {
    if (!isElementNode(target)) {
      return null;
    }
    const owner = closestRegisteredOwner(target);
    if (owner === null || !ownerWithinRoot(root, owner)) {
      return null;
    }
    const descriptor = descriptorByOwner.get(owner);
    return descriptor === undefined
      ? null
      : { element: owner, descriptor, trigger: target };
  }

  /**
   * Recompute the visible card after data, layout or owner changes.
   * Do nothing after destroy. Respect Escape dismissal until interaction;
   * otherwise prefer the last pointer, then a still-connected focused owner.
   * Hide if no candidate remains. May render, measure and reposition DOM.
   */
  function refresh() {
    if (destroyed) {
      return;
    }
    if (dismissedUntilInteraction) {
      hide();
      return;
    }
    if (lastPointer !== null) {
      const candidate = candidateAtPointer(lastPointer);
      if (candidate === null) {
        hide();
      } else {
        show(candidate, lastPointer, false);
      }
      return;
    }
    if (focusedOwner !== null) {
      const descriptor = descriptorByOwner.get(focusedOwner);
      if (
        descriptor !== undefined &&
        focusedTrigger !== null &&
        ownerWithinRoot(root, focusedOwner) &&
        ownerWithinRoot(root, focusedTrigger)
      ) {
        show(
          { element: focusedOwner, descriptor, trigger: focusedTrigger },
          null,
          true,
        );
        return;
      }
    }
    hide();
  }

  /**
   * Handle a delegated pointer event using finite client coordinates.
   * Clear Escape dismissal, remember the pointer and show its best owner or
   * hide when none exists. Invalid coordinates hide the card. No event default
   * is prevented and no inspection callback runs.
   *
   * @param {Event} event
   */
  function onPointerMove(event) {
    const point = eventPoint(event);
    if (point === null) {
      hide();
      return;
    }
    dismissedUntilInteraction = false;
    lastPointer = point;
    const candidate = candidateAtPointer(point);
    if (candidate === null) {
      hide();
    } else {
      show(candidate, point, false);
    }
  }

  /**
   * Forget the last pointer and refresh from any remaining focused owner.
   * This can hide or reposition the card; it does not clear focus state.
   */
  function onPointerLeave() {
    lastPointer = null;
    refresh();
  }

  /**
   * Resolve the focused target, clear Escape dismissal and show its card
   * with an ARIA link. Forget pointer state for a valid candidate. Invalid
   * targets clear remembered focus and hide. No native focus change is made.
   *
   * @param {Event} event
   */
  function onFocusIn(event) {
    const candidate = candidateForTarget(event.target);
    if (candidate === null) {
      focusedOwner = null;
      focusedTrigger = null;
      hide();
      return;
    }
    dismissedUntilInteraction = false;
    focusedOwner = candidate.element;
    focusedTrigger = candidate.trigger;
    lastPointer = null;
    show(candidate, null, true);
  }

  /**
   * Use relatedTarget to follow focus to another registered owner.
   * Show that card with focus linkage, or clear focus and hide when no
   * candidate remains. This does not call focus() or submit an action.
   *
   * @param {Event} event
   */
  function onFocusOut(event) {
    const nextTarget = "relatedTarget" in event ? event.relatedTarget : null;
    const candidate = candidateForTarget(nextTarget);
    if (candidate === null) {
      focusedOwner = null;
      focusedTrigger = null;
      hide();
      return;
    }
    focusedOwner = candidate.element;
    focusedTrigger = candidate.trigger;
    show(candidate, null, true);
  }

  /**
   * Hide a visible tooltip on Escape until a new interaction. On Enter
   * or Space, request inspection and prevent the native default only when
   * inspection succeeds. Leave native interactive controls to their own
   * actions; do not stop event propagation.
   *
   * @param {Event} event
   */
  function onKeyDown(event) {
    if ("key" in event && event.key === "Escape" && !tooltip.hidden) {
      dismissedUntilInteraction = true;
      hide();
      return;
    }
    if (
      "key" in event &&
      (event.key === "Enter" || event.key === " ") &&
      inspect(event.target)
    ) {
      event.preventDefault();
    }
  }

  /**
   * Try to inspect the event target through the shared inspect method.
   * Ignore its boolean result and leave event propagation/defaults unchanged.
   *
   * @param {Event} event
   */
  function onClick(event) {
    inspect(event.target);
  }

  /**
   * Request full-card inspection for target, defaulting to activeOwner.
   * Return false after destroy, without onInspect, for invalid/unregistered
   * owners, explicitly noninspectable owners or native interactive targets.
   * Otherwise call onInspect once with frozen owner/trigger context and return
   * true. The callback owns rendering; callback exceptions propagate.
   *
   * @param {unknown} [target]
   * @returns {boolean}
   */
  function inspect(target = activeOwner) {
    if (destroyed || onInspect === undefined || target === null) {
      return false;
    }
    const candidate = candidateForTarget(target);
    if (
      candidate === null ||
      inspectableByOwner.get(candidate.element) === false ||
      isNativeInteractiveTarget(candidate.trigger, candidate.element)
    ) {
      return false;
    }
    onInspect(
      candidate.descriptor,
      Object.freeze({ owner: candidate.element, trigger: candidate.trigger ?? null }),
    );
    return true;
  }

  /**
   * Refresh card content/placement after a delegated scroll or window
   * resize. This uses current owner geometry; no event object is required.
   */
  function onViewportChange() {
    refresh();
  }

  const eventRoot = /** @type {EventTarget} */ (root);
  eventRoot.addEventListener("pointermove", onPointerMove, true);
  eventRoot.addEventListener("pointerleave", onPointerLeave);
  eventRoot.addEventListener("focusin", onFocusIn, true);
  eventRoot.addEventListener("focusout", onFocusOut, true);
  eventRoot.addEventListener("keydown", onKeyDown, true);
  eventRoot.addEventListener("click", onClick, true);
  eventRoot.addEventListener("scroll", onViewportChange, true);

  const ownerDocument =
    "ownerDocument" in root && root.ownerDocument !== null
      ? root.ownerDocument
      : /** @type {Document} */ (root);
  const ownerWindow = ownerDocument.defaultView;
  ownerWindow?.addEventListener?.("resize", onViewportChange);

  const mutationObserver = createRemovalObserver(root, () => {
    const activeRemoved = activeOwner !== null && !ownerWithinRoot(root, activeOwner);
    const focusRemoved =
      (focusedOwner !== null && !ownerWithinRoot(root, focusedOwner)) ||
      (focusedTrigger !== null && !ownerWithinRoot(root, focusedTrigger));
    if (focusRemoved) {
      focusedOwner = null;
      focusedTrigger = null;
    }
    if (activeRemoved || focusRemoved) {
      refresh();
    }
  });

  /**
   * Remove all controller listeners, disconnect the removal observer,
   * hide the card and release tooltip ownership. Repeat calls do nothing.
   * This does not delete registered owner descriptions or remove supplied DOM.
   */
  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    eventRoot.removeEventListener("pointermove", onPointerMove, true);
    eventRoot.removeEventListener("pointerleave", onPointerLeave);
    eventRoot.removeEventListener("focusin", onFocusIn, true);
    eventRoot.removeEventListener("focusout", onFocusOut, true);
    eventRoot.removeEventListener("keydown", onKeyDown, true);
    eventRoot.removeEventListener("click", onClick, true);
    eventRoot.removeEventListener("scroll", onViewportChange, true);
    ownerWindow?.removeEventListener?.("resize", onViewportChange);
    mutationObserver?.disconnect();
    hide();
    controllerByTooltip.delete(tooltip);
  }

  /** @type {TooltipController} */
  const controller = Object.freeze({ refresh, hide, inspect, destroy });
  controllerByTooltip.set(tooltip, controller);
  hide();
  return controller;
}

/**
 * Return a known frozen descriptor unchanged, otherwise validate/create
 * one through createSemanticDescriptor. Nonobjects throw TypeError. This
 * normalizes a display record, not its scientific provenance.
 *
 * @param {unknown} descriptor
 * @returns {AnyTooltipDescriptor}
 */
function normalizeDescriptor(descriptor) {
  if (!isRecord(descriptor)) {
    throw new TypeError("tooltip descriptor must be an object.");
  }
  if (semanticDescriptors.has(descriptor)) {
    return /** @type {AnyTooltipDescriptor} */ (descriptor);
  }
  return /** @type {AnyTooltipDescriptor} */ (createSemanticDescriptor(descriptor));
}

/**
 * Return whether rows or a visible section contain a row enabled for
 * surface. Call with normalized metadata and compact/full. Section headings
 * or summaries alone do not satisfy this row-presence check.
 *
 * @param {ReadonlyArray<SemanticRow>} rows
 * @param {ReadonlyArray<SemanticSection>} sections
 * @param {SemanticSurface} surface
 */
function hasVisibleSemanticRow(rows, sections, surface) {
  return (
    rows.some((row) => row.metadata[surface]) ||
    sections.some(
      (section) =>
        section.metadata[surface] && section.rows.some((row) => row.metadata[surface]),
    )
  );
}

/**
 * Require exactly compact/full boolean fields and return a frozen copy.
 * Invalid fields throw TypeError naming name. This uses ordinary property
 * reads and enumerable-key checks, not an accessor/prototype security check.
 *
 * @param {unknown} raw
 * @param {string} name
 * @returns {Readonly<SemanticMetadata>}
 */
function normalizeMetadata(raw, name) {
  if (
    !isRecord(raw) ||
    typeof raw.compact !== "boolean" ||
    typeof raw.full !== "boolean"
  ) {
    throw new TypeError(`${name} must contain compact and full booleans.`);
  }
  assertExactKeys(raw, ["compact", "full"], name);
  return Object.freeze({ compact: raw.compact, full: raw.full });
}

/**
 * Require exactly nonempty label/value strings plus valid metadata and
 * return a frozen normalized row. Trim strings; invalid input throws
 * TypeError naming name. No DOM creation or scientific validation occurs.
 *
 * @param {unknown} raw
 * @param {string} name
 * @returns {SemanticRow}
 */
function normalizeSemanticRow(raw, name) {
  if (!isRecord(raw)) {
    throw new TypeError(`${name} must be an object.`);
  }
  assertExactKeys(raw, ["label", "value", "metadata"], name);
  return Object.freeze({
    label: nonEmptyString(raw.label, `${name}.label`),
    value: nonEmptyString(raw.value, `${name}.value`),
    metadata: normalizeMetadata(raw.metadata, `${name}.metadata`),
  });
}

/**
 * Require exactly title, summary, rows and metadata, then return a frozen
 * section with normalized/frozen rows. title must be nonempty; summary may
 * be null or undefined, otherwise must be nonempty text. Invalid fields
 * throw TypeError naming name. Preserve row order and do not mutate raw.
 *
 * @param {unknown} raw
 * @param {string} name
 * @returns {SemanticSection}
 */
function normalizeSemanticSection(raw, name) {
  if (!isRecord(raw)) {
    throw new TypeError(`${name} must be an object.`);
  }
  assertExactKeys(raw, ["title", "summary", "rows", "metadata"], name);
  const summary =
    raw.summary === undefined || raw.summary === null
      ? null
      : nonEmptyString(raw.summary, `${name}.summary`);
  return Object.freeze({
    title: nonEmptyString(raw.title, `${name}.title`),
    summary,
    rows: Object.freeze(
      normalizeArray(raw.rows, `${name}.rows`).map((row, index) =>
        normalizeSemanticRow(row, `${name}.rows[${index}]`),
      ),
    ),
    metadata: normalizeMetadata(raw.metadata, `${name}.metadata`),
  });
}

/**
 * Return value as-is when it is an array; otherwise throw TypeError
 * naming name. This checks neither dense indices nor element validity.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {unknown[]}
 */
function normalizeArray(value, name) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${name} must be an array.`);
  }
  return value;
}

/**
 * Require enumerable string keys to match expected, ignoring order.
 * Throw TypeError naming name for extra/missing keys. Do not mutate either
 * input. Symbols, inherited fields and accessors are not rejected here.
 *
 * @param {Record<string, unknown>} value
 * @param {ReadonlyArray<string>} expected
 * @param {string} name
 */
function assertExactKeys(value, expected, name) {
  const actual = Object.keys(value).sort();
  const required = [...expected].sort();
  if (
    actual.length !== required.length ||
    actual.some((key, index) => key !== required[index])
  ) {
    throw new TypeError(`${name} must contain exactly: ${expected.join(", ")}.`);
  }
}

/**
 * Create an unattached definition list for normalized rows in order.
 * Use fixed class names and textContent for label/value pairs. Return the
 * list; no markup is interpreted and inputs are unchanged.
 *
 * @param {Document} ownerDocument
 * @param {ReadonlyArray<SemanticRow>} rows
 */
function renderRows(ownerDocument, rows) {
  const list = ownerDocument.createElement("dl");
  list.className = "semantic-explanation__rows";
  for (const row of rows) {
    const term = ownerDocument.createElement("dt");
    term.className = "semantic-explanation__label";
    term.textContent = row.label;
    const value = ownerDocument.createElement("dd");
    value.className = "semantic-explanation__value";
    value.textContent = row.value;
    list.append(term, value);
  }
  return list;
}

/**
 * Return the declared numeric priority for a known semantic kind.
 * Unknown and inherited-property names receive Number.MAX_SAFE_INTEGER.
 * Smaller numbers win; no descriptor validation occurs.
 *
 * @param {string} kind
 * @returns {number}
 */
function kindPriority(kind) {
  return Object.hasOwn(KIND_PRIORITY, kind)
    ? KIND_PRIORITY[/** @type {keyof typeof KIND_PRIORITY} */ (kind)]
    : Number.MAX_SAFE_INTEGER;
}

/**
 * Compare [kind priority, paint order, stable ID, input index] tuples.
 * Return negative/zero/positive for first before/equal/after second. IDs use
 * JavaScript lexical order; no locale sorting or mutation occurs.
 *
 * @param {[number, number, string, number]} first
 * @param {[number, number, string, number]} second
 * @returns {number}
 */
function compareRanks(first, second) {
  const priorityComparison = first[0] - second[0];
  if (priorityComparison !== 0) {
    return priorityComparison;
  }
  const paintOrderComparison = first[1] - second[1];
  if (paintOrderComparison !== 0) {
    return paintOrderComparison;
  }
  const stableIdComparison = first[2] === second[2] ? 0 : first[2] < second[2] ? -1 : 1;
  return stableIdComparison === 0 ? first[3] - second[3] : stableIdComparison;
}

/**
 * Compare numeric arrays lexicographically, with shorter length winning
 * when their common prefix matches. Return a signed difference without
 * mutation. Caller supplies finite scores; this does not handle NaN specially.
 *
 * @param {number[]} first
 * @param {number[]} second
 * @returns {number}
 */
function compareNumericRanks(first, second) {
  for (let index = 0; index < Math.min(first.length, second.length); index += 1) {
    const comparison = first[index] - second[index];
    if (comparison !== 0) {
      return comparison;
    }
  }
  return first.length - second.length;
}

/**
 * Sum how far rectangle extends past each bounds edge in CSS pixels.
 * Return zero when contained. This is edge-distance cost, not overflow area;
 * inputs must already have finite ordered edges.
 *
 * @param {TooltipRectangle} rectangle
 * @param {TooltipRectangle} bounds
 * @returns {number}
 */
function rectangleOverflow(rectangle, bounds) {
  return (
    Math.max(bounds.left - rectangle.left, 0) +
    Math.max(rectangle.right - bounds.right, 0) +
    Math.max(bounds.top - rectangle.top, 0) +
    Math.max(rectangle.bottom - bounds.bottom, 0)
  );
}

/**
 * Return intersection area of two ordered rectangles in square CSS
 * pixels. Disjoint or edge-touching rectangles return zero. No validation
 * or mutation occurs.
 *
 * @param {TooltipRectangle} first
 * @param {TooltipRectangle} second
 * @returns {number}
 */
function rectangleOverlapArea(first, second) {
  const width = Math.min(first.right, second.right) - Math.max(first.left, second.left);
  const height =
    Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
  return Math.max(width, 0) * Math.max(height, 0);
}

/**
 * Return the shortest Euclidean CSS-pixel distance from point to an
 * ordered rectangle. Points inside or on the edge return zero. Caller
 * supplies finite validated geometry.
 *
 * @param {TooltipPoint} point
 * @param {TooltipRectangle} rectangle
 * @returns {number}
 */
function pointToRectangleDistance(point, rectangle) {
  const horizontal =
    point.x < rectangle.left
      ? rectangle.left - point.x
      : point.x > rectangle.right
        ? point.x - rectangle.right
        : 0;
  const vertical =
    point.y < rectangle.top
      ? rectangle.top - point.y
      : point.y > rectangle.bottom
        ? point.y - rectangle.bottom
        : 0;
  return Math.hypot(horizontal, vertical);
}

/**
 * Return a new rectangle with left/top and right/bottom derived from
 * size.width/height. Values are CSS pixels; caller validates finite
 * coordinates and nonnegative size. Inputs remain unchanged.
 *
 * @param {number} left
 * @param {number} top
 * @param {TooltipSize} size
 * @returns {TooltipRectangle}
 */
function rectangleFromPosition(left, top, size) {
  return {
    left,
    top,
    right: left + size.width,
    bottom: top + size.height,
  };
}

/**
 * Return left/top clamped so size fits viewport where possible. When a
 * card is larger than an axis, anchor it at that viewport's starting edge.
 * Inputs are validated CSS pixels; no resizing or DOM mutation occurs.
 *
 * @param {number} left
 * @param {number} top
 * @param {TooltipSize} size
 * @param {TooltipRectangle} viewport
 * @returns {{left: number, top: number}}
 */
function clampPosition(left, top, size, viewport) {
  const maxLeft = viewport.right - size.width;
  const maxTop = viewport.bottom - size.height;
  return {
    left: maxLeft < viewport.left ? viewport.left : clamp(left, viewport.left, maxLeft),
    top: maxTop < viewport.top ? viewport.top : clamp(top, viewport.top, maxTop),
  };
}

/**
 * Clamp numeric value to inclusive lower/upper. Caller provides ordered
 * finite bounds; this helper performs no validation or rounding.
 *
 * @param {number} value
 * @param {number} lower
 * @param {number} upper
 * @returns {number}
 */
function clamp(value, lower, upper) {
  return Math.min(Math.max(value, lower), upper);
}

/**
 * Require finite numeric left/top/right/bottom and return a frozen copy.
 * Zero width/height is allowed; reversed edges throw RangeError and invalid
 * fields throw TypeError naming name. Extra fields are ignored.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {TooltipRectangle}
 */
function normalizeRectangle(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  const left = finiteNumber(value.left, `${name}.left`);
  const top = finiteNumber(value.top, `${name}.top`);
  const right = finiteNumber(value.right, `${name}.right`);
  const bottom = finiteNumber(value.bottom, `${name}.bottom`);
  if (right < left || bottom < top) {
    throw new RangeError(`${name} edges must be ordered.`);
  }
  return Object.freeze({ left, top, right, bottom });
}

/**
 * Require finite numeric x/y and return a frozen copy. Invalid input
 * throws TypeError naming name; extra fields are ignored. Units remain
 * those supplied by the caller, normally viewport CSS pixels.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {TooltipPoint}
 */
function normalizePoint(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  return Object.freeze({
    x: finiteNumber(value.x, `${name}.x`),
    y: finiteNumber(value.y, `${name}.y`),
  });
}

/**
 * Require nonnegative finite width/height and return a frozen copy.
 * Zero dimensions are allowed. Invalid types throw TypeError and negative
 * values RangeError naming name. Extra fields are ignored.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {TooltipSize}
 */
function normalizeSize(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  const width = nonNegativeFinite(value.width, `${name}.width`);
  const height = nonNegativeFinite(value.height, `${name}.height`);
  return Object.freeze({ width, height });
}

/**
 * Return the document's viewport bounds at origin in CSS pixels.
 * Prefer window innerWidth/innerHeight, falling back to documentElement
 * client sizes. Nonpositive dimensions throw RangeError. This reads layout
 * without changing the document.
 *
 * @param {Document} ownerDocument
 * @returns {TooltipRectangle}
 */
function viewportRectangle(ownerDocument) {
  const view = ownerDocument.defaultView;
  const width = view?.innerWidth ?? ownerDocument.documentElement.clientWidth;
  const height = view?.innerHeight ?? ownerDocument.documentElement.clientHeight;
  if (width <= 0 || height <= 0) {
    throw new RangeError("tooltip viewport must have positive dimensions.");
  }
  return Object.freeze({ left: 0, top: 0, right: width, bottom: height });
}

/**
 * Return the visible intersection of the owner's registered surface
 * (or closest SVG) with the document viewport. If no surface or no positive
 * intersection exists, use the viewport. Read/validate DOM rectangles;
 * malformed bounds can throw. No registration or layout mutation occurs.
 *
 * @param {Element} owner
 * @param {Document} ownerDocument
 * @returns {TooltipRectangle}
 */
function ownerSurfaceRectangle(owner, ownerDocument) {
  const viewport = viewportRectangle(ownerDocument);
  const explicit = surfaceByOwner.get(owner);
  const inferred =
    explicit ?? (typeof owner.closest === "function" ? owner.closest("svg") : null);
  if (inferred === null || inferred === undefined) {
    return viewport;
  }
  const raw = normalizeRectangle(
    inferred.getBoundingClientRect(),
    "tooltip owner surface rectangle",
  );
  const intersection = Object.freeze({
    left: Math.max(raw.left, viewport.left),
    top: Math.max(raw.top, viewport.top),
    right: Math.min(raw.right, viewport.right),
    bottom: Math.min(raw.bottom, viewport.bottom),
  });
  return intersection.right > intersection.left &&
    intersection.bottom > intersection.top
    ? intersection
    : viewport;
}

/**
 * Hit-test viewport CSS coordinates through root when supported, otherwise
 * its ownerDocument. Return elements in the browser's paint order, or []
 * when neither supplies hit testing. No containment filtering occurs here.
 *
 * @param {Document | ShadowRoot | HTMLElement} root
 * @param {number} x
 * @param {number} y
 * @returns {Element[]}
 */
function elementsFromPoint(root, x, y) {
  const pointSource = isPointSource(root)
    ? root
    : isPointSource(root.ownerDocument)
      ? root.ownerDocument
      : null;
  return pointSource === null ? [] : pointSource.elementsFromPoint(x, y);
}

/**
 * Walk element and its parentElement chain to the first owner carrying
 * both the marker attribute and a registered descriptor. Return that element
 * or null. A forged marker alone is insufficient; Shadow DOM host traversal
 * is not added by this helper.
 *
 * @param {Element} element
 * @returns {Element | null}
 */
function closestRegisteredOwner(element) {
  /** @type {Element | null} */
  let current = element;
  while (current !== null) {
    if (
      current.hasAttribute(TOOLTIP_OWNER_ATTRIBUTE) &&
      descriptorByOwner.has(current)
    ) {
      return current;
    }
    current = current.parentElement;
  }
  return null;
}

/**
 * Return whether trigger-to-owner ancestor traversal encounters a native
 * interactive tag or contenteditable=true. Include owner itself. This
 * preserves native actions during inspection. Caller ensures owner is the
 * relevant ancestor; no event or DOM change occurs.
 *
 * @param {Element} trigger
 * @param {Element} owner
 * @returns {boolean}
 */
function isNativeInteractiveTarget(trigger, owner) {
  /** @type {Element | null} */
  let current = trigger;
  while (current !== null) {
    const tagName = current.tagName.toLowerCase();
    if (
      ["a", "button", "input", "select", "textarea", "summary"].includes(tagName) ||
      current.getAttribute("contenteditable") === "true"
    ) {
      return true;
    }
    if (current === owner) {
      break;
    }
    current = current.parentElement;
  }
  return false;
}

/**
 * Collect positive-area rectangles for an inspected battlefield agent
 * and its durable docks sharing data-slot. Return [] outside a usable SVG
 * context. Only the declared agent/dock classes are included. This reads
 * layout for tooltip avoidance; it does not interpret global slot identity
 * or reveal additional scientific information.
 *
 * @param {Element} owner
 * @returns {TooltipRectangle[]}
 */
function localProtectedRects(owner) {
  if (typeof owner.closest !== "function") {
    return [];
  }
  const battlefield = owner.closest("svg");
  const slotOwner = owner.closest("[data-slot]");
  const slot = slotOwner?.getAttribute("data-slot");
  if (
    battlefield === null ||
    battlefield.tagName.toLowerCase() !== "svg" ||
    slot === null
  ) {
    return [];
  }
  const protectedClasses = new Set([
    "agent",
    "status-dock",
    "cooldown-dock",
    "required-dock-fallback-dock",
    "modifier-dock",
    "legality-dock",
  ]);
  const rectangles = [];
  for (const element of battlefield.querySelectorAll("[data-slot]")) {
    if (
      element.getAttribute("data-slot") !== slot ||
      ![...protectedClasses].some((className) => element.classList.contains(className))
    ) {
      continue;
    }
    const bounds = element.getBoundingClientRect();
    if (bounds.width > 0 && bounds.height > 0) {
      rectangles.push(normalizeRectangle(bounds, "local protected rectangle"));
    }
  }
  return rectangles;
}

/**
 * Return whether owner is connected inside root. Document roots check
 * documentElement containment when available; other roots use contains.
 * Minimal document stand-ins without contains are accepted. This is a DOM
 * ownership check, not a scientific authorization check.
 *
 * @param {Document | ShadowRoot | HTMLElement} root
 * @param {Element} owner
 * @returns {boolean}
 */
function ownerWithinRoot(root, owner) {
  if ("isConnected" in owner && owner.isConnected === false) {
    return false;
  }
  if (root === owner.ownerDocument) {
    const documentElement = root.documentElement;
    return typeof documentElement.contains === "function"
      ? documentElement.contains(owner)
      : true;
  }
  return typeof root.contains === "function" && root.contains(owner);
}

/**
 * Return a frozen x/y point from finite numeric clientX/clientY, or null
 * when unavailable. Coordinates remain viewport CSS pixels; no page-offset
 * or device-pixel conversion is applied.
 *
 * @param {Event} event
 * @returns {TooltipPoint | null}
 */
function eventPoint(event) {
  if (
    !("clientX" in event) ||
    !("clientY" in event) ||
    typeof event.clientX !== "number" ||
    typeof event.clientY !== "number" ||
    !Number.isFinite(event.clientX) ||
    !Number.isFinite(event.clientY)
  ) {
    return null;
  }
  return Object.freeze({ x: event.clientX, y: event.clientY });
}

/**
 * Split element's named attribute on whitespace and remove empty tokens.
 * Return a new string array; a missing attribute yields []. Preserve token
 * order and duplicates and leave the attribute unchanged.
 *
 * @param {Element} element
 * @param {string} name
 * @returns {string[]}
 */
function attributeTokens(element, name) {
  return (element.getAttribute(name) ?? "").split(/\s+/u).filter(Boolean);
}

/**
 * Observe child-list changes throughout root and call onRemoval.
 * Document roots observe documentElement; other roots observe themselves.
 * Return the MutationObserver, or null when the owning window lacks it.
 * The callback runs for all child-list mutations, not removals alone; the
 * caller checks ownership and disconnects the observer on disposal.
 *
 * @param {Document | ShadowRoot | HTMLElement} root
 * @param {() => void} onRemoval
 * @returns {MutationObserver | null}
 */
function createRemovalObserver(root, onRemoval) {
  const ownerDocument =
    "ownerDocument" in root && root.ownerDocument !== null
      ? root.ownerDocument
      : /** @type {Document} */ (root);
  const Observer = ownerDocument.defaultView?.MutationObserver;
  if (Observer === undefined) {
    return null;
  }
  const observer = new Observer(onRemoval);
  const target =
    "documentElement" in root && root.documentElement !== null
      ? root.documentElement
      : root;
  observer.observe(target, { childList: true, subtree: true });
  return observer;
}

/**
 * Return whether value is any non-null object, including arrays. This
 * shallow guard does not establish a plain-data record or validate fields.
 *
 * @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null;
}

/**
 * Check for the attribute/geometry methods and parentElement field used
 * by tooltip owners. Accept compatible DOM stand-ins. This structural check
 * is not an instanceof check or proof of a connected element.
 *
 * @param {unknown} value
 * @returns {value is Element}
 */
function isElementNode(value) {
  return (
    isRecord(value) &&
    typeof value.hasAttribute === "function" &&
    typeof value.getAttribute === "function" &&
    typeof value.setAttribute === "function" &&
    typeof value.removeAttribute === "function" &&
    typeof value.getBoundingClientRect === "function" &&
    "parentElement" in value
  );
}

/**
 * Apply the shared element-shape check for tooltip ownership. This does
 * not mean the element has a registered descriptor or belongs to a root.
 *
 * @param {unknown} value
 * @returns {value is Element}
 */
function isTooltipOwner(value) {
  return isElementNode(value);
}

/**
 * Check that value supports event listener add/remove and exposes an
 * ownerDocument or documentElement. Accept compatible stand-ins; membership
 * and connectedness are checked separately.
 *
 * @param {unknown} value
 * @returns {value is Document | ShadowRoot | HTMLElement}
 */
function isEventRoot(value) {
  return (
    isRecord(value) &&
    typeof value.addEventListener === "function" &&
    typeof value.removeEventListener === "function" &&
    ("ownerDocument" in value || "documentElement" in value)
  );
}

/**
 * Check the element-like shape plus id, hidden, style.setProperty and
 * ownerDocument needed by a tooltip. This does not validate a nonempty ID
 * or whether another controller owns the surface.
 *
 * @param {unknown} value
 * @returns {value is HTMLElement}
 */
function isTooltipSurface(value) {
  return (
    isElementNode(value) &&
    "id" in value &&
    "hidden" in value &&
    "style" in value &&
    isRecord(value.style) &&
    typeof value.style.setProperty === "function" &&
    "ownerDocument" in value
  );
}

/**
 * Check element-like shape and a textContent property. Minimal compatible
 * DOM stand-ins are allowed; this does not check a particular HTML tag.
 *
 * @param {unknown} value
 * @returns {value is HTMLElement}
 */
function isTextSurface(value) {
  return isElementNode(value) && "textContent" in value;
}

/**
 * Return whether value exposes an elementsFromPoint function. This does
 * not call it, validate its results or establish a DOM root type.
 *
 * @param {unknown} value
 * @returns {value is {elementsFromPoint: (x: number, y: number) => Element[]}}
 */
function isPointSource(value) {
  return isRecord(value) && typeof value.elementsFromPoint === "function";
}

/**
 * Return trimmed string value when nonempty; otherwise throw TypeError
 * naming name. No markup parsing, maximum length or identity check occurs.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {string}
 */
function nonEmptyString(value, name) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return value.trim();
}

/**
 * Return a finite numeric value unchanged, or throw TypeError naming
 * name. Do not coerce strings/booleans or round values.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function finiteNumber(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be finite.`);
  }
  return value;
}

/**
 * Return a finite numeric value at least zero. Invalid types/nonfinite
 * values throw TypeError; negative numbers throw RangeError naming name.
 * Zero and fractions are accepted without rounding.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function nonNegativeFinite(value, name) {
  const finite = finiteNumber(value, name);
  if (finite < 0) {
    throw new RangeError(`${name} must be non-negative.`);
  }
  return finite;
}
