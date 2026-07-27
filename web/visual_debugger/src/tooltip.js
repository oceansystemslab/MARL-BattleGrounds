const TOOLTIP_OWNER_ATTRIBUTE = "data-tooltip-owner";
const TOOLTIP_KIND_ATTRIBUTE = "data-tooltip-kind";
const VIEWPORT_GUTTER = 8;
const TOOLTIP_GAP = 12;

const descriptorByOwner = new WeakMap();
const controllerByTooltip = new WeakMap();

const KIND_PRIORITY = Object.freeze({
  status: 0,
  modifier: 0,
  overflow: 0,
  "status-overflow": 0,
  "modifier-overflow": 0,
  legality: 0,
  cooldown: 0,
  visibility: 0,
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
});

/** @typedef {"element" | "pointer"} TooltipAnchor */
/**
 * @typedef {{
 *   kind: string,
 *   id: string,
 *   title: string,
 *   details: ReadonlyArray<string>,
 *   anchor: TooltipAnchor,
 * }} TooltipDescriptor
 */
/**
 * @typedef {{
 *   descriptor: TooltipDescriptor,
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
 * }} TooltipControllerOptions
 */
/**
 * @typedef {{
 *   refresh: () => void,
 *   hide: () => void,
 *   destroy: () => void,
 * }} TooltipController
 */

/**
 * Register one rendered element as a tooltip owner.
 *
 * Descriptors stay in a WeakMap rather than in DOM attributes, so explanatory
 * prose is never duplicated into markup or interpreted as HTML.
 *
 * @param {Element} element
 * @param {TooltipDescriptor} descriptor
 * @returns {void}
 */
export function registerTooltipOwner(element, descriptor) {
  if (!isTooltipOwner(element)) {
    throw new TypeError("tooltip owner must be an element.");
  }
  const normalized = normalizeDescriptor(descriptor);
  descriptorByOwner.set(element, normalized);
  element.setAttribute(TOOLTIP_OWNER_ATTRIBUTE, "");
}

/**
 * Choose the highest-priority candidate without mutating the input.
 *
 * Semantic kind wins first. Within one semantic tier, the lowest paint order
 * wins because `elementsFromPoint` reports the topmost element first. Stable
 * descriptor ID is the final deterministic tie-break.
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
 * Place one fixed-position tooltip against an inspected object or pointer.
 *
 * Candidate order intentionally resolves a perfect tie toward right/below.
 * Owner avoidance is absolute: a clamped placement that covers the inspected
 * cue never wins merely because its unclamped origin fit the viewport better.
 * Remaining ties favor less overflow, then the viewport edge nearest the cue,
 * keeping large explanations out of the battlefield's visual center.
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
    const nearestViewportEdge = Math.min(
      clampedBounds.left - usableViewport.left,
      usableViewport.right - clampedBounds.right,
      clampedBounds.top - usableViewport.top,
      usableViewport.bottom - clampedBounds.bottom,
    );
    const displacement =
      Math.abs(clamped.left - candidate.left) + Math.abs(clamped.top - candidate.top);
    const score = [
      protectedOverlap,
      inspectedOverlap,
      overflow,
      nearestViewportEdge,
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
 * Create the one delegated controller for a tooltip surface.
 *
 * Pointer and focus listeners live on the supplied root. Rendered cue elements
 * only register immutable descriptors; they do not install per-node listeners.
 *
 * @param {TooltipControllerOptions} options
 * @returns {TooltipController}
 */
export function createTooltipController(options) {
  if (!isRecord(options)) {
    throw new TypeError("tooltip controller options must be an object.");
  }
  const { root, tooltip, title, details } = options;
  if (!isEventRoot(root)) {
    throw new TypeError("tooltip root must support delegated events.");
  }
  if (!isTooltipSurface(tooltip)) {
    throw new TypeError("tooltip surface must be an HTML element.");
  }
  if (!isTextSurface(title) || !isTextSurface(details)) {
    throw new TypeError("tooltip title and details must be text elements.");
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

  function hide() {
    activeOwner = null;
    clearDescribedByTrigger();
    tooltip.hidden = true;
    tooltip.style.visibility = "";
    tooltip.removeAttribute(TOOLTIP_KIND_ATTRIBUTE);
    tooltip.removeAttribute("data-tooltip-placement");
    title.textContent = "";
    details.textContent = "";
  }

  /**
   * @param {{element: Element, descriptor: TooltipDescriptor, trigger?: Element}} candidate
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
    title.textContent = descriptor.title;
    details.textContent = descriptor.details.join("\n");
    tooltip.setAttribute(TOOLTIP_KIND_ATTRIBUTE, descriptor.kind);
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
    const measured = tooltip.getBoundingClientRect();
    const placement = placeTooltip({
      anchorRect,
      pointer: anchorPoint,
      tooltipSize: {
        width: measured.width,
        height: measured.height,
      },
      viewport: viewportRectangle(tooltip.ownerDocument),
      protectedRects: localProtectedRects(candidate.element),
    });
    tooltip.style.left = `${placement.left}px`;
    tooltip.style.top = `${placement.top}px`;
    tooltip.setAttribute("data-tooltip-placement", placement.placement);
    tooltip.style.visibility = "";
  }

  /**
   * @param {TooltipPoint} pointer
   * @returns {{element: Element, descriptor: TooltipDescriptor} | null}
   */
  function candidateAtPointer(pointer) {
    const elements = elementsFromPoint(root, pointer.x, pointer.y);
    /** @type {Array<{element: Element, descriptor: TooltipDescriptor, paintOrder: number}>} */
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
   * @param {unknown} target
   * @returns {{element: Element, descriptor: TooltipDescriptor, trigger: Element} | null}
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

  /** @param {Event} event */
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

  function onPointerLeave() {
    lastPointer = null;
    refresh();
  }

  /** @param {Event} event */
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

  /** @param {Event} event */
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

  /** @param {Event} event */
  function onKeyDown(event) {
    if ("key" in event && event.key === "Escape" && !tooltip.hidden) {
      dismissedUntilInteraction = true;
      hide();
    }
  }

  function onViewportChange() {
    refresh();
  }

  const eventRoot = /** @type {EventTarget} */ (root);
  eventRoot.addEventListener("pointermove", onPointerMove, true);
  eventRoot.addEventListener("pointerleave", onPointerLeave);
  eventRoot.addEventListener("focusin", onFocusIn, true);
  eventRoot.addEventListener("focusout", onFocusOut, true);
  eventRoot.addEventListener("keydown", onKeyDown, true);
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
    eventRoot.removeEventListener("scroll", onViewportChange, true);
    ownerWindow?.removeEventListener?.("resize", onViewportChange);
    mutationObserver?.disconnect();
    hide();
    controllerByTooltip.delete(tooltip);
  }

  /** @type {TooltipController} */
  const controller = Object.freeze({ refresh, hide, destroy });
  controllerByTooltip.set(tooltip, controller);
  hide();
  return controller;
}

/**
 * @param {TooltipDescriptor} descriptor
 * @returns {TooltipDescriptor}
 */
function normalizeDescriptor(descriptor) {
  if (!isRecord(descriptor)) {
    throw new TypeError("tooltip descriptor must be an object.");
  }
  const kind = nonEmptyString(descriptor.kind, "descriptor.kind");
  const id = nonEmptyString(descriptor.id, "descriptor.id");
  const title = nonEmptyString(descriptor.title, "descriptor.title");
  if (!Array.isArray(descriptor.details)) {
    throw new TypeError("descriptor.details must be an array.");
  }
  const details = Object.freeze(
    descriptor.details.map((detail, index) =>
      nonEmptyString(detail, `descriptor.details[${index}]`),
    ),
  );
  if (descriptor.anchor !== "element" && descriptor.anchor !== "pointer") {
    throw new TypeError('descriptor.anchor must be "element" or "pointer".');
  }
  return Object.freeze({
    kind,
    id,
    title,
    details,
    anchor: descriptor.anchor,
  });
}

/**
 * @param {string} kind
 * @returns {number}
 */
function kindPriority(kind) {
  return Object.hasOwn(KIND_PRIORITY, kind)
    ? KIND_PRIORITY[/** @type {keyof typeof KIND_PRIORITY} */ (kind)]
    : Number.MAX_SAFE_INTEGER;
}

/**
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
 * @param {number} value
 * @param {number} lower
 * @param {number} upper
 * @returns {number}
 */
function clamp(value, lower, upper) {
  return Math.min(Math.max(value, lower), upper);
}

/**
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
 * Collect the inspected agent and its durable local cue envelope. This keeps a
 * compact explanation from hiding the very tactical cluster it describes.
 * Non-battlefield owners intentionally return no additional protected region.
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
 * @param {Element} element
 * @param {string} name
 * @returns {string[]}
 */
function attributeTokens(element, name) {
  return (element.getAttribute(name) ?? "").split(/\s+/u).filter(Boolean);
}

/**
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
 * @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null;
}

/**
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
 * @param {unknown} value
 * @returns {value is Element}
 */
function isTooltipOwner(value) {
  return isElementNode(value);
}

/**
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
 * @param {unknown} value
 * @returns {value is HTMLElement}
 */
function isTextSurface(value) {
  return isElementNode(value) && "textContent" in value;
}

/**
 * @param {unknown} value
 * @returns {value is {elementsFromPoint: (x: number, y: number) => Element[]}}
 */
function isPointSource(value) {
  return isRecord(value) && typeof value.elementsFromPoint === "function";
}

/**
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
