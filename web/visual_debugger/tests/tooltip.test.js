import assert from "node:assert/strict";
import test from "node:test";

import {
  chooseTooltipCandidate,
  createSemanticDescriptor,
  createTooltipController,
  placeTooltip,
  projectSemanticDescriptor,
  registerTooltipOwner,
  renderSemanticDescriptor,
} from "../src/tooltip.js";

/**
 * @param {string} kind
 * @param {string} id
 * @param {{title?: string, summary?: string, anchor?: "element" | "pointer", tone?: string, accent?: string, rows?: unknown[], sections?: unknown[]}} [overrides]
 */
function semanticDescriptor(kind, id, overrides = {}) {
  return createSemanticDescriptor({
    kind,
    id,
    title: overrides.title ?? id,
    tone: overrides.tone ?? "neutral",
    accent: overrides.accent ?? "none",
    summary: overrides.summary ?? `${id} summary`,
    rows: overrides.rows ?? [],
    sections: overrides.sections ?? [],
    metadata: { compact: true, full: true },
    anchor: overrides.anchor ?? "element",
  });
}

/**
 * @param {string} kind
 * @param {string} id
 * @param {number} paintOrder
 */
function candidate(kind, id, paintOrder) {
  return Object.freeze({
    descriptor: Object.freeze({
      ...semanticDescriptor(kind, id),
    }),
    paintOrder,
  });
}

/**
 * @param {Document} ownerDocument
 * @param {{left: number, top: number, right: number, bottom: number, width: number, height: number}} bounds
 * @param {string} [id]
 * @param {Element | null} [parentElement]
 */
function fakeElement(ownerDocument, bounds, id = "", parentElement = null) {
  /** @type {Map<string, string>} */
  const attributes = new Map();
  const style = {
    left: "",
    maxHeight: "",
    maxWidth: "",
    pointerEvents: "",
    top: "",
    visibility: "",
    /**
     * @param {string} _name
     * @param {string | null} _value
     */
    setProperty(_name, _value) {},
  };
  const element = {
    id,
    hidden: false,
    ownerDocument,
    parentElement,
    tagName: "DIV",
    textContent: "",
    style,
    getBoundingClientRect() {
      const maxWidth = Number.parseFloat(style.maxWidth);
      const maxHeight = Number.parseFloat(style.maxHeight);
      const width = Number.isFinite(maxWidth)
        ? Math.min(bounds.width, maxWidth)
        : bounds.width;
      const height = Number.isFinite(maxHeight)
        ? Math.min(bounds.height, maxHeight)
        : bounds.height;
      return {
        left: bounds.left,
        top: bounds.top,
        right: bounds.left + width,
        bottom: bounds.top + height,
        width,
        height,
      };
    },
    /** @param {string} name */
    hasAttribute(name) {
      return attributes.has(name);
    },
    /** @param {string} name */
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
    /**
     * @param {string} name
     * @param {string} value
     */
    setAttribute(name, value) {
      attributes.set(name, value);
    },
    /** @param {string} name */
    removeAttribute(name) {
      attributes.delete(name);
    },
  };
  return Object.freeze({
    attributes,
    element: /** @type {HTMLElement} */ (/** @type {unknown} */ (element)),
  });
}

/**
 * @param {EventTarget} root
 * @param {number} x
 * @param {number} y
 */
function dispatchPointerMove(root, x, y) {
  const event = new Event("pointermove");
  Object.defineProperties(event, {
    clientX: { value: x },
    clientY: { value: y },
  });
  root.dispatchEvent(event);
}

/**
 * @param {EventTarget} root
 * @param {string} type
 * @param {Element} target
 * @param {Element | null} [relatedTarget]
 */
function dispatchFocus(root, type, target, relatedTarget = null) {
  const event = new Event(type);
  Object.defineProperties(event, {
    target: { value: target },
    relatedTarget: { value: relatedTarget },
  });
  root.dispatchEvent(event);
}

/**
 * @param {EventTarget} root
 * @param {string} type
 * @param {Element} target
 * @param {string | null} [key]
 */
function dispatchAction(root, type, target, key = null) {
  const event = new Event(type, { cancelable: true });
  Object.defineProperty(event, "target", { value: target });
  if (key !== null) {
    Object.defineProperty(event, "key", { value: key });
  }
  root.dispatchEvent(event);
  return event;
}

function fakeDom() {
  /** @type {string[]} */
  const createdTags = [];
  /** @type {any} */
  const ownerDocument = {
    defaultView: { innerWidth: 800, innerHeight: 600 },
    documentElement: { clientWidth: 800, clientHeight: 600 },
    /** @param {string} tagName */
    createElement(tagName) {
      createdTags.push(tagName);
      return node(tagName);
    },
  };

  /** @param {string} tagName */
  function node(tagName) {
    const attributes = new Map();
    /** @type {any[]} */
    const children = [];
    return {
      ownerDocument,
      parentElement: null,
      tagName: tagName.toUpperCase(),
      className: "",
      textContent: "",
      children,
      getBoundingClientRect() {
        return { left: 0, top: 0, right: 10, bottom: 10, width: 10, height: 10 };
      },
      /** @param {string} name */
      hasAttribute(name) {
        return attributes.has(name);
      },
      /** @param {string} name */
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
      /** @param {string} name @param {string} value */
      setAttribute(name, value) {
        attributes.set(name, value);
      },
      /** @param {string} name */
      removeAttribute(name) {
        attributes.delete(name);
      },
      /** @param {...any} nodes */
      append(...nodes) {
        for (const child of nodes) {
          child.parentElement = this;
          children.push(child);
        }
      },
      /** @param {...any} nodes */
      replaceChildren(...nodes) {
        for (const child of children) {
          child.parentElement = null;
        }
        children.splice(0);
        this.append(...nodes);
      },
    };
  }

  /** @param {any} root */
  function textTree(root) {
    return [root.textContent, ...root.children.flatMap(textTree)]
      .filter(Boolean)
      .join("\n");
  }
  return { createdTags, node, ownerDocument, textTree };
}

/**
 * @param {{width?: number, height?: number, tooltipWidth?: number, tooltipHeight?: number}} [options]
 */
function controllerHarness(options = {}) {
  const width = options.width ?? 640;
  const height = options.height ?? 480;
  const ownerDocument = /** @type {Document} */ (
    /** @type {unknown} */ ({
      defaultView: { innerWidth: width, innerHeight: height },
      documentElement: { clientWidth: width, clientHeight: height },
    })
  );
  const rootTarget = new EventTarget();
  /** @type {Element[]} */
  let hitElements = [];
  const root = Object.assign(rootTarget, {
    ownerDocument,
    elementsFromPoint() {
      return hitElements;
    },
    /** @param {Element} element */
    contains(element) {
      return element.ownerDocument === ownerDocument;
    },
  });
  const tooltip = fakeElement(
    ownerDocument,
    {
      left: 0,
      top: 0,
      right: options.tooltipWidth ?? 100,
      bottom: options.tooltipHeight ?? 60,
      width: options.tooltipWidth ?? 100,
      height: options.tooltipHeight ?? 60,
    },
    "semantic-tooltip",
  );
  const title = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });
  const details = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });
  return {
    details,
    ownerDocument,
    root,
    rootTarget,
    /** @param {Element[]} elements */
    setHitElements(elements) {
      hitElements = elements;
    },
    title,
    tooltip,
  };
}

test("descriptor validation is exact while tone and accent values fail closed", () => {
  const raw = {
    kind: "status",
    id: "strict",
    title: "Strict",
    tone: "payload-created-tone",
    accent: "url(javascript:bad)",
    summary: "Summary",
    rows: [
      {
        label: "First",
        value: "One",
        metadata: { compact: true, full: true },
      },
    ],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "element",
  };
  const descriptor = createSemanticDescriptor(raw);
  assert.equal(descriptor.tone, "neutral");
  assert.equal(descriptor.accent, "none");
  assert.throws(
    () => createSemanticDescriptor({ ...raw, unexpected: "payload" }),
    /must contain exactly/u,
  );
  assert.throws(
    () =>
      createSemanticDescriptor({
        ...raw,
        rows: [{ ...raw.rows[0], extra: true }],
      }),
    /descriptor\.rows\[0\] must contain exactly/u,
  );
  assert.throws(
    () =>
      createSemanticDescriptor({
        ...raw,
        sections: [
          {
            title: "Section",
            summary: null,
            rows: [],
            metadata: { compact: false, full: true },
            payloadClass: "danger",
          },
        ],
      }),
    /descriptor\.sections\[0\] must contain exactly/u,
  );
  assert.throws(
    () =>
      createSemanticDescriptor({
        ...raw,
        metadata: { compact: true, full: true, hidden: false },
      }),
    /descriptor\.metadata must contain exactly/u,
  );
  for (const key of ["rows", "sections", "metadata", "anchor"]) {
    assert.throws(
      () => createSemanticDescriptor({ ...raw, [key]: undefined }),
      /must be|must contain/u,
      `${key} must reject an undefined value even when its key is present`,
    );
  }
});

test("compact and full projections preserve descriptor order without recomputation", () => {
  const descriptor = semanticDescriptor("agent", "ordered", {
    rows: [
      { label: "Both", value: "A", metadata: { compact: true, full: true } },
      { label: "Full", value: "B", metadata: { compact: false, full: true } },
      { label: "Compact", value: "C", metadata: { compact: true, full: false } },
    ],
    sections: [
      {
        title: "Full Section",
        summary: "Full only",
        rows: [
          { label: "Nested", value: "D", metadata: { compact: false, full: true } },
        ],
        metadata: { compact: false, full: true },
      },
    ],
  });
  const compact = projectSemanticDescriptor(descriptor, "compact");
  const full = projectSemanticDescriptor(descriptor, "full");
  assert.deepEqual(
    compact.rows.map((row) => row.label),
    ["Both", "Compact"],
  );
  assert.deepEqual(
    full.rows.map((row) => row.label),
    ["Both", "Full"],
  );
  assert.deepEqual(compact.sections, []);
  assert.equal(full.sections[0].rows[0].label, "Nested");
  assert.equal(Object.isFrozen(compact), true);
  assert.equal(Object.isFrozen(full.sections[0].rows), true);
  assert.deepEqual(
    descriptor.rows.map((row) => row.label),
    ["Both", "Full", "Compact"],
  );
});

test("safe renderer creates fixed hierarchy and preserves hostile strings as text", () => {
  const dom = fakeDom();
  const title = dom.node("h2");
  const details = dom.node("div");
  const attack =
    '<img src=x onerror="globalThis.pwned=true"> javascript:alert(1); } body{display:none}';
  const descriptor = semanticDescriptor("status", "injection", {
    title: attack,
    summary: attack,
    rows: [
      {
        label: "Unsafe-looking label",
        value: attack,
        metadata: { compact: true, full: true },
      },
    ],
    sections: [
      {
        title: attack,
        summary: attack,
        rows: [],
        metadata: { compact: false, full: true },
      },
    ],
  });
  renderSemanticDescriptor({
    descriptor,
    title: /** @type {HTMLElement} */ (/** @type {unknown} */ (title)),
    details: /** @type {HTMLElement} */ (/** @type {unknown} */ (details)),
    surface: "full",
  });
  assert.equal(title.textContent, attack);
  assert.equal(dom.textTree(details).includes(attack), true);
  assert.deepEqual(dom.createdTags, ["p", "dl", "dt", "dd", "section", "h3", "p"]);
  assert.equal(dom.createdTags.includes("img"), false);
  assert.equal(Object.hasOwn(details, "innerHTML"), false);
});

test("candidate arbitration prefers exact compact cues over overlapping agents", () => {
  const range = candidate("range-ultimate", "range", 0);
  const agent = candidate("agent", "agent", 1);
  const status = candidate("status", "status", 2);

  assert.equal(chooseTooltipCandidate([range, agent, status]), status);
  assert.equal(chooseTooltipCandidate([range, agent]), agent);
  assert.equal(chooseTooltipCandidate([]), null);
});

test("candidate arbitration uses paint order and stable ID within one tier", () => {
  const behind = candidate("modifier", "behind", 4);
  const topmost = candidate("status", "topmost", 1);
  const stableB = candidate("status", "stable-b", 2);
  const stableA = candidate("modifier", "stable-a", 2);

  assert.equal(chooseTooltipCandidate([behind, topmost]), topmost);
  assert.equal(chooseTooltipCandidate([stableB, stableA]), stableA);
  assert.deepEqual(
    [stableB, stableA],
    [candidate("status", "stable-b", 2), candidate("modifier", "stable-a", 2)],
  );
});

test("unknown future kinds remain deterministic and rank below known map cues", () => {
  const future = candidate("future-cue", "future", 0);
  const map = candidate("map", "map", 10);

  assert.equal(chooseTooltipCandidate([future, map]), map);
  assert.equal(chooseTooltipCandidate([future]), future);
  assert.throws(
    () =>
      chooseTooltipCandidate([
        /** @type {never} */ ({ descriptor: { kind: "agent" } }),
      ]),
    /descriptor\.id must be a non-empty string/,
  );
});

test("placement stays in the nearest legal pointer quadrant instead of drifting to an edge", () => {
  const centered = placeTooltip({
    anchorRect: { left: 120, top: 80, right: 160, bottom: 120 },
    pointer: { x: 140, y: 100 },
    tooltipSize: { width: 80, height: 40 },
    viewport: { left: 0, top: 0, right: 400, bottom: 300 },
  });
  assert.deepEqual(centered, {
    left: 172,
    top: 132,
    placement: "right-below",
  });
  assert.equal(Object.isFrozen(centered), true);

  const nearTopLeft = placeTooltip({
    anchorRect: { left: 120, top: 80, right: 160, bottom: 120 },
    pointer: { x: 123, y: 83 },
    tooltipSize: { width: 80, height: 40 },
    viewport: { left: 0, top: 0, right: 400, bottom: 300 },
  });
  assert.deepEqual(nearTopLeft, {
    left: 28,
    top: 28,
    placement: "left-above",
  });
});

test("placement flips away from the bottom-right viewport edge", () => {
  const placement = placeTooltip({
    anchorRect: { left: 260, top: 160, right: 280, bottom: 180 },
    pointer: { x: 270, y: 170 },
    tooltipSize: { width: 100, height: 60 },
    viewport: { left: 0, top: 0, right: 300, bottom: 200 },
  });

  assert.deepEqual(placement, {
    left: 148,
    top: 88,
    placement: "left-above",
  });
  assert.ok(placement.left >= 8);
  assert.ok(placement.top >= 8);
  assert.ok(placement.left + 100 <= 292);
  assert.ok(placement.top + 60 <= 192);
});

test("placement clamps to the viewport gutter without overlapping the anchor", () => {
  const anchorRect = { left: 8, top: 80, right: 28, bottom: 100 };
  const placement = placeTooltip({
    anchorRect,
    pointer: { x: 18, y: 90 },
    tooltipSize: { width: 110, height: 50 },
    viewport: { left: 0, top: 0, right: 180, bottom: 180 },
  });

  assert.equal(placement.placement, "right-below");
  assert.equal(placement.left, 40);
  assert.equal(placement.top, 112);
  assert.ok(placement.left >= anchorRect.right);
  assert.ok(placement.left + 110 <= 172);
  assert.ok(placement.top + 50 <= 172);
});

test("placement chooses owner clearance before a nominally better viewport fit", () => {
  const anchorRect = { left: 145, top: 132, right: 165, bottom: 152 };
  const placement = placeTooltip({
    anchorRect,
    pointer: { x: 155, y: 142 },
    tooltipSize: { width: 250, height: 110 },
    viewport: { left: 0, top: 0, right: 320, bottom: 240 },
  });
  const tooltipRect = {
    left: placement.left,
    top: placement.top,
    right: placement.left + 250,
    bottom: placement.top + 110,
  };
  const overlapWidth =
    Math.min(tooltipRect.right, anchorRect.right) -
    Math.max(tooltipRect.left, anchorRect.left);
  const overlapHeight =
    Math.min(tooltipRect.bottom, anchorRect.bottom) -
    Math.max(tooltipRect.top, anchorRect.top);

  assert.equal(Math.max(overlapWidth, 0) * Math.max(overlapHeight, 0), 0);
  assert.ok(placement.left >= 8);
  assert.ok(placement.top >= 8);
  assert.ok(tooltipRect.right <= 312);
  assert.ok(tooltipRect.bottom <= 232);
});

test("placement avoids the inspected agent and its local dock envelope", () => {
  const localEnvelope = { left: 105, top: 72, right: 225, bottom: 178 };
  const placement = placeTooltip({
    anchorRect: { left: 180, top: 82, right: 202, bottom: 100 },
    pointer: { x: 191, y: 91 },
    tooltipSize: { width: 138, height: 72 },
    viewport: { left: 0, top: 0, right: 420, bottom: 280 },
    protectedRects: [localEnvelope],
  });
  const tooltipRect = {
    left: placement.left,
    top: placement.top,
    right: placement.left + 138,
    bottom: placement.top + 72,
  };
  const overlapWidth =
    Math.min(tooltipRect.right, localEnvelope.right) -
    Math.max(tooltipRect.left, localEnvelope.left);
  const overlapHeight =
    Math.min(tooltipRect.bottom, localEnvelope.bottom) -
    Math.max(tooltipRect.top, localEnvelope.top);

  assert.equal(Math.max(overlapWidth, 0) * Math.max(overlapHeight, 0), 0);
  assert.ok(tooltipRect.left >= 8);
  assert.ok(tooltipRect.top >= 8);
  assert.ok(tooltipRect.right <= 412);
  assert.ok(tooltipRect.bottom <= 272);
});

test("all four owner-surface corners keep a compact card within local bounds", () => {
  const viewport = { left: 100, top: 50, right: 500, bottom: 350 };
  const anchors = [
    { left: 108, top: 58, right: 128, bottom: 78 },
    { left: 472, top: 58, right: 492, bottom: 78 },
    { left: 108, top: 322, right: 128, bottom: 342 },
    { left: 472, top: 322, right: 492, bottom: 342 },
  ];
  for (const anchorRect of anchors) {
    const placement = placeTooltip({
      anchorRect,
      pointer: {
        x: (anchorRect.left + anchorRect.right) / 2,
        y: (anchorRect.top + anchorRect.bottom) / 2,
      },
      tooltipSize: { width: 130, height: 70 },
      viewport,
    });
    assert.ok(placement.left >= 108);
    assert.ok(placement.top >= 58);
    assert.ok(placement.left + 130 <= 492);
    assert.ok(placement.top + 70 <= 342);
  }
});

test("registration stores a defensive descriptor and exposes no prose in markup", () => {
  /** @type {Map<string, string>} */
  const attributes = new Map();
  const owner = {
    ownerDocument: null,
    parentElement: null,
    getBoundingClientRect() {
      return { left: 0, top: 0, right: 10, bottom: 10 };
    },
    /** @param {string} name */
    hasAttribute(name) {
      return attributes.has(name);
    },
    /** @param {string} name */
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
    /**
     * @param {string} name
     * @param {string} value
     */
    setAttribute(name, value) {
      attributes.set(name, value);
    },
    /** @param {string} name */
    removeAttribute(name) {
      attributes.delete(name);
    },
  };
  const rows = [
    {
      label: "Duration",
      value: "3 Ticks",
      metadata: { compact: true, full: true },
    },
  ];
  const descriptor = {
    kind: "status",
    id: "status-3",
    title: "Stunned",
    tone: "warning",
    accent: "warrior",
    summary: "Exact status duration.",
    rows,
    sections: [],
    metadata: { compact: true, full: true },
    anchor: /** @type {const} */ ("element"),
  };

  registerTooltipOwner(
    /** @type {Element} */ (/** @type {unknown} */ (owner)),
    descriptor,
  );
  rows.push({
    label: "Mutation",
    value: "must not appear",
    metadata: { compact: true, full: true },
  });

  assert.equal(attributes.get("data-tooltip-owner"), "");
  assert.equal([...attributes.values()].includes("Stunned"), false);
  assert.equal([...attributes.values()].includes("Exact status duration."), false);
  assert.throws(
    () =>
      registerTooltipOwner(/** @type {Element} */ (/** @type {unknown} */ (owner)), {
        ...descriptor,
        anchor: /** @type {never} */ ("screen"),
      }),
    /descriptor\.anchor/,
  );
});

test("delegated controller switches one singleton tooltip and hides on leave", () => {
  const ownerDocument = /** @type {Document} */ (
    /** @type {unknown} */ ({
      defaultView: { innerWidth: 320, innerHeight: 220 },
      documentElement: { clientWidth: 320, clientHeight: 220 },
    })
  );
  const rootTarget = new EventTarget();
  /** @type {Element[]} */
  let hitElements = [];
  const root = Object.assign(rootTarget, {
    ownerDocument,
    elementsFromPoint() {
      return hitElements;
    },
    /** @param {Element} element */
    contains(element) {
      return element.ownerDocument === ownerDocument;
    },
  });
  const tooltip = fakeElement(
    ownerDocument,
    { left: 0, top: 0, right: 90, bottom: 50, width: 90, height: 50 },
    "researcher-tooltip",
  );
  const title = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });
  const details = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });
  const range = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 300,
    bottom: 200,
    width: 300,
    height: 200,
  });
  const agent = fakeElement(ownerDocument, {
    left: 110,
    top: 80,
    right: 150,
    bottom: 120,
    width: 40,
    height: 40,
  });
  const status = fakeElement(ownerDocument, {
    left: 115,
    top: 70,
    right: 139,
    bottom: 88,
    width: 24,
    height: 18,
  });
  registerTooltipOwner(
    range.element,
    semanticDescriptor("range-ultimate", "range-ultimate-4", {
      title: "Ultimate range",
      summary: "Geometry only",
      anchor: "pointer",
    }),
  );
  registerTooltipOwner(
    agent.element,
    semanticDescriptor("agent", "agent-4", {
      title: "Agent ID arbitrary-four",
      summary: "HP 80 / 100",
    }),
  );
  registerTooltipOwner(
    status.element,
    semanticDescriptor("status", "status-4-0", {
      title: "Stunned",
      summary: "Exact duration: 3 ticks",
      tone: "warning",
      accent: "warrior",
    }),
  );

  const options = {
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (root)),
    tooltip: tooltip.element,
    title: title.element,
    details: details.element,
  };
  const controller = createTooltipController(options);
  try {
    hitElements = [range.element, agent.element];
    dispatchPointerMove(rootTarget, 125, 85);
    assert.equal(tooltip.element.hidden, false);
    assert.equal(title.element.textContent, "Agent ID arbitrary-four");
    assert.equal(tooltip.attributes.get("data-tooltip-kind"), "agent");
    assert.match(
      tooltip.attributes.get("data-tooltip-placement") ?? "",
      /^(?:right|left)-(?:below|above)$/,
    );

    hitElements = [status.element, agent.element, range.element];
    dispatchPointerMove(rootTarget, 125, 85);
    assert.equal(title.element.textContent, "Stunned");
    assert.equal(details.element.textContent, "Exact duration: 3 ticks");
    assert.equal(tooltip.attributes.get("data-tooltip-kind"), "status");
    assert.equal(tooltip.attributes.get("data-tooltip-tone"), "warning");
    assert.equal(tooltip.attributes.get("data-tooltip-accent"), "warrior");

    assert.throws(
      () => createTooltipController(options),
      /already has an active controller/,
    );

    rootTarget.dispatchEvent(new Event("pointerleave"));
    assert.equal(tooltip.element.hidden, true);
    assert.equal(title.element.textContent, "");
    assert.equal(details.element.textContent, "");
    assert.equal(tooltip.attributes.has("data-tooltip-placement"), false);
  } finally {
    controller.destroy();
  }

  const replacement = createTooltipController(options);
  replacement.destroy();
});

test("focus describes the real trigger and pointer leave restores its explanation", () => {
  const ownerDocument = /** @type {Document} */ (
    /** @type {unknown} */ ({
      defaultView: { innerWidth: 320, innerHeight: 220 },
      documentElement: { clientWidth: 320, clientHeight: 220 },
    })
  );
  const rootTarget = new EventTarget();
  /** @type {Element[]} */
  let hitElements = [];
  const root = Object.assign(rootTarget, {
    ownerDocument,
    elementsFromPoint() {
      return hitElements;
    },
    /** @param {Element} element */
    contains(element) {
      return element.ownerDocument === ownerDocument;
    },
  });
  const tooltip = fakeElement(
    ownerDocument,
    { left: 0, top: 0, right: 90, bottom: 50, width: 90, height: 50 },
    "researcher-tooltip",
  );
  const title = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });
  const details = fakeElement(ownerDocument, {
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: 0,
    height: 0,
  });
  const owner = fakeElement(ownerDocument, {
    left: 20,
    top: 20,
    right: 120,
    bottom: 70,
    width: 100,
    height: 50,
  });
  const trigger = fakeElement(
    ownerDocument,
    { left: 30, top: 30, right: 80, bottom: 55, width: 50, height: 25 },
    "focused-control",
    owner.element,
  );
  const hoverOwner = fakeElement(ownerDocument, {
    left: 170,
    top: 80,
    right: 210,
    bottom: 120,
    width: 40,
    height: 40,
  });

  registerTooltipOwner(
    owner.element,
    semanticDescriptor("legality", "movement-east", {
      title: "Move east",
      summary: "Available",
    }),
  );
  registerTooltipOwner(
    hoverOwner.element,
    semanticDescriptor("agent", "agent-5", {
      title: "Agent ID opaque-five",
      summary: "Hunter",
    }),
  );

  const controller = createTooltipController({
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (root)),
    tooltip: tooltip.element,
    title: title.element,
    details: details.element,
  });
  try {
    dispatchFocus(rootTarget, "focusin", trigger.element);
    assert.equal(title.element.textContent, "Move east");
    assert.equal(trigger.attributes.get("aria-describedby"), "researcher-tooltip");
    assert.equal(owner.attributes.has("aria-describedby"), false);

    hitElements = [hoverOwner.element];
    dispatchPointerMove(rootTarget, 190, 100);
    assert.equal(title.element.textContent, "Agent ID opaque-five");
    assert.equal(trigger.attributes.has("aria-describedby"), false);

    rootTarget.dispatchEvent(new Event("pointerleave"));
    assert.equal(title.element.textContent, "Move east");
    assert.equal(trigger.attributes.get("aria-describedby"), "researcher-tooltip");

    const escapeEvent = new Event("keydown");
    Object.defineProperty(escapeEvent, "key", { value: "Escape" });
    rootTarget.dispatchEvent(escapeEvent);
    assert.equal(tooltip.element.hidden, true);
    assert.equal(trigger.attributes.has("aria-describedby"), false);
  } finally {
    controller.destroy();
  }
});

test("delegated inspection reuses the registered descriptor and preserves native controls", () => {
  const harness = controllerHarness();
  const owner = fakeElement(harness.ownerDocument, {
    left: 100,
    top: 100,
    right: 180,
    bottom: 150,
    width: 80,
    height: 50,
  });
  Object.defineProperty(owner.element, "tagName", { value: "ARTICLE" });
  const trigger = fakeElement(
    harness.ownerDocument,
    { left: 110, top: 110, right: 140, bottom: 130, width: 30, height: 20 },
    "semantic-trigger",
    owner.element,
  );
  Object.defineProperty(trigger.element, "tagName", { value: "SPAN" });
  const nativeButton = fakeElement(
    harness.ownerDocument,
    { left: 140, top: 110, right: 170, bottom: 130, width: 30, height: 20 },
    "native-command",
    owner.element,
  );
  Object.defineProperty(nativeButton.element, "tagName", { value: "BUTTON" });
  const descriptor = semanticDescriptor("agent", "inspection", {
    title: "Agent ID inspect-me",
    summary: "Now",
  });
  registerTooltipOwner(owner.element, descriptor);
  /** @type {Array<{received: ReturnType<typeof createSemanticDescriptor>, context: Readonly<{owner: Element, trigger: Element | null}>}>} */
  const calls = [];
  const controller = createTooltipController({
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (harness.root)),
    tooltip: harness.tooltip.element,
    title: harness.title.element,
    details: harness.details.element,
    onInspect(received, context) {
      calls.push({ received, context });
    },
  });
  try {
    dispatchAction(harness.rootTarget, "click", trigger.element);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].received, descriptor);
    assert.equal(calls[0].context.owner, owner.element);
    assert.equal(calls[0].context.trigger, trigger.element);

    const enter = dispatchAction(
      harness.rootTarget,
      "keydown",
      trigger.element,
      "Enter",
    );
    const space = dispatchAction(harness.rootTarget, "keydown", trigger.element, " ");
    assert.equal(enter.defaultPrevented, true);
    assert.equal(space.defaultPrevented, true);
    assert.equal(calls.length, 3);

    dispatchAction(harness.rootTarget, "click", nativeButton.element);
    const nativeEnter = dispatchAction(
      harness.rootTarget,
      "keydown",
      nativeButton.element,
      "Enter",
    );
    assert.equal(nativeEnter.defaultPrevented, false);
    assert.equal(controller.inspect(nativeButton.element), false);
    assert.equal(calls.length, 3);
  } finally {
    controller.destroy();
  }
});

test("inspectable false disables full inspection without disabling compact hover", () => {
  const harness = controllerHarness();
  const owner = fakeElement(harness.ownerDocument, {
    left: 40,
    top: 40,
    right: 100,
    bottom: 90,
    width: 60,
    height: 50,
  });
  registerTooltipOwner(
    owner.element,
    semanticDescriptor("control", "hover-only", { summary: "Compact help" }),
    { inspectable: false },
  );
  let inspections = 0;
  const controller = createTooltipController({
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (harness.root)),
    tooltip: harness.tooltip.element,
    title: harness.title.element,
    details: harness.details.element,
    onInspect() {
      inspections += 1;
    },
  });
  try {
    harness.setHitElements([owner.element]);
    dispatchPointerMove(harness.rootTarget, 60, 60);
    assert.equal(harness.tooltip.element.hidden, false);
    assert.equal(controller.inspect(owner.element), false);
    dispatchAction(harness.rootTarget, "click", owner.element);
    assert.equal(inspections, 0);
  } finally {
    controller.destroy();
  }
});

test("explicit owner surface constrains an oversized tooltip before placement", () => {
  const harness = controllerHarness({
    width: 800,
    height: 600,
    tooltipWidth: 500,
    tooltipHeight: 400,
  });
  const surface = fakeElement(harness.ownerDocument, {
    left: 100,
    top: 50,
    right: 300,
    bottom: 250,
    width: 200,
    height: 200,
  });
  Object.defineProperty(surface.element, "tagName", { value: "SVG" });
  const owner = fakeElement(harness.ownerDocument, {
    left: 265,
    top: 215,
    right: 285,
    bottom: 235,
    width: 20,
    height: 20,
  });
  registerTooltipOwner(
    owner.element,
    semanticDescriptor("aura", "surface-bound", {
      anchor: "pointer",
      summary: "A deliberately oversized compact card",
    }),
    { surface: surface.element },
  );
  const controller = createTooltipController({
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (harness.root)),
    tooltip: harness.tooltip.element,
    title: harness.title.element,
    details: harness.details.element,
  });
  try {
    harness.setHitElements([owner.element]);
    dispatchPointerMove(harness.rootTarget, 275, 225);
    assert.equal(harness.tooltip.element.style.maxWidth, "184px");
    assert.equal(harness.tooltip.element.style.maxHeight, "155px");
    const left = Number.parseFloat(harness.tooltip.element.style.left);
    const top = Number.parseFloat(harness.tooltip.element.style.top);
    assert.ok(left >= 108);
    assert.ok(top >= 58);
    assert.ok(left + 184 <= 292);
    assert.ok(top + 155 <= 242);
  } finally {
    controller.destroy();
  }
});

test("tall local cards shrink to a clear side for pointer and focus placement", () => {
  const harness = controllerHarness({
    width: 960,
    height: 600,
    tooltipWidth: 600,
    tooltipHeight: 500,
  });
  const surface = fakeElement(harness.ownerDocument, {
    left: 20,
    top: 240,
    right: 580,
    bottom: 600,
    width: 560,
    height: 360,
  });
  Object.defineProperty(surface.element, "tagName", { value: "SVG" });
  const protectedCue = fakeElement(harness.ownerDocument, {
    left: 270,
    top: 350,
    right: 370,
    bottom: 430,
    width: 100,
    height: 80,
  });
  protectedCue.element.setAttribute("data-slot", "7");
  Object.defineProperty(protectedCue.element, "classList", {
    value: {
      contains: (/** @type {string} */ className) => className === "status-dock",
    },
  });
  Object.defineProperty(surface.element, "querySelectorAll", {
    value: () => [protectedCue.element],
  });
  const owner = fakeElement(harness.ownerDocument, {
    left: 310,
    top: 370,
    right: 330,
    bottom: 390,
    width: 20,
    height: 20,
  });
  owner.element.setAttribute("data-slot", "7");
  Object.defineProperty(owner.element, "closest", {
    value: (/** @type {string} */ selector) => {
      if (selector === "svg") {
        return surface.element;
      }
      return selector === "[data-slot]" ? owner.element : null;
    },
  });
  registerTooltipOwner(
    owner.element,
    semanticDescriptor("status-overflow", "tall-local-card", {
      summary: "A compact card whose remaining rows must scroll",
    }),
    { surface: surface.element },
  );
  const controller = createTooltipController({
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (harness.root)),
    tooltip: harness.tooltip.element,
    title: harness.title.element,
    details: harness.details.element,
  });
  try {
    harness.setHitElements([owner.element]);
    dispatchPointerMove(harness.rootTarget, 320, 380);
    const pointerPlacement = {
      left: harness.tooltip.element.style.left,
      maxHeight: harness.tooltip.element.style.maxHeight,
      maxWidth: harness.tooltip.element.style.maxWidth,
      placement: harness.tooltip.attributes.get("data-tooltip-placement"),
      top: harness.tooltip.element.style.top,
    };
    assert.deepEqual(pointerPlacement, {
      left: "28px",
      maxHeight: "150px",
      maxWidth: "544px",
      placement: "left-below",
      top: "442px",
    });
    assert.equal(harness.tooltip.element.getBoundingClientRect().height, 150);
    assert.ok(Number.parseFloat(pointerPlacement.top) >= 430);
    assert.ok(Number.parseFloat(pointerPlacement.top) + 150 <= 592);

    dispatchFocus(harness.rootTarget, "focusin", owner.element);
    assert.deepEqual(
      {
        left: harness.tooltip.element.style.left,
        maxHeight: harness.tooltip.element.style.maxHeight,
        maxWidth: harness.tooltip.element.style.maxWidth,
        placement: harness.tooltip.attributes.get("data-tooltip-placement"),
        top: harness.tooltip.element.style.top,
      },
      pointerPlacement,
    );
  } finally {
    controller.destroy();
  }
});

test("removing the active owner hides the singleton through delegated observation", () => {
  let removalCallback = () => {};
  let disconnected = false;
  class FakeMutationObserver {
    /** @param {() => void} callback */
    constructor(callback) {
      removalCallback = callback;
    }
    observe() {}
    disconnect() {
      disconnected = true;
    }
  }
  const harness = controllerHarness();
  const ownerWindow = /** @type {any} */ (harness.ownerDocument.defaultView);
  ownerWindow.MutationObserver = FakeMutationObserver;
  const owner = fakeElement(harness.ownerDocument, {
    left: 40,
    top: 40,
    right: 80,
    bottom: 80,
    width: 40,
    height: 40,
  });
  let connected = true;
  Object.defineProperty(owner.element, "isConnected", {
    configurable: true,
    get() {
      return connected;
    },
  });
  registerTooltipOwner(owner.element, semanticDescriptor("status", "removed"));
  const controller = createTooltipController({
    root: /** @type {HTMLElement} */ (/** @type {unknown} */ (harness.root)),
    tooltip: harness.tooltip.element,
    title: harness.title.element,
    details: harness.details.element,
  });
  try {
    harness.setHitElements([owner.element]);
    dispatchPointerMove(harness.rootTarget, 60, 60);
    assert.equal(harness.tooltip.element.hidden, false);
    connected = false;
    removalCallback();
    assert.equal(harness.tooltip.element.hidden, true);
  } finally {
    controller.destroy();
  }
  assert.equal(disconnected, true);
});
