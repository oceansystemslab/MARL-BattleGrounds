import assert from "node:assert/strict";
import test from "node:test";

import {
  chooseTooltipCandidate,
  createTooltipController,
  placeTooltip,
  registerTooltipOwner,
} from "../src/tooltip.js";

/**
 * @param {string} kind
 * @param {string} id
 * @param {number} paintOrder
 */
function candidate(kind, id, paintOrder) {
  return Object.freeze({
    descriptor: Object.freeze({
      kind,
      id,
      title: id,
      details: Object.freeze([`${id} details`]),
      anchor: "element",
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
  const element = {
    id,
    hidden: false,
    ownerDocument,
    parentElement,
    textContent: "",
    style: {
      left: "",
      pointerEvents: "",
      top: "",
      visibility: "",
      /**
       * @param {string} _name
       * @param {string | null} _value
       */
      setProperty(_name, _value) {},
    },
    getBoundingClientRect() {
      return bounds;
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

test("placement uses right-below when every quadrant is equally valid", () => {
  const placement = placeTooltip({
    anchorRect: { left: 120, top: 80, right: 160, bottom: 120 },
    pointer: { x: 140, y: 100 },
    tooltipSize: { width: 80, height: 40 },
    viewport: { left: 0, top: 0, right: 400, bottom: 300 },
  });

  assert.deepEqual(placement, {
    left: 172,
    top: 132,
    placement: "right-below",
  });
  assert.equal(Object.isFrozen(placement), true);
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
  const details = ["Exact duration: 3"];
  const descriptor = {
    kind: "status",
    id: "status-3",
    title: "Stunned",
    details,
    anchor: /** @type {const} */ ("element"),
  };

  registerTooltipOwner(
    /** @type {Element} */ (/** @type {unknown} */ (owner)),
    descriptor,
  );
  details.push("mutated after registration");

  assert.equal(attributes.get("data-tooltip-owner"), "");
  assert.equal([...attributes.values()].includes("Stunned"), false);
  assert.equal([...attributes.values()].includes("Exact duration: 3"), false);
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
  const statusDetails = ["Exact duration: 3 ticks"];

  registerTooltipOwner(range.element, {
    kind: "range-ultimate",
    id: "range-ultimate-4",
    title: "Ultimate range",
    details: ["Geometry only"],
    anchor: "pointer",
  });
  registerTooltipOwner(agent.element, {
    kind: "agent",
    id: "agent-4",
    title: "Agent id_4",
    details: ["HP 80 / 100"],
    anchor: "element",
  });
  registerTooltipOwner(status.element, {
    kind: "status",
    id: "status-4-0",
    title: "Stunned",
    details: statusDetails,
    anchor: "element",
  });
  statusDetails.push("mutation after registration");

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
    assert.equal(title.element.textContent, "Agent id_4");
    assert.equal(tooltip.attributes.get("data-tooltip-kind"), "agent");

    hitElements = [status.element, agent.element, range.element];
    dispatchPointerMove(rootTarget, 125, 85);
    assert.equal(title.element.textContent, "Stunned");
    assert.equal(details.element.textContent, "Exact duration: 3 ticks");
    assert.equal(tooltip.attributes.get("data-tooltip-kind"), "status");

    assert.throws(
      () => createTooltipController(options),
      /already has an active controller/,
    );

    rootTarget.dispatchEvent(new Event("pointerleave"));
    assert.equal(tooltip.element.hidden, true);
    assert.equal(title.element.textContent, "");
    assert.equal(details.element.textContent, "");
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

  registerTooltipOwner(owner.element, {
    kind: "legality",
    id: "movement-east",
    title: "Move east",
    details: ["Available"],
    anchor: "element",
  });
  registerTooltipOwner(hoverOwner.element, {
    kind: "agent",
    id: "agent-5",
    title: "Agent id_5",
    details: ["Hunter"],
    anchor: "element",
  });

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
    assert.equal(title.element.textContent, "Agent id_5");
    assert.equal(trigger.attributes.has("aria-describedby"), false);

    rootTarget.dispatchEvent(new Event("pointerleave"));
    assert.equal(title.element.textContent, "Move east");
    assert.equal(trigger.attributes.get("aria-describedby"), "researcher-tooltip");

    const escape = new Event("keydown");
    Object.defineProperty(escape, "key", { value: "Escape" });
    rootTarget.dispatchEvent(escape);
    assert.equal(tooltip.element.hidden, true);
    assert.equal(trigger.attributes.has("aria-describedby"), false);
  } finally {
    controller.destroy();
  }
});
