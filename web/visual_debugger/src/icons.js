const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * @typedef {"circle" | "line" | "path" | "polyline" | "rect"} IconPrimitiveTag
 */

/**
 * @typedef {{
 *   tag: IconPrimitiveTag,
 *   attributes: Readonly<Record<string, string | number>>,
 * }} IconPrimitive
 */

/**
 * @typedef {{
 *   glyphKey: string,
 *   primitives: readonly Readonly<IconPrimitive>[],
 * }} IconDefinition
 */

/**
 * @param {IconPrimitiveTag} tag
 * @param {Record<string, string | number>} attributes
 * @returns {Readonly<IconPrimitive>}
 */
function primitive(tag, attributes) {
  return Object.freeze({
    tag,
    attributes: Object.freeze(attributes),
  });
}

/**
 * @param {string} glyphKey
 * @param {IconPrimitive[]} primitives
 * @returns {Readonly<IconDefinition>}
 */
function icon(glyphKey, primitives) {
  return Object.freeze({
    glyphKey,
    primitives: Object.freeze(primitives),
  });
}

/** @type {Readonly<Record<string, Readonly<IconDefinition>>>} */
const ICONS = Object.freeze({
  "class-mage": icon("class-mage", [
    primitive("path", {
      d: "M12 2.5 14.3 9.7 21.5 12l-7.2 2.3L12 21.5l-2.3-7.2L2.5 12l7.2-2.3Z",
    }),
    primitive("circle", { cx: 12, cy: 12, r: 1.8, fill: "currentColor" }),
  ]),
  "class-warrior": icon("class-warrior", [
    primitive("path", {
      d: "M12 2.5 20 5.4v5.8c0 5-3.2 8.3-8 10.3-4.8-2-8-5.3-8-10.3V5.4Z",
    }),
    primitive("path", { d: "M12 5.8v11.7M8.3 9.4h7.4" }),
  ]),
  "class-hunter": icon("class-hunter", [
    primitive("circle", { cx: 12, cy: 12, r: 6.2 }),
    primitive("circle", { cx: 12, cy: 12, r: 2 }),
    primitive("path", { d: "M12 2v4M12 18v4M2 12h4M18 12h4" }),
  ]),
  "class-rogue": icon("class-rogue", [
    primitive("path", { d: "m5 4 6.5 6.5-2.7 2.7L2.3 6.7Z" }),
    primitive("path", { d: "m19 4-6.5 6.5 2.7 2.7 6.5-6.5Z" }),
    primitive("path", { d: "m8.8 13.2-3.1 3.1M15.2 13.2l3.1 3.1" }),
    primitive("circle", { cx: 12, cy: 17.5, r: 2.4 }),
  ]),
  "class-priest": icon("class-priest", [
    primitive("circle", { cx: 12, cy: 7, r: 4.2 }),
    primitive("path", { d: "M12 11.2V22M7.5 16h9" }),
  ]),
  "team-a": icon("team-a", [
    primitive("path", {
      d: "M12 2.8 20 6v5.3c0 4.8-3.1 8-8 9.9-4.9-1.9-8-5.1-8-9.9V6Z",
      fill: "currentColor",
    }),
  ]),
  "team-b": icon("team-b", [
    primitive("path", {
      d: "M12 2.8 20 6v5.3c0 4.8-3.1 8-8 9.9-4.9-1.9-8-5.1-8-9.9V6Z",
    }),
    primitive("path", { d: "M8 12h8" }),
  ]),
  "status-charge-stun": icon("status-charge-stun", [
    primitive("path", { d: "m12 2 8 5v10l-8 5-8-5V7Z" }),
    primitive("path", { d: "m8.5 8.5 7 7M15.5 8.5l-7 7" }),
  ]),
  "status-trap": icon("status-trap", [
    primitive("rect", { x: 4, y: 4, width: 16, height: 16, rx: 2 }),
    primitive("path", { d: "M4 9h16M4 15h16M9 4v16M15 4v16" }),
  ]),
  "status-poison-stun": icon("status-poison-stun", [
    primitive("path", {
      d: "M12 2.5c3 4.3 5.2 7.1 5.2 10.6A5.2 5.2 0 0 1 6.8 13c0-3.5 2.2-6.3 5.2-10.5Z",
    }),
    primitive("path", { d: "m8.5 16.5 7-7M8.5 9.5l7 7" }),
  ]),
  "status-slow": icon("status-slow", [
    primitive("polyline", { points: "5,7 12,14 19,7" }),
    primitive("polyline", { points: "5,12 12,19 19,12" }),
  ]),
  "status-anti-heal": icon("status-anti-heal", [
    primitive("path", {
      d: "M12 20.2 4.2 12.8A5.1 5.1 0 0 1 11.4 5l.6.7.6-.7a5.1 5.1 0 0 1 7.2 7.8Z",
    }),
    primitive("line", { x1: 4, y1: 20, x2: 20, y2: 4 }),
  ]),
  "status-freedom": icon("status-freedom", [
    primitive("path", { d: "M9.5 7.5 7 5a3 3 0 0 0-4.2 4.2l3 3" }),
    primitive("path", { d: "m14.5 16.5 2.5 2.5a3 3 0 0 0 4.2-4.2l-3-3" }),
    primitive("path", { d: "m8 16 8-8M5 19l3-3M16 8l3-3" }),
  ]),
  "status-burst": icon("status-burst", [
    primitive("path", {
      d: "m12 2 2 6 5.5-3.5-2.3 6.2 6.3.3-5.6 3.2 5.1 3.7-6.3-.2 1.7 6.1-4.9-4-2.4 5.8-.2-6.3-5.6 3 4-4.9-6.2.9 5.2-3.7Z",
    }),
  ]),
  "activation-basic-damage": icon("activation-basic-damage", [
    primitive("path", { d: "M3 12h14M12 6l6 6-6 6" }),
    primitive("circle", { cx: 5, cy: 12, r: 2, fill: "currentColor" }),
  ]),
  "activation-basic-heal": icon("activation-basic-heal", [
    primitive("path", { d: "M12 4v16M4 12h16" }),
    primitive("circle", { cx: 12, cy: 12, r: 8 }),
  ]),
  "activation-holy-word": icon("activation-holy-word", [
    primitive("circle", { cx: 12, cy: 12, r: 9 }),
    primitive("path", { d: "M12 5v14M5 12h14" }),
    primitive("path", { d: "M12 2v3M12 19v3M2 12h3M19 12h3" }),
  ]),
  "activation-burst": icon("activation-burst", [
    primitive("path", {
      d: "M12 2.5 14.5 9 21 6.5 17 12l4 5.5-6.5-2.5L12 21.5 9.5 15 3 17.5 7 12 3 6.5 9.5 9Z",
    }),
    primitive("circle", { cx: 12, cy: 12, r: 2, fill: "currentColor" }),
  ]),
  "activation-charge": icon("activation-charge", [
    primitive("path", { d: "M3 12h14M12 5l7 7-7 7" }),
    primitive("path", { d: "M3 7h5M3 17h5" }),
  ]),
  "activation-trap": icon("activation-trap", [
    primitive("path", { d: "M4 20 12 4l8 16Z" }),
    primitive("circle", { cx: 12, cy: 14, r: 3.5 }),
    primitive("path", { d: "M12 10.5V20M8.5 14h7" }),
  ]),
  "activation-poison": icon("activation-poison", [
    primitive("path", {
      d: "M12 2.5c3.4 4.7 5.6 7.7 5.6 11.3a5.6 5.6 0 1 1-11.2 0C6.4 10.2 8.6 7.2 12 2.5Z",
    }),
    primitive("path", { d: "M10 16.5h4" }),
  ]),
  "modifier-amplification": icon("modifier-amplification", [
    primitive("path", { d: "M12 21V4M5 11l7-7 7 7" }),
    primitive("path", { d: "M6 17h12" }),
  ]),
  "modifier-mitigation": icon("modifier-mitigation", [
    primitive("path", {
      d: "M12 2.8 20 6v5.3c0 4.8-3.1 8-8 9.9-4.9-1.9-8-5.1-8-9.9V6Z",
    }),
    primitive("path", { d: "M7.5 12h9" }),
  ]),
  "lifecycle-applied": icon("lifecycle-applied", [
    primitive("circle", { cx: 12, cy: 12, r: 9 }),
    primitive("path", { d: "M12 7v10M7 12h10" }),
  ]),
  "lifecycle-refreshed": icon("lifecycle-refreshed", [
    primitive("path", { d: "M19 8V3l-2.2 2.2A8 8 0 1 0 20 12" }),
    primitive("polyline", { points: "15,3 19,3 19,7" }),
  ]),
  "lifecycle-decremented": icon("lifecycle-decremented", [
    primitive("circle", { cx: 12, cy: 12, r: 9 }),
    primitive("path", { d: "M7 12h10" }),
  ]),
  "lifecycle-expired": icon("lifecycle-expired", [
    primitive("path", {
      d: "M7 3h10M7 21h10M8 3c0 5 3 5.5 4 9-1 3.5-4 4-4 9M16 3c0 5-3 5.5-4 9 1 3.5 4 4 4 9",
    }),
  ]),
  "lifecycle-trap-broken": icon("lifecycle-trap-broken", [
    primitive("rect", { x: 4, y: 4, width: 16, height: 16, rx: 2 }),
    primitive("path", { d: "m5 19 5-7-2-2 5-6-1 7 3 2-5 7 1-6Z" }),
  ]),
  "lifecycle-ended": icon("lifecycle-ended", [
    primitive("circle", { cx: 12, cy: 12, r: 8 }),
    primitive("path", { d: "M8.5 8.5h.1M15.5 8.5h.1M9 16c1.8-1.3 4.2-1.3 6 0" }),
  ]),
  "lifecycle-trap-broken-reapplied": icon("lifecycle-trap-broken-reapplied", [
    primitive("rect", { x: 3, y: 4, width: 14, height: 16, rx: 2 }),
    primitive("path", { d: "m4 19 5-7-2-2 5-6-1 7 3 2-5 7 1-6Z" }),
    primitive("path", { d: "M19 12v8M15 16h8" }),
  ]),
  unknown: icon("unknown", [
    primitive("path", {
      d: "M8.8 8.3a3.4 3.4 0 1 1 5.8 2.4c-1.7 1.4-2.6 2-2.6 4",
    }),
    primitive("circle", { cx: 12, cy: 19, r: 1, fill: "currentColor" }),
  ]),
});

export const KNOWN_GLYPH_KEYS = Object.freeze(Object.keys(ICONS));

/**
 * Return an allowlisted inline-vector definition or the safe unknown icon.
 *
 * @param {unknown} glyphKey
 * @returns {Readonly<IconDefinition>}
 */
export function iconDefinition(glyphKey) {
  if (typeof glyphKey !== "string" || !glyphKey.trim()) {
    return ICONS.unknown;
  }
  const normalized = glyphKey.trim();
  return Object.hasOwn(ICONS, normalized) ? ICONS[normalized] : ICONS.unknown;
}

/**
 * Create a nested SVG icon using only local primitives and `currentColor`.
 *
 * A supplied accessible name creates one labelled image. Without a label the
 * icon is decorative and hidden from accessibility APIs.
 *
 * @param {Document} ownerDocument
 * @param {unknown} glyphKey
 * @param {{accessibleName?: string, className?: string}} [options]
 * @returns {SVGSVGElement}
 */
export function createSvgIcon(ownerDocument, glyphKey, options = {}) {
  const definition = iconDefinition(glyphKey);
  const svg = /** @type {SVGSVGElement} */ (
    ownerDocument.createElementNS(SVG_NAMESPACE, "svg")
  );
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("focusable", "false");
  svg.dataset.icon = definition.glyphKey;
  if (options.className) {
    svg.setAttribute("class", options.className);
  }

  const accessibleName =
    typeof options.accessibleName === "string" &&
    options.accessibleName.trim().length > 0
      ? options.accessibleName.trim()
      : null;
  if (accessibleName) {
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", accessibleName);
  } else {
    svg.setAttribute("aria-hidden", "true");
  }

  for (const descriptor of definition.primitives) {
    const element = ownerDocument.createElementNS(SVG_NAMESPACE, descriptor.tag);
    for (const [name, value] of Object.entries(descriptor.attributes)) {
      element.setAttribute(name, String(value));
    }
    svg.append(element);
  }
  return svg;
}
