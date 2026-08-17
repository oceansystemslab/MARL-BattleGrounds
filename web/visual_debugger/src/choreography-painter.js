import { formatDisplayNumber } from "./display.js";
import { explainActivation, explainNetHealth } from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import { routeMarkerPose } from "./routes.js";
import { createSemanticDescriptor, registerTooltipOwner } from "./tooltip.js";
import { resolveVisualToken } from "./vocabulary.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * @param {unknown} publicAgentId
 */
function formatAgentIdentity(publicAgentId) {
  return typeof publicAgentId === "string" && publicAgentId.trim()
    ? `Agent ID ${publicAgentId}`
    : "Agent ID unavailable";
}

/**
 * Build one structured semantic explanation from fields already authorized in
 * the choreography plan. Internal slots may key DOM records, but never supply
 * display identity.
 *
 * @param {Record<string, any>} event
 */
export function explainChoreographyEvent(event) {
  const visibleEvent = paintAwareExplanationEvent(event);
  const title = String(
    visibleEvent.lifecycleToken?.label ??
      visibleEvent.token?.label ??
      visibleEvent.eventType ??
      "Semantic event",
  );
  const rows = [];
  const applicationSources = Array.isArray(visibleEvent.applicationSources)
    ? visibleEvent.applicationSources
    : [];
  if (applicationSources.length > 0) {
    rows.push({
      label: applicationSources.length === 1 ? "Source" : "Application Sources",
      value: applicationSources
        .map((source) => formatAgentIdentity(source?.sourcePublicAgentId))
        .join("; "),
      metadata: { compact: true, full: true },
    });
  }
  for (const [label, value] of [
    ["Actor", visibleEvent.actorPublicAgentId],
    [
      "Source",
      applicationSources.length === 0 ? visibleEvent.sourcePublicAgentId : null,
    ],
    ["Recipient", visibleEvent.recipientPublicAgentId],
    ["Agent", visibleEvent.agentPublicAgentId],
  ]) {
    if (typeof value === "string" && value.trim()) {
      rows.push({
        label,
        value: formatAgentIdentity(value),
        metadata: { compact: true, full: true },
      });
    }
  }
  const semanticOwnerKey = [
    visibleEvent.kind,
    visibleEvent.eventType,
    visibleEvent.tokenId,
    visibleEvent.lifecycle,
    visibleEvent.actorPresentationKey,
    visibleEvent.sourcePresentationKey,
    ...applicationSources.map((source) => source?.sourcePresentationKey),
    visibleEvent.recipientPresentationKey,
    visibleEvent.agentPresentationKey,
    visibleEvent.actorPublicAgentId,
    visibleEvent.sourcePublicAgentId,
    ...applicationSources.map((source) => source?.sourcePublicAgentId),
    visibleEvent.recipientPublicAgentId,
    visibleEvent.agentPublicAgentId,
  ]
    .filter((value) => typeof value === "string" && value.length > 0)
    .join(":");
  return createSemanticDescriptor({
    kind: "event",
    id: `semantic-event:${semanticOwnerKey || "unclassified"}`,
    title,
    tone: visibleEvent.kind === "rejected_action" ? "warning" : "information",
    accent: "none",
    summary: String(
      visibleEvent.lifecycleToken?.accessibleName ??
        visibleEvent.token?.accessibleName ??
        `Authoritative ${String(visibleEvent.kind ?? "event").replaceAll("_", " ")}`,
    ),
    rows,
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "pointer",
  });
}

/**
 * @typedef {Record<string, any>} JsonRecord
 * @typedef {{
 *   motionMode: "normal" | "reduced" | "off",
 *   settled: boolean,
 *   persistentOnly: boolean,
 *   retainTransientOnSettle?: boolean,
 * }} PainterOptions
 * @typedef {{
 *   element: Element,
 *   keyframes: Keyframe[] | PropertyIndexedKeyframes,
 *   options: KeyframeAnimationOptions,
 *   id: string,
 * }} AnimationSpec
 * @typedef {{
 *   root: SVGElement,
 *   routeRoot: SVGElement,
 *   eventNodes: Map<
 *     string,
 *     {group: SVGElement, underlay: SVGElement | null, event: JsonRecord}
 *   >,
 *   animationSpecs: readonly AnimationSpec[],
 *   nodeCount: number,
 *   persistentNodeCount: number,
 *   motionMode: "normal" | "reduced" | "off",
 *   retainTransientOnSettle: boolean,
 * }} PainterInstallation
 */

/**
 * Retained SVG painter for presentation-only choreography plans.
 */
export class SvgChoreographyPainter {
  /**
   * @param {JsonRecord} plan
   * @param {JsonRecord} surface
   * @param {PainterOptions} options
   * @returns {PainterInstallation}
   */
  install(plan, surface, options) {
    const root = svgElement(surface.ownerDocument, "g", {
      class: "combat-choreography",
      "aria-hidden": "true",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    const routeRoot = svgElement(surface.ownerDocument, "g", {
      class: "combat-choreography-routes",
      "aria-hidden": "true",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    /** @type {Map<
     *   string,
     *   {group: SVGElement, underlay: SVGElement | null, event: JsonRecord}
     * >} */
    const eventNodes = new Map();
    /** @type {AnimationSpec[]} */
    const animationSpecs = [];

    for (const event of plan.events) {
      if (
        !event.spatial ||
        event.presentationSuppressed === true ||
        event.kind === "unknown" ||
        (options.persistentOnly && !event.persistent)
      ) {
        continue;
      }
      const rendered = this.#renderEvent(
        surface.ownerDocument,
        event,
        plan,
        options,
        animationSpecs,
      );
      if (!rendered) {
        continue;
      }
      const { group, underlay } = rendered;
      eventNodes.set(event.eventId, { group, underlay, event });
      if (underlay) {
        routeRoot.append(underlay);
      }
      root.append(group);
    }
    const nodeCount =
      root.querySelectorAll("*").length + routeRoot.querySelectorAll("*").length + 2;
    const persistentNodeCount = Array.from(eventNodes.values())
      .filter(({ event }) => event.persistent)
      .reduce(
        (count, { group, underlay }) =>
          count +
          group.querySelectorAll("*").length +
          1 +
          (underlay ? underlay.querySelectorAll("*").length + 1 : 0),
        0,
      );
    if (nodeCount > Number(plan.bounds?.nodes ?? 512)) {
      throw new RangeError("choreography painter exceeded the planned node bound.");
    }
    if (animationSpecs.length > Number(plan.bounds?.animations ?? 512)) {
      throw new RangeError(
        "choreography painter exceeded the planned animation bound.",
      );
    }
    if (persistentNodeCount > Number(plan.bounds?.persistentNodes ?? 64)) {
      throw new RangeError(
        "choreography painter exceeded the planned persistent node bound.",
      );
    }
    (surface.routeLayer ?? surface.layer).append(routeRoot);
    surface.layer.append(root);
    /** @type {PainterInstallation} */
    const installation = {
      root,
      routeRoot,
      eventNodes,
      animationSpecs: Object.freeze(animationSpecs),
      nodeCount,
      persistentNodeCount,
      motionMode: options.motionMode,
      retainTransientOnSettle:
        options.settled && options.retainTransientOnSettle === true,
    };
    if (options.settled) {
      this.settle(installation);
    }
    return installation;
  }

  /**
   * @param {PainterInstallation | null} installation
   * @param {string} [_reason]
   */
  clear(installation, _reason) {
    installation?.root?.remove();
    installation?.routeRoot?.remove();
  }

  /**
   * @param {PainterInstallation} installation
   */
  settle(installation) {
    installation.root.dataset.state = "settled";
    installation.routeRoot.dataset.state = "settled";
    for (const [eventId, record] of installation.eventNodes) {
      const group = record.group;
      const persistent = group.dataset.persistent === "true";
      if (persistent || installation.retainTransientOnSettle) {
        group.dataset.settled = "true";
        group.setAttribute("opacity", "1");
        if (record.underlay) {
          record.underlay.dataset.settled = "true";
          record.underlay.setAttribute("opacity", "1");
        }
      } else {
        group.remove();
        record.underlay?.remove();
        installation.eventNodes.delete(eventId);
      }
    }
  }

  /**
   * Update geometry in place so active WAAPI objects retain current time.
   *
   * @param {PainterInstallation} installation
   * @param {JsonRecord} plan
   * @param {JsonRecord} surface
   */
  reproject(installation, plan, surface) {
    installation.root.dataset.viewportKey = surface.viewportKey;
    installation.routeRoot.dataset.viewportKey = surface.viewportKey;
    const nextEvents = /** @type {JsonRecord[]} */ (plan.events);
    const nextById = new Map(nextEvents.map((event) => [event.eventId, event]));
    for (const [eventId, record] of installation.eventNodes) {
      const next = nextById.get(eventId);
      if (!next?.spatial) {
        record.group.remove();
        record.underlay?.remove();
        installation.eventNodes.delete(eventId);
        continue;
      }
      this.#updateGeometry(record.group, record.underlay, next);
      record.event = next;
    }
  }

  /**
   * @param {Document} ownerDocument
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   * @returns {{group: SVGElement, underlay: SVGElement | null} | null}
   */
  #renderEvent(ownerDocument, event, plan, options, animationSpecs) {
    const group = svgElement(ownerDocument, "g", {
      class: `combat-effect combat-effect--${cssIdentifier(event.kind)}`,
      opacity: options.settled || options.motionMode === "off" ? 1 : 0,
      "data-event-id": event.eventId,
      "data-event-type": event.eventType,
      "data-phase": phaseFor(event),
      "data-persistent": Boolean(event.persistent),
    });
    assignSlot(group, "source", event.sourceSlot);
    assignSlot(group, "target", event.targetSlot);
    assignSlot(group, "recipient", event.recipientSlot);
    assignSlot(group, "actor", event.actorSlot);
    assignPresentationKey(group, "source", event.sourcePresentationKey);
    assignPresentationKey(group, "target", event.targetPresentationKey);
    assignPresentationKey(group, "recipient", event.recipientPresentationKey);
    assignPresentationKey(group, "actor", event.actorPresentationKey);
    if (typeof event.tokenId === "string") {
      group.dataset.tokenId = event.tokenId;
    }
    if (typeof event.sourceClass?.cssKey === "string") {
      group.dataset.sourceClass = event.sourceClass.cssKey;
    }
    if (typeof event.outcome === "string") {
      group.dataset.outcome = event.outcome;
    }
    if (typeof event.impactSemantic === "string") {
      group.dataset.impactSemantic = event.impactSemantic;
    }
    if (typeof event.lifecycle === "string") {
      group.dataset.lifecycle = event.lifecycle;
    }
    if (Number.isInteger(event.lane)) {
      group.dataset.lane = String(event.lane);
    }
    if (Number.isInteger(event.laneCount)) {
      group.dataset.laneCount = String(event.laneCount);
    }
    if (typeof event.component === "string") {
      group.dataset.component = event.component;
    }
    if (typeof event.cueSemantic === "string") {
      group.dataset.cueSemantic = event.cueSemantic;
    }
    if (Number.isInteger(event.teamId)) {
      group.dataset.teamId = String(event.teamId);
    }
    if (Number.isInteger(event.teamIndex)) {
      group.dataset.teamIndex = String(event.teamIndex);
    }
    if (typeof event.teamSide === "string") {
      group.dataset.teamSide = event.teamSide;
    }
    if (typeof event.label === "string") {
      group.dataset.label = event.label;
    }
    if (Number.isInteger(event.agentSlot)) {
      group.dataset.agentSlot = String(event.agentSlot);
    }
    assignPresentationKey(group, "agent", event.agentPresentationKey);
    if (typeof event.movementMaskValue === "boolean") {
      group.dataset.movementMaskValue = String(event.movementMaskValue);
    }
    if (typeof event.pairMaskValue === "boolean") {
      group.dataset.pairMaskValue = String(event.pairMaskValue);
    }
    if (Number.isInteger(event.durationBefore)) {
      group.dataset.durationBefore = String(event.durationBefore);
    }
    if (Number.isInteger(event.durationAfter)) {
      group.dataset.durationAfter = String(event.durationAfter);
    }
    if (Array.isArray(event.applicationEventIds)) {
      group.dataset.applicationEventIds = JSON.stringify(event.applicationEventIds);
    }
    if (Array.isArray(event.atomicEventIds)) {
      group.dataset.atomicEventIds = JSON.stringify(event.atomicEventIds);
    }
    const underlay =
      event.route || event.kind === "charge_displacement"
        ? svgElement(ownerDocument, "g", {
            class: `combat-route-effect combat-route-effect--${cssIdentifier(event.kind)}`,
            opacity: options.settled || options.motionMode === "off" ? 1 : 0,
          })
        : null;
    if (underlay) {
      copyEventMetadata(group, underlay);
    }

    if (event.kind === "activation") {
      this.#renderActivation(
        ownerDocument,
        group,
        underlay,
        event,
        plan,
        options,
        animationSpecs,
      );
    } else if (event.kind === "net_health") {
      this.#renderNet(ownerDocument, group, event);
    } else if (event.kind === "regeneration") {
      this.#renderRegeneration(ownerDocument, group, event);
    } else if (event.kind === "charge_displacement") {
      this.#renderCharge(ownerDocument, group, underlay, event);
    } else if (event.kind === "status_lifecycle") {
      this.#renderLifecycle(ownerDocument, group, event, plan, options, animationSpecs);
    } else if (event.kind === "rejected_action") {
      this.#renderRejection(ownerDocument, group, underlay, event);
    } else if (event.kind === "semantic_pulse") {
      this.#renderSemanticPulse(
        ownerDocument,
        group,
        event,
        plan,
        options,
        animationSpecs,
      );
    } else {
      return null;
    }
    this.#registerEventExplanation(group, underlay, event);
    this.#applySpatialDisposition(group, event);

    if (!options.settled && options.motionMode !== "off") {
      const reduced = options.motionMode === "reduced";
      const authoredPhaseStart = Number(event.phaseStart ?? 0);
      const authoredPhaseEnd = Number(event.phaseEnd ?? plan.phases.total);
      const reducedScale =
        Number(plan.phases.reducedTotal ?? 220) /
        Math.max(Number(plan.phases.total ?? 900), 1);
      const phaseStart = reduced
        ? authoredPhaseStart * reducedScale
        : authoredPhaseStart;
      const phaseEnd = reduced ? authoredPhaseEnd * reducedScale : authoredPhaseEnd;
      const activationHasImpact =
        event.kind === "activation" && group.querySelector(".combat-impact") !== null;
      const targets = underlay
        ? event.kind === "activation"
          ? [{ element: underlay, part: "route" }]
          : [
              { element: underlay, part: "route" },
              { element: group, part: "group" },
            ]
        : activationHasImpact
          ? []
          : [{ element: group, part: "group" }];
      if (event.kind === "activation" && (underlay || activationHasImpact)) {
        group.setAttribute("opacity", "1");
      }
      const duration = phaseEnd - phaseStart;
      if (!(duration > 0)) {
        return { group, underlay };
      }
      for (const target of targets) {
        animationSpecs.push(
          animationSpec(
            target.element,
            eventKeyframes(event, options.motionMode),
            {
              delay: phaseStart,
              duration,
              easing: "ease-out",
              fill: "both",
            },
            plan,
            event,
            target.part,
          ),
        );
      }
    }
    return { group, underlay };
  }

  /**
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #renderActivation(
    ownerDocument,
    group,
    underlay,
    event,
    plan,
    options,
    animationSpecs,
  ) {
    const abilityEnabled = paintPartEnabled(event, "ability");
    if (event.route) {
      if (!underlay) {
        return;
      }
      const path = svgElement(ownerDocument, "path", {
        class: "combat-route__path",
        d: event.route.path,
        pathLength: 1,
      });
      const hitPath = svgElement(ownerDocument, "path", {
        class: "combat-route__hit",
        d: event.route.path,
      });
      const arrow = svgElement(ownerDocument, "path", {
        class: "combat-route__arrow",
        d: "M -11 -6 L 2 0 L -11 6 L -7 0 Z",
      });
      underlay.append(hitPath, path, arrow);
      if (
        event.tokenId === "warrior_charge" &&
        ((Number.isInteger(event.sourceSlot) && Number.isInteger(event.targetSlot)) ||
          (typeof event.sourcePresentationKey === "string" &&
            event.sourcePresentationKey &&
            typeof event.targetPresentationKey === "string" &&
            event.targetPresentationKey))
      ) {
        const ownership = svgElement(ownerDocument, "g", {
          class: "combat-route__ownership",
          "aria-hidden": "true",
          "data-source-slot": event.sourceSlot,
          "data-target-slot": event.targetSlot,
          "data-source-presentation-key": event.sourcePresentationKey,
          "data-target-presentation-key": event.targetPresentationKey,
        });
        const ownershipLabel = `${formatAgentIdentity(event.sourcePublicAgentId)} → ${formatAgentIdentity(event.targetPublicAgentId)}`;
        ownership.append(
          svgElement(ownerDocument, "line", {
            class: "combat-route__ownership-leader",
          }),
          svgElement(ownerDocument, "rect", {
            class: "combat-route__ownership-box",
            x: -34,
            y: -9,
            width: 68,
            height: 18,
            rx: 5,
          }),
          svgElement(ownerDocument, "text", {
            class: "combat-route__ownership-label",
            x: 0,
            y: 0,
          }),
        );
        const label = ownership.lastElementChild;
        if (label) {
          label.textContent = ownershipLabel;
        }
        underlay.append(ownership);
      }
      if (options.motionMode === "normal") {
        const particle = svgElement(ownerDocument, "circle", {
          class: "combat-route__particle",
          cx: 0,
          cy: 0,
          r: event.tokenId === "holy_word" ? 4 : 3,
        });
        particle.style.offsetPath = `path("${event.route.path}")`;
        particle.style.offsetRotate = "auto";
        underlay.append(particle);
        animationSpecs.push(
          animationSpec(
            particle,
            [
              { offsetDistance: "0%", opacity: 0 },
              { opacity: 1, offset: 0.15 },
              { offsetDistance: "100%", opacity: 1, offset: 0.86 },
              { opacity: 0 },
            ],
            {
              delay: Number(plan.phases.travelStart ?? 80),
              duration: 430,
              easing: "cubic-bezier(.2,.72,.25,1)",
              fill: "both",
            },
            plan,
            event,
            "particle",
          ),
        );
      }
      const impact = this.#appendImpact(ownerDocument, group, event.route.end, event);
      this.#animateImpact(impact, event, plan, options, animationSpecs);
      this.#updateActivationGeometry(group, underlay, event);
      return;
    }
    if (event.presentationKind === "target_only_impact") {
      const impact = this.#appendImpact(ownerDocument, group, event.target, event);
      this.#animateImpact(impact, event, plan, options, animationSpecs);
      this.#updateActivationGeometry(group, underlay, event);
      return;
    }
    if (!abilityEnabled) {
      return;
    }
    const anchor = event.source;
    if (!anchor) {
      return;
    }
    const local = svgElement(ownerDocument, "g", {
      class: `combat-local combat-local--${cssIdentifier(event.tokenId)}`,
    });
    local.append(
      svgElement(ownerDocument, "circle", {
        class: "combat-local__core",
        r: event.tokenId === "mage_burst" ? 24 : 16,
      }),
      svgElement(ownerDocument, "circle", {
        class: "combat-local__ring",
        r: event.tokenId === "mage_burst" ? 28 : 20,
      }),
    );
    if (event.tokenId === "mage_burst") {
      local.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-burst__wave combat-burst__wave--inner",
          r: 18,
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-burst__wave combat-burst__wave--outer",
          r: 28,
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-burst__flare",
          d: "M 0 -34 V -25 M 0 25 V 34 M -34 0 H -25 M 25 0 H 34 M -24 -24 L -18 -18 M 18 18 L 24 24 M 24 -24 L 18 -18 M -18 18 L -24 24",
        }),
      );
    }
    const icon = createSvgIcon(ownerDocument, event.token?.glyphKey ?? "unknown", {
      className: "combat-local__icon",
    });
    setAttributes(icon, { x: -10, y: -10, width: 20, height: 20 });
    local.append(icon);
    group.append(local);
    setAttributes(local, { transform: `translate(${anchor.x} ${anchor.y})` });
    if (
      event.tokenId === "mage_burst" &&
      !options.settled &&
      options.motionMode !== "off"
    ) {
      const wave = local.querySelector(".combat-burst__wave--outer");
      if (wave) {
        const waveKeyframes =
          options.motionMode === "reduced"
            ? [{ opacity: 0 }, { opacity: 1, offset: 0.25 }, { opacity: 0 }]
            : [
                { opacity: 0, transform: "scale(.45)" },
                { opacity: 1, offset: 0.25, transform: "scale(.85)" },
                { opacity: 0, transform: "scale(1.5)" },
              ];
        animationSpecs.push(
          animationSpec(
            wave,
            waveKeyframes,
            {
              duration:
                options.motionMode === "reduced"
                  ? Number(plan.phases.reducedTotal ?? 220)
                  : 550,
              easing: "ease-out",
              fill: "both",
            },
            plan,
            event,
            "burst-wave",
          ),
        );
      }
    }
  }

  /**
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord | null} target
   * @param {JsonRecord} event
   * @returns {SVGElement | null}
   */
  #appendImpact(ownerDocument, group, target, event) {
    const abilityEnabled = paintPartEnabled(event, "ability");
    const semanticEnabled = paintPartEnabled(event, "semantic");
    if (!target || (!abilityEnabled && !semanticEnabled)) {
      return null;
    }
    const impact = svgElement(ownerDocument, "g", {
      class: `combat-impact combat-impact--${cssIdentifier(event.tokenId)}`,
    });
    if (abilityEnabled) {
      const compact =
        event.tokenId === "basic_damage" || event.tokenId === "basic_heal";
      impact.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-impact__hit",
          r: 22,
          "aria-hidden": "true",
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-impact__core",
          r: compact ? 7 : 13,
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-impact__ring",
          r: compact ? 10 : 18,
        }),
      );
    }
    if (semanticEnabled) {
      impact.append(semanticImpactGlyph(ownerDocument, event.impactSemantic));
    }
    if (abilityEnabled && event.tokenId === "holy_word") {
      impact.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-holy__pulse combat-holy__pulse--inner",
          r: 12,
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-holy__pulse combat-holy__pulse--outer",
          r: 22,
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-holy__flare",
          d: "M 0 -27 V -21 M 0 21 V 27 M -27 0 H -21 M 21 0 H 27",
        }),
      );
    } else if (abilityEnabled && event.tokenId === "hunter_trap") {
      impact.append(
        svgElement(ownerDocument, "path", {
          class: "combat-trap__lattice",
          d: "M -17 -17 H 17 V 17 H -17 Z M -17 -6 H 17 M -17 6 H 17 M -6 -17 V 17 M 6 -17 V 17",
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-trap__flare",
          d: "M 0 -25 25 0 0 25 -25 0 Z",
        }),
      );
    } else if (abilityEnabled && event.tokenId === "rogue_poison") {
      impact.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-poison__splash",
          cx: -15,
          cy: 8,
          r: 3,
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-poison__splash",
          cx: 14,
          cy: 10,
          r: 4,
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-poison__splash",
          cx: 10,
          cy: -13,
          r: 2.5,
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-poison__flare",
          d: "M -23 16 Q -14 7 -7 18 M 7 18 Q 14 7 23 16",
        }),
      );
    } else if (abilityEnabled && event.tokenId === "warrior_charge") {
      impact.append(
        svgElement(ownerDocument, "path", {
          class: "combat-charge__impact",
          d: "M -22 0 H -13 M 13 0 H 22 M 0 -22 V -13 M 0 13 V 22",
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-charge__flare",
          d: "M -26 -11 L -18 -7 M 18 7 L 26 11 M -26 11 L -18 7 M 18 -7 L 26 -11",
        }),
      );
    }
    group.append(impact);
    setAttributes(impact, { transform: impactTransform(event, target) });
    return impact;
  }

  /**
   * @param {SVGElement | null} impact
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #animateImpact(impact, event, plan, options, animationSpecs) {
    if (!impact || options.settled || options.motionMode === "off") {
      return;
    }
    const reduced = options.motionMode === "reduced";
    const authoredDelay = Number(event.phaseImpact ?? plan.phases.impactStart ?? 360);
    const authoredPhaseEnd = Number(event.phaseEnd ?? plan.phases.settleStart ?? 760);
    const reducedScale =
      Number(plan.phases.reducedTotal ?? 220) /
      Math.max(Number(plan.phases.total ?? 900), 1);
    const delay = reduced ? authoredDelay * reducedScale : authoredDelay;
    const phaseEnd = reduced ? authoredPhaseEnd * reducedScale : authoredPhaseEnd;
    const duration = phaseEnd - delay;
    if (!(duration > 0)) {
      return;
    }
    const keyframes =
      event.tokenId === "holy_word"
        ? [{ opacity: 0 }, { opacity: 1, offset: 0.28 }, { opacity: 0 }]
        : [{ opacity: 0 }, { opacity: 1, offset: 0.24 }, { opacity: 0 }];
    animationSpecs.push(
      animationSpec(
        impact,
        keyframes,
        {
          delay,
          duration,
          easing: "ease-out",
          fill: "both",
        },
        plan,
        event,
        "impact",
      ),
    );
  }

  /**
   * Attach authorized-data-only explanations to the visible event and its
   * independently layered route hit region.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #registerEventExplanation(group, underlay, event) {
    if (event.tokenId === "basic_damage" || event.tokenId === "basic_heal") {
      return;
    }
    const explanationEvent = paintAwareExplanationEvent(event);
    const explanation =
      event.kind === "activation" && paintPartEnabled(event, "ability")
        ? explainActivation(explanationEvent)
        : event.kind === "net_health" &&
            (paintPartEnabled(event, "battleText") ||
              paintPartEnabled(event, "recipientText"))
          ? explainNetHealth(explanationEvent)
          : explainChoreographyEvent(explanationEvent);
    registerTooltipOwner(group, explanation);
    if (underlay) {
      registerTooltipOwner(
        underlay,
        createSemanticDescriptor({
          ...explanation,
          kind: event.kind === "activation" ? "accepted-route" : explanation.kind,
          id: `${explanation.id}:route`,
          anchor: "pointer",
        }),
      );
    }
  }

  /**
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord} event
   */
  #renderNet(ownerDocument, group, event) {
    if (!event.recipient) {
      return;
    }
    const effectEnabled = paintPartEnabled(event, "effect");
    const battleTextEnabled = paintPartEnabled(event, "battleText");
    const recipientTextEnabled = paintPartEnabled(event, "recipientText");
    const cueGeometryEnabled =
      effectEnabled ||
      (event.outcome === "unchanged" && (battleTextEnabled || recipientTextEnabled));
    group.dataset.netDelta = String(event.netDelta);
    group.dataset.layoutCollisionFree = String(event.cueCollisionFree !== false);
    if (cueGeometryEnabled) {
      group.append(
        svgElement(ownerDocument, "line", {
          class: "combat-cue__leader",
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-net__recipient-anchor",
          r: 3,
        }),
      );
    }
    if (recipientTextEnabled) {
      const recipientLabel = svgElement(ownerDocument, "text", {
        class: "combat-net__recipient",
      });
      recipientLabel.textContent = formatAgentIdentity(event.recipientPublicAgentId);
      group.append(recipientLabel);
    }
    if (battleTextEnabled) {
      const label = svgElement(ownerDocument, "text", {
        class: "combat-net__label",
      });
      label.textContent = netLabel(event.netDelta, event.outcome);
      group.append(label);
    }
    this.#updateNetGeometry(group, event);
  }

  /**
   * Paint one successor-anchored regeneration result with the same universal
   * plus grammar as Priest healing. Its collision-packed cue remains linked to
   * the exact authorized agent endpoint without inventing a route or source.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord} event
   */
  #renderRegeneration(ownerDocument, group, event) {
    if (!event.recipient || !Number.isFinite(event.value)) {
      return;
    }
    const effectEnabled = paintPartEnabled(event, "effect");
    const battleTextEnabled = paintPartEnabled(event, "battleText");
    const cue = svgElement(ownerDocument, "g", {
      class: "combat-regeneration",
      "data-layout-collision-free": String(event.cueCollisionFree !== false),
    });
    if (effectEnabled) {
      const hit = svgElement(ownerDocument, "circle", {
        class: "combat-regeneration__hit",
        r: 24,
      });
      hit.addEventListener("pointerdown", (pointerEvent) => {
        if (pointerEvent.button === 0) {
          pointerEvent.stopPropagation();
        }
      });
      const plus = semanticImpactGlyph(ownerDocument, "healing");
      plus.classList.add("combat-regeneration__plus");
      cue.append(
        hit,
        svgElement(ownerDocument, "circle", {
          class: "combat-regeneration__pulse",
          r: 13,
        }),
        plus,
      );
    }
    if (battleTextEnabled) {
      const value = svgElement(ownerDocument, "text", {
        class: "combat-regeneration__value",
        x: 0,
        y: 25,
      });
      value.textContent = `+${formatDisplayNumber(event.value)}`;
      cue.append(value);
    }
    group.dataset.value = String(event.value);
    group.dataset.layoutCollisionFree = String(event.cueCollisionFree !== false);
    if (effectEnabled) {
      group.append(
        svgElement(ownerDocument, "line", {
          class: "combat-cue__leader",
        }),
        svgElement(ownerDocument, "circle", {
          class: "combat-regeneration__recipient-anchor",
          r: 3,
        }),
      );
    }
    group.append(cue);
    const position = event.cue ?? event.recipient;
    setAttributes(cue, {
      transform: `translate(${position.x} ${position.y})`,
    });
    this.#updateCueLeader(group, event, 18);
  }

  /**
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #renderCharge(ownerDocument, group, underlay, event) {
    if (!event.start || !event.end || !underlay) {
      return;
    }
    group.dataset.pathKind = event.pathKind;
    underlay.append(
      svgElement(ownerDocument, "path", {
        class: "combat-charge__path",
      }),
    );
    group.append(
      svgElement(ownerDocument, "circle", {
        class: "combat-charge__endpoint combat-charge__endpoint--start",
        r: 4,
      }),
      svgElement(ownerDocument, "circle", {
        class: "combat-charge__endpoint combat-charge__endpoint--end",
        r: 5,
      }),
      svgElement(ownerDocument, "path", {
        class: "combat-charge__direction",
        d: "M -9 -5 L 1 0 L -9 5 Z",
      }),
    );
    this.#updateChargeGeometry(group, underlay, event);
  }

  /**
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #renderLifecycle(ownerDocument, group, event, plan, options, animationSpecs) {
    if (!event.recipient) {
      return;
    }
    const combinedLifecycle = event.lifecycle === "trap_broken_and_reapplied";
    const effectEnabled = paintPartEnabled(event, "effect");
    const breakEnabled = combinedLifecycle
      ? paintPartEnabled(event, "break")
      : effectEnabled;
    const reapplicationEnabled = combinedLifecycle
      ? paintPartEnabled(event, "reapplication")
      : effectEnabled && event.lifecycle === "reapplied";
    const lifecycle = svgElement(ownerDocument, "g", {
      class: "combat-lifecycle",
      "data-layout-collision-free": String(event.cueCollisionFree !== false),
    });
    if (breakEnabled) {
      const lifecycleHit = svgElement(ownerDocument, "circle", {
        class: "combat-lifecycle__hit",
        r: 26,
      });
      lifecycleHit.addEventListener("pointerdown", (pointerEvent) => {
        if (pointerEvent.button === 0) {
          pointerEvent.stopPropagation();
        }
      });
      lifecycle.append(
        lifecycleHit,
        svgElement(ownerDocument, "circle", {
          class: "combat-lifecycle__ring",
          r: 17,
        }),
      );
      const statusIcon = createSvgIcon(
        ownerDocument,
        event.token?.glyphKey ?? "unknown",
        { className: "combat-lifecycle__status-icon" },
      );
      setAttributes(statusIcon, { x: -10, y: -10, width: 20, height: 20 });
      const change = svgElement(ownerDocument, "g", {
        class: "combat-lifecycle__change",
        transform: "translate(13 -13)",
      });
      change.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-lifecycle__change-disc",
          r: 8,
        }),
      );
      const changeToken =
        combinedLifecycle && !reapplicationEnabled
          ? resolveVisualToken("lifecycle", "trap_broken", event)
          : event.lifecycleToken;
      const changeIcon = createSvgIcon(
        ownerDocument,
        changeToken?.glyphKey ?? "unknown",
        { className: "combat-lifecycle__change-icon" },
      );
      setAttributes(changeIcon, { x: -6, y: -6, width: 12, height: 12 });
      change.append(changeIcon);
      lifecycle.append(statusIcon, change);
    }
    if (breakEnabled && event.lifecycle === "cleared_by_death") {
      const sweep = svgElement(ownerDocument, "g", {
        class: "combat-lifecycle__death-sweep",
      });
      sweep.append(
        svgElement(ownerDocument, "path", {
          class: "combat-lifecycle__death-sweep-arc",
          d: "M -23 -8 C -8 -23 10 -23 23 -7",
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-lifecycle__death-sweep-cut",
          d: "M -22 10 L 22 -10",
        }),
      );
      lifecycle.append(sweep);
    }
    if (
      breakEnabled &&
      (event.lifecycle === "trap_broken" ||
        event.lifecycle === "trap_broken_and_reapplied")
    ) {
      for (let index = 0; index < 6; index += 1) {
        const angle = (Math.PI * 2 * index) / 6;
        lifecycle.append(
          svgElement(ownerDocument, "line", {
            class: "combat-lifecycle__shard",
            x1: Math.cos(angle) * 11,
            y1: Math.sin(angle) * 11,
            x2: Math.cos(angle) * 23,
            y2: Math.sin(angle) * 23,
          }),
        );
      }
    }
    if (reapplicationEnabled) {
      const reapply = svgElement(ownerDocument, "g", {
        class: "combat-lifecycle__reapply",
        opacity: options.settled || options.motionMode === "off" ? 1 : 0,
      });
      reapply.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-lifecycle__reapply-ring",
          r: 21,
        }),
      );
      const reapplyIcon = createSvgIcon(
        ownerDocument,
        event.token?.glyphKey ?? "unknown",
        { className: "combat-lifecycle__reapply-icon" },
      );
      setAttributes(reapplyIcon, { x: -9, y: -9, width: 18, height: 18 });
      reapply.append(reapplyIcon);
      lifecycle.append(reapply);
      if (!options.settled && options.motionMode !== "off") {
        const reduced = options.motionMode === "reduced";
        const authoredDelay =
          Number(event.phaseStart ?? plan.phases.outcomeStart ?? 420) + 150;
        const authoredDuration = 320;
        const reducedScale =
          Number(plan.phases.reducedTotal ?? 220) /
          Math.max(Number(plan.phases.total ?? 900), 1);
        animationSpecs.push(
          animationSpec(
            reapply,
            [{ opacity: 0 }, { opacity: 0, offset: 0.35 }, { opacity: 1 }],
            {
              delay: reduced ? authoredDelay * reducedScale : authoredDelay,
              duration: reduced ? authoredDuration * reducedScale : authoredDuration,
              easing: "ease-out",
              fill: "both",
            },
            plan,
            event,
            "reapply",
          ),
        );
      }
    }
    group.append(
      svgElement(ownerDocument, "line", {
        class: "combat-cue__leader",
      }),
      lifecycle,
    );
    const position = event.cue ?? event.recipient;
    setAttributes(lifecycle, {
      transform: `translate(${position.x} ${position.y})`,
    });
    this.#updateCueLeader(group, event, 21);
  }

  /**
   * Render a compact, event-specific pulse at an anchor supplied directly by
   * the V2 visual adapter (or by the presentation-only team-clock corner).
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #renderSemanticPulse(ownerDocument, group, event, plan, options, animationSpecs) {
    if (!event.anchor) {
      return;
    }
    if (event.cueSemantic === "agent_died" || event.cueSemantic === "agent_respawned") {
      this.#renderLifecycleRing(
        ownerDocument,
        group,
        event,
        plan,
        options,
        animationSpecs,
      );
      return;
    }
    if (event.cueSemantic === "respawn_wave_occurred") {
      this.#renderRespawnWave(ownerDocument, group, event);
      return;
    }
    const pulse = svgElement(ownerDocument, "g", {
      class: `combat-semantic-pulse combat-semantic-pulse--${cssIdentifier(event.cueSemantic)}`,
      "data-semantic": event.cueSemantic,
    });
    const pulseHit = svgElement(ownerDocument, "circle", {
      class: "combat-semantic-pulse__hit",
      r: 31,
    });
    pulseHit.addEventListener("pointerdown", (pointerEvent) => {
      if (pointerEvent.button === 0) {
        pointerEvent.stopPropagation();
      }
    });
    pulse.append(
      pulseHit,
      svgElement(ownerDocument, "circle", {
        class: "combat-semantic-pulse__ring",
        r: 19,
      }),
      svgElement(ownerDocument, "circle", {
        class: "combat-semantic-pulse__core",
        r: 10,
      }),
    );
    if (
      event.cueSemantic === "cooldown_started" ||
      event.cueSemantic === "cooldown_ready"
    ) {
      pulse.append(
        svgElement(ownerDocument, "path", {
          class: "combat-semantic-pulse__mark",
          d:
            event.cueSemantic === "cooldown_ready"
              ? "M -7 0 L -2 6 L 8 -7"
              : "M 0 -8 V 0 L 6 4 M -5 -11 H 5",
        }),
      );
    } else if (event.cueSemantic === "spawn_shield_expired") {
      pulse.append(
        svgElement(ownerDocument, "circle", {
          class: "combat-semantic-pulse__shield-shell",
          r: 24,
          fill: "none",
          stroke: "currentColor",
          "stroke-width": 3,
          "stroke-dasharray": "3 3",
          "vector-effect": "non-scaling-stroke",
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-semantic-pulse__mark",
          d: "M 0 -11 L 9 -7 V 1 C 9 7 5 10 0 13 C -5 10 -9 7 -9 1 V -7 Z M -3 -7 L 2 -1 L -2 3 L 4 9",
        }),
      );
    }
    if (Number.isFinite(event.value)) {
      const value = svgElement(ownerDocument, "text", {
        class: "combat-semantic-pulse__value",
        x: 0,
        y: 29,
      });
      value.textContent = `+${formatDisplayNumber(event.value)}`;
      pulse.append(value);
    }
    group.append(pulse);
    setAttributes(pulse, {
      transform: `translate(${event.anchor.x} ${event.anchor.y})`,
    });
  }

  /**
   * Render one successor-anchored outward lifecycle ring. The DOM radius is
   * the settled endpoint; only normal motion animates the child circle from a
   * smaller radius, so reduced/off/settled modes never show an inward state.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #renderLifecycleRing(ownerDocument, group, event, plan, options, animationSpecs) {
    const lifecycle = event.cueSemantic === "agent_died" ? "death" : "resurrection";
    const ringGroup = svgElement(ownerDocument, "g", {
      class: `combat-lifecycle-ring combat-lifecycle-ring--${lifecycle}`,
      "data-lifecycle-ring": lifecycle,
    });
    const hit = svgElement(ownerDocument, "circle", {
      class: "combat-lifecycle-ring__hit",
      r: 34,
    });
    hit.addEventListener("pointerdown", (pointerEvent) => {
      if (pointerEvent.button === 0) {
        pointerEvent.stopPropagation();
      }
    });
    const ring = svgElement(ownerDocument, "circle", {
      class: "combat-lifecycle-ring__ring",
      r: 32,
    });
    ringGroup.append(hit, ring);
    group.append(ringGroup);
    setAttributes(ringGroup, {
      transform: `translate(${event.anchor.x} ${event.anchor.y})`,
    });
    if (!options.settled && options.motionMode === "normal") {
      const phaseStart = Number(event.phaseStart ?? 0);
      const phaseEnd = Number(event.phaseEnd ?? plan.phases.total);
      const duration = phaseEnd - phaseStart;
      if (duration > 0) {
        animationSpecs.push(
          animationSpec(
            ring,
            [
              { r: "9px", opacity: 0 },
              { r: "17px", opacity: 1, offset: 0.24 },
              { r: "25px", opacity: 1, offset: 0.58 },
              { r: "32px", opacity: 1 },
            ],
            {
              delay: phaseStart,
              duration,
              easing: "ease-out",
              fill: "both",
            },
            plan,
            event,
            "lifecycle-ring",
          ),
        );
      }
    }
  }

  /**
   * Render one compact persistent wave banner at its authorized team side.
   * The banner itself is the event hit surface; its text is not a second
   * tooltip or scientific owner.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {JsonRecord} event
   */
  #renderRespawnWave(ownerDocument, group, event) {
    const wave = svgElement(ownerDocument, "g", {
      class: `combat-respawn-wave combat-respawn-wave--team-${event.teamIndex === 0 ? "a" : "b"}`,
      "data-team-index": event.teamIndex,
      "data-team-id": event.teamId,
      "data-team-side": event.teamSide,
    });
    wave.append(
      svgElement(ownerDocument, "rect", {
        class: "combat-respawn-wave__panel",
        x: -76,
        y: -16,
        width: 152,
        height: 32,
        rx: 8,
      }),
      svgElement(ownerDocument, "text", {
        class: "combat-respawn-wave__label",
        x: 0,
        y: 1,
      }),
    );
    const label = wave.lastElementChild;
    if (label) {
      label.textContent = String(event.label);
    }
    group.append(wave);
    setAttributes(wave, {
      transform: `translate(${event.anchor.x} ${event.anchor.y})`,
    });
  }

  /**
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #renderRejection(ownerDocument, group, underlay, event) {
    if (!event.actor) {
      return;
    }
    const ring = svgElement(ownerDocument, "circle", {
      class: "combat-rejection__ring",
      cx: event.actor.x,
      cy: event.actor.y,
      r: 24,
    });
    group.append(ring);
    if (event.route && underlay) {
      underlay.append(
        svgElement(ownerDocument, "path", {
          class: "combat-rejection__route",
          d: event.route.path,
        }),
      );
    }
  }

  /**
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #updateGeometry(group, underlay, event) {
    this.#applySpatialDisposition(group, event);
    if (event.kind === "activation") {
      this.#updateActivationGeometry(group, underlay, event);
    } else if (event.kind === "net_health") {
      this.#updateNetGeometry(group, event);
    } else if (event.kind === "regeneration") {
      const cue = group.querySelector(".combat-regeneration");
      if (cue instanceof SVGElement && event.recipient) {
        cue.dataset.layoutCollisionFree = String(event.cueCollisionFree !== false);
        const position = event.cue ?? event.recipient;
        setAttributes(cue, {
          transform: `translate(${position.x} ${position.y})`,
        });
        this.#updateCueLeader(group, event, 18);
      }
    } else if (event.kind === "charge_displacement") {
      this.#updateChargeGeometry(group, underlay, event);
    } else if (event.kind === "status_lifecycle") {
      const lifecycle = group.querySelector(".combat-lifecycle");
      if (lifecycle instanceof SVGElement && event.recipient) {
        lifecycle.dataset.layoutCollisionFree = String(
          event.cueCollisionFree !== false,
        );
        const position = event.cue ?? event.recipient;
        setAttributes(lifecycle, {
          transform: `translate(${position.x} ${position.y})`,
        });
        this.#updateCueLeader(group, event, 21);
      }
    } else if (event.kind === "rejected_action") {
      const ring = group.querySelector(".combat-rejection__ring");
      if (ring && event.actor) {
        setAttributes(ring, { cx: event.actor.x, cy: event.actor.y });
      }
      const route = underlay?.querySelector(".combat-rejection__route");
      if (route && event.route) {
        route.setAttribute("d", event.route.path);
      }
    } else if (event.kind === "semantic_pulse") {
      const pulse = group.querySelector(
        ".combat-semantic-pulse, .combat-lifecycle-ring, .combat-respawn-wave",
      );
      if (pulse && event.anchor) {
        pulse.setAttribute(
          "transform",
          `translate(${event.anchor.x} ${event.anchor.y})`,
        );
      }
    }
  }

  /**
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #updateActivationGeometry(group, underlay, event) {
    const path = underlay?.querySelector(".combat-route__path");
    const hitPath = underlay?.querySelector(".combat-route__hit");
    const arrow = underlay?.querySelector(".combat-route__arrow");
    const ownership = underlay?.querySelector(".combat-route__ownership");
    if (path && event.route) {
      path.setAttribute("d", event.route.path);
    }
    if (hitPath && event.route) {
      hitPath.setAttribute("d", event.route.path);
    }
    if (arrow && event.route) {
      const marker = routeMarkerPose(
        event.route,
        event.tokenId === "warrior_charge" ? 0.42 : undefined,
      );
      setAttributes(arrow, {
        transform: `translate(${marker.x} ${marker.y}) rotate(${marker.degrees})`,
      });
    }
    if (ownership && event.route) {
      this.#updateChargeOwnershipGeometry(ownership, event);
    }
    const impact = group.querySelector(".combat-impact");
    if (impact && (event.route?.end || event.target)) {
      const anchor = event.route?.end ?? event.target;
      setAttributes(impact, {
        transform: impactTransform(event, anchor),
      });
    }
    const local = group.querySelector(".combat-local");
    const anchor = event.source;
    if (local && anchor) {
      setAttributes(local, {
        transform: `translate(${anchor.x} ${anchor.y})`,
      });
    }
    const particle = underlay?.querySelector(".combat-route__particle");
    if (particle instanceof SVGElement && event.route) {
      particle.style.offsetPath = `path("${event.route.path}")`;
    }
  }

  /**
   * Keep a Charge ownership pill associated with its route while respecting
   * the collision-aware plan. The leader is only needed when dense geometry
   * displaces the pill away from its route anchor.
   *
   * @param {Element} ownership
   * @param {JsonRecord} event
   */
  #updateChargeOwnershipGeometry(ownership, event) {
    const cue = event.ownershipCue;
    const anchor = event.ownershipAnchor;
    const rendered =
      event.ownershipCueCollisionFree === true &&
      event.ownershipSpatialDisposition === "rendered" &&
      cue &&
      anchor;
    ownership.setAttribute(
      "data-layout-collision-free",
      String(event.ownershipCueCollisionFree === true),
    );
    ownership.setAttribute(
      "data-spatial-disposition",
      rendered ? "rendered" : "suppressed-collision",
    );
    if (!rendered) {
      ownership.setAttribute("visibility", "hidden");
      return;
    }
    ownership.removeAttribute("visibility");
    setAttributes(ownership, {
      transform: `translate(${cue.x} ${cue.y})`,
    });

    const leader = ownership.querySelector(".combat-route__ownership-leader");
    if (!(leader instanceof SVGElement)) {
      return;
    }
    const deltaX = anchor.x - cue.x;
    const deltaY = anchor.y - cue.y;
    const distance = Math.hypot(deltaX, deltaY);
    if (distance <= 4) {
      leader.setAttribute("visibility", "hidden");
      return;
    }
    const unitX = deltaX / distance;
    const unitY = deltaY / distance;
    const horizontalScale =
      Math.abs(unitX) > Number.EPSILON ? 34 / Math.abs(unitX) : Infinity;
    const verticalScale =
      Math.abs(unitY) > Number.EPSILON ? 9 / Math.abs(unitY) : Infinity;
    const edgeScale = Math.min(horizontalScale, verticalScale);
    setAttributes(leader, {
      visibility: "visible",
      x1: deltaX,
      y1: deltaY,
      x2: unitX * edgeScale,
      y2: unitY * edgeScale,
    });
  }

  /**
   * @param {SVGElement} group
   * @param {JsonRecord} event
   */
  #updateNetGeometry(group, event) {
    const label = group.querySelector(".combat-net__label");
    const recipientLabel = group.querySelector(".combat-net__recipient");
    if (!event.recipient) {
      return;
    }
    group.dataset.layoutCollisionFree = String(event.cueCollisionFree !== false);
    const x = event.cue?.x ?? event.recipient.x;
    const y = event.cue?.y ?? event.recipient.y - 32 - event.lane * 18;
    if (recipientLabel) {
      setAttributes(recipientLabel, {
        x,
        y: y - 10,
      });
    }
    if (label) {
      setAttributes(label, {
        x,
        y: y + 6,
      });
    }
    this.#updateCueLeader(group, event, 24);
  }

  /**
   * Retain collision-suppressed outcome nodes so resize reprojection can reveal
   * them without replacing the event subtree or restarting its animation.
   *
   * @param {SVGElement} group
   * @param {Record<string, any>} event
   */
  #applySpatialDisposition(group, event) {
    const disposition =
      event.spatialDisposition === "suppressed_collision"
        ? "suppressed-collision"
        : "rendered";
    group.dataset.spatialDisposition = disposition;
    if (disposition === "suppressed-collision") {
      group.setAttribute("visibility", "hidden");
      group.setAttribute("aria-hidden", "true");
      return;
    }
    group.removeAttribute("visibility");
    group.removeAttribute("aria-hidden");
  }

  /**
   * Keep displaced recipient cues visibly associated without covering their
   * glyph or label.
   *
   * @param {SVGElement} group
   * @param {JsonRecord} event
   * @param {number} cueGap
   */
  #updateCueLeader(group, event, cueGap) {
    const leader = group.querySelector(".combat-cue__leader");
    const recipientAnchor = group.querySelector(
      ".combat-net__recipient-anchor, .combat-regeneration__recipient-anchor",
    );
    const recipient = event.recipient;
    const cue = event.cue;
    if (!leader || !recipient || !cue) {
      leader?.removeAttribute("x1");
      leader?.removeAttribute("x2");
      recipientAnchor?.setAttribute("visibility", "hidden");
      return;
    }
    const deltaX = cue.x - recipient.x;
    const deltaY = cue.y - recipient.y;
    const distance = Math.hypot(deltaX, deltaY);
    const recipientGap = 24;
    if (distance <= cueGap + recipientGap) {
      leader.setAttribute("visibility", "hidden");
      recipientAnchor?.setAttribute("visibility", "hidden");
      return;
    }
    const unitX = deltaX / distance;
    const unitY = deltaY / distance;
    setAttributes(leader, {
      visibility: "visible",
      x1: recipient.x + unitX * recipientGap,
      y1: recipient.y + unitY * recipientGap,
      x2: cue.x - unitX * cueGap,
      y2: cue.y - unitY * cueGap,
    });
    if (recipientAnchor) {
      setAttributes(recipientAnchor, {
        visibility: "visible",
        cx: recipient.x + unitX * recipientGap,
        cy: recipient.y + unitY * recipientGap,
      });
    }
  }

  /**
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #updateChargeGeometry(group, underlay, event) {
    if (!event.start || !event.end) {
      return;
    }
    const path = underlay?.querySelector(".combat-charge__path");
    const start = group.querySelector(".combat-charge__endpoint--start");
    const end = group.querySelector(".combat-charge__endpoint--end");
    const direction = group.querySelector(".combat-charge__direction");
    if (path) {
      path.setAttribute(
        "d",
        `M ${event.start.x} ${event.start.y} L ${event.end.x} ${event.end.y}`,
      );
    }
    if (start) {
      setAttributes(start, { cx: event.start.x, cy: event.start.y });
    }
    if (end) {
      setAttributes(end, { cx: event.end.x, cy: event.end.y });
    }
    if (direction) {
      const deltaX = event.end.x - event.start.x;
      const deltaY = event.end.y - event.start.y;
      const distance = Math.hypot(deltaX, deltaY);
      const unitX = distance > 0 ? deltaX / distance : 1;
      const unitY = distance > 0 ? deltaY / distance : 0;
      const markerX = event.end.x - unitX * 12;
      const markerY = event.end.y - unitY * 12;
      const angle = (Math.atan2(deltaY, deltaX) * 180) / Math.PI;
      setAttributes(direction, {
        transform: `translate(${markerX} ${markerY}) rotate(${angle})`,
      });
    }
  }
}

/**
 * @param {Document} ownerDocument
 * @param {string} tagName
 * @param {Record<string, string | number | boolean | null | undefined>} attributes
 */
function svgElement(ownerDocument, tagName, attributes = {}) {
  const element = ownerDocument.createElementNS(SVG_NAMESPACE, tagName);
  setAttributes(element, attributes);
  return element;
}

/**
 * @param {Element} element
 * @param {Record<string, string | number | boolean | null | undefined>} attributes
 */
function setAttributes(element, attributes) {
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) {
      element.removeAttribute(name);
    } else {
      element.setAttribute(name, String(value));
    }
  }
}

/**
 * @param {SVGElement} group
 * @param {string} role
 * @param {unknown} value
 */
function assignSlot(group, role, value) {
  if (Number.isInteger(value)) {
    group.dataset[`${role}Slot`] = String(value);
  }
}

/**
 * @param {SVGElement} group
 * @param {string} role
 * @param {unknown} value
 */
function assignPresentationKey(group, role, value) {
  if (typeof value === "string" && value) {
    group.dataset[`${role}PresentationKey`] = value;
  }
}

/**
 * Copy only inert event metadata to the route underlay. Presentation geometry
 * remains split while semantic inspection retains one keyed identity.
 *
 * @param {SVGElement} source
 * @param {SVGElement} target
 */
function copyEventMetadata(source, target) {
  for (const attribute of source.attributes) {
    if (attribute.name.startsWith("data-")) {
      target.setAttribute(attribute.name, attribute.value);
    }
  }
}

/**
 * @param {unknown} value
 */
function cssIdentifier(value) {
  return typeof value === "string" && /^[a-z0-9_]+$/.test(value)
    ? value.replaceAll("_", "-")
    : "unknown";
}

/**
 * Render one intentionally non-numeric recipient mark. Damage and healing use
 * the universal minus/plus grammar; status-only or unknown activations fail
 * closed to a neutral diamond.
 *
 * @param {Document} ownerDocument
 * @param {unknown} value
 * @returns {SVGElement}
 */
function semanticImpactGlyph(ownerDocument, value) {
  const semantic = value === "damage" || value === "healing" ? value : "neutral";
  const group = svgElement(ownerDocument, "g", {
    class: `combat-impact__semantic combat-impact__semantic--${semantic}`,
  });
  if (semantic === "neutral") {
    group.append(
      svgElement(ownerDocument, "path", {
        d: "M 0 -6 L 6 0 0 6 -6 0 Z",
      }),
    );
    return group;
  }
  group.append(
    svgElement(ownerDocument, "line", {
      x1: -6,
      y1: 0,
      x2: 6,
      y2: 0,
    }),
  );
  if (semantic === "healing") {
    group.append(
      svgElement(ownerDocument, "line", {
        x1: 0,
        y1: -6,
        x2: 0,
        y2: 6,
      }),
    );
  }
  return group;
}

/**
 * Keep the largest Ultimate ornaments outside the body they identify. Route
 * planning supplies an exterior impact port; this local scale bounds only the
 * transient flare and does not alter the authoritative source or recipient.
 *
 * @param {Record<string, any>} event
 * @param {Record<string, any>} anchor
 */
function impactTransform(event, anchor) {
  const scale =
    (event.tokenId === "basic_damage" || event.tokenId === "basic_heal") &&
    Number(event.routeMultiplicity) >= 4
      ? 0.48
      : event.tokenId === "hunter_trap"
        ? 0.5
        : event.tokenId === "warrior_charge"
          ? 0.58
          : 1;
  return `translate(${anchor.x} ${anchor.y}) scale(${scale})`;
}

/**
 * @param {Record<string, any>} event
 */
function phaseFor(event) {
  if (event.kind === "regeneration") {
    return "health_regenerated";
  }
  if (event.kind === "semantic_pulse") {
    return String(event.cueSemantic ?? "semantic");
  }
  if (event.kind === "net_health" || event.kind === "status_lifecycle") {
    return "outcome";
  }
  if (
    event.kind === "charge_displacement" ||
    (event.kind === "activation" && event.presentationKind === "target_only_impact")
  ) {
    return "impact";
  }
  return "activation";
}

/**
 * @param {number} delta
 * @param {string} outcome
 */
function netLabel(delta, outcome) {
  if (outcome === "unchanged" || delta === 0) {
    return "HP unchanged";
  }
  const magnitude = formatDisplayNumber(Math.abs(delta));
  const visibleMagnitude = magnitude === "0" ? "<0.01" : magnitude;
  return delta < 0 ? `NET −${visibleMagnitude}` : `NET +${visibleMagnitude}`;
}

/**
 * Plans built before local visual filters existed remain all-on at this narrow
 * rendering boundary. Current plans carry an exact frozen boolean part map;
 * an omitted key in that map fails closed.
 *
 * @param {JsonRecord} event
 * @param {string} part
 */
function paintPartEnabled(event, part) {
  if (!event.paintParts || typeof event.paintParts !== "object") {
    return true;
  }
  return event.paintParts[part] === true;
}

/**
 * Keep tooltip semantics on the same side of a multipart filter gate as their
 * visible owner. Scientific event identity remains on the original plan row;
 * this shallow view exists only for explanatory copy.
 *
 * @param {JsonRecord} event
 */
function paintAwareExplanationEvent(event) {
  if (event.kind === "activation" && !paintPartEnabled(event, "ability")) {
    return {
      ...event,
      eventType: `${event.impactSemantic ?? "activation"}_effect`,
      token: null,
      tokenId: null,
    };
  }
  if (event.kind === "net_health" && !paintPartEnabled(event, "recipientText")) {
    return {
      ...event,
      recipientPublicAgentId: null,
    };
  }
  if (
    event.kind === "status_lifecycle" &&
    event.lifecycle === "trap_broken_and_reapplied"
  ) {
    const breakEnabled = paintPartEnabled(event, "break");
    const reapplicationEnabled = paintPartEnabled(event, "reapplication");
    const visibleLifecycle = breakEnabled
      ? reapplicationEnabled
        ? event.lifecycle
        : "trap_broken"
      : "reapplied";
    if (breakEnabled && !reapplicationEnabled) {
      return {
        ...event,
        eventType: "status_broken_by_damage",
        lifecycle: visibleLifecycle,
        lifecycleToken: resolveVisualToken("lifecycle", visibleLifecycle, event),
        sourcePresentationKey: null,
        sourcePublicAgentId: null,
        applicationSources: Object.freeze([]),
      };
    }
    return {
      ...event,
      lifecycle: visibleLifecycle,
      lifecycleToken: resolveVisualToken("lifecycle", visibleLifecycle, event),
    };
  }
  return event;
}

/**
 * @param {JsonRecord} event
 * @param {"normal" | "reduced" | "off"} motionMode
 * @returns {Keyframe[]}
 */
function eventKeyframes(event, motionMode) {
  if (
    event.kind === "semantic_pulse" &&
    event.cueSemantic === "spawn_shield_expired" &&
    motionMode === "normal"
  ) {
    return [
      { opacity: 0 },
      { opacity: 1, offset: 0.14 },
      { opacity: 1, offset: 0.44 },
      { opacity: 0.35, offset: 0.62 },
      { opacity: 0.75, offset: 0.72 },
      { opacity: 0 },
    ];
  }
  if (event.kind === "net_health" && motionMode === "normal") {
    return [
      { opacity: 0 },
      { opacity: 1, offset: 0.18 },
      { opacity: 1, offset: 0.72 },
      { opacity: 0 },
    ];
  }
  return [
    { opacity: 0 },
    { opacity: 1, offset: 0.18 },
    { opacity: 1, offset: event.persistent ? 1 : 0.72 },
    { opacity: event.persistent ? 1 : 0 },
  ];
}

/**
 * @param {Element} element
 * @param {Keyframe[] | PropertyIndexedKeyframes} keyframes
 * @param {KeyframeAnimationOptions} options
 * @param {Record<string, any>} plan
 * @param {Record<string, any>} event
 * @param {string} part
 */
function animationSpec(element, keyframes, options, plan, event, part) {
  return Object.freeze({
    element,
    keyframes,
    options: Object.freeze(options),
    id: `mbg:${plan.epochKey}:${event.eventId}:${part}`,
  });
}
