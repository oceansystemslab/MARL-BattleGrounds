/**
 * @file Paint authorized combat plans into retained SVG and animation descriptions.
 * SvgChoreographyPainter owns its three installation roots: foreground events,
 * connectors and route underlays. It creates and updates DOM geometry, registers
 * semantic tooltips and returns animation specs for CombatChoreographer to run.
 * All event identity, positions, phases and layout come from the supplied plan;
 * this module does not query simulator state, infer hidden bodies or submit actions.
 */
import {
  canonicalAgentIdentity,
  exactAuthorizedAgentIdentityV1,
} from "./agent-identity.js";
import { CHOREOGRAPHY_PAINT_FOOTPRINTS } from "./choreography-plan.js";
import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import {
  explainActivation,
  explainDeathAnnouncement,
  explainNetHealth,
} from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import { routeMarkerPose } from "./routes.js";
import { statusLifecyclePresentation } from "./semantic-vocabulary.js";
import { createSemanticDescriptor, registerTooltipOwner } from "./tooltip.js";
import { resolveVisualToken } from "./vocabulary.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * Format a public agent label through the shared identity authority.
 *
 * publicAgentId is the fallback; identity defaults to null and, when supplied,
 * takes precedence as the complete identity input. Return publicIdentity text.
 * Internal slot numbers are never used as the fallback display identity.
 *
 * @param {unknown} publicAgentId
 */
function formatAgentIdentity(publicAgentId, identity = null) {
  return canonicalAgentIdentity(identity ?? { public_agent_id: publicAgentId })
    .publicIdentity;
}

/**
 * Return the exact authorized public identity title, or null.
 *
 * value is checked by exactAuthorizedAgentIdentityV1. Missing or invalid identity
 * cannot be replaced with an internal slot or a guessed source.
 *
 * @param {unknown} value
 */
function authorizedIdentityTitle(value) {
  return exactAuthorizedAgentIdentityV1(value)?.title ?? null;
}

/**
 * Turn a machine event name into a simple title.
 *
 * value defaults to event when nullish, is stringified, then has underscores
 * replaced and word initials capitalized. This fallback adds no event facts.
 *
 * @param {unknown} value
 */
function humanizeEventName(value) {
  return String(value ?? "event")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * Return frozen title/summary text for an already admitted event.
 *
 * event.cueSemantic selects death, respawn, team wave or finite regeneration text.
 * Other values use a humanized eventType and generic incoming-transition sentence.
 * No scientific event is detected or reconstructed here.
 *
 * @param {Record<string, any>} event
 */
function semanticEventCopy(event) {
  if (event.cueSemantic === "agent_died") {
    return Object.freeze({
      title: "Agent Died",
      summary: "This agent died on the incoming transition.",
    });
  }
  if (event.cueSemantic === "agent_respawned") {
    return Object.freeze({
      title: "Agent Respawned",
      summary: "This agent respawned on the incoming transition.",
    });
  }
  if (event.cueSemantic === "respawn_wave_occurred") {
    return Object.freeze({
      title:
        typeof event.label === "string" && event.label.trim()
          ? event.label
          : "Team Respawn",
      summary: "This team respawn occurred on the incoming transition.",
    });
  }
  if (event.cueSemantic === "health_regenerated" && Number.isFinite(event.value)) {
    return Object.freeze({
      title: "Health Regenerated",
      summary: `The recipient regenerated ${formatDisplayNumber(event.value)} health while out of combat.`,
    });
  }
  return Object.freeze({
    title: humanizeEventName(event.eventType),
    summary: "This event occurred on the incoming transition.",
  });
}

/**
 * Build a semantic tooltip from already authorized plan fields.
 *
 * event supplies identity, kind/tokens, lifecycle and optional application sources.
 * Apply the visible paint-part view first. Death announcements use their dedicated
 * explanation. Include source attribution only when the supplied exact public
 * identities are valid; multiple applications require all source titles to be known.
 * Return the shared semantic descriptor with title, summary and source/recipient
 * rows. Slot IDs may key DOM records but never supply display names. The input is
 * unchanged; shared descriptor/identity validation errors may propagate.
 *
 * @param {Record<string, any>} event
 */
export function explainChoreographyEvent(event) {
  const visibleEvent = paintAwareExplanationEvent(event);
  if (visibleEvent.cueSemantic === "death_announcement") {
    return explainDeathAnnouncement(visibleEvent);
  }
  const semanticCopy = semanticEventCopy(visibleEvent);
  const lifecycleCopy =
    visibleEvent.kind === "status_lifecycle"
      ? statusLifecyclePresentation(visibleEvent.tokenId, visibleEvent.lifecycle)
      : null;
  const title = String(
    lifecycleCopy?.title ??
      visibleEvent.lifecycleToken?.label ??
      visibleEvent.token?.label ??
      semanticCopy.title,
  );
  const rows = [];
  const applicationSources =
    visibleEvent.lifecycle === "applied" &&
    Array.isArray(visibleEvent.applicationSources)
      ? visibleEvent.applicationSources
      : [];
  const applicationSourceIdentities = applicationSources.map((source) =>
    authorizedIdentityTitle(source?.sourceIdentity),
  );
  const distinctApplicationSourceIdentities = [...new Set(applicationSourceIdentities)];
  if (
    distinctApplicationSourceIdentities.length > 0 &&
    applicationSourceIdentities.every((identity) => identity !== null)
  ) {
    rows.push({
      label: distinctApplicationSourceIdentities.length === 1 ? "Source" : "Sources",
      value: distinctApplicationSourceIdentities.join("; "),
      metadata: { compact: true, full: true },
    });
  }
  const directSourceIdentity =
    visibleEvent.kind === "activation" && applicationSources.length === 0
      ? authorizedIdentityTitle(visibleEvent.sourceIdentity)
      : null;
  const recipientIdentity = authorizedIdentityTitle(visibleEvent.recipientIdentity);
  for (const [label, value] of [
    ["Source", directSourceIdentity],
    ["Recipient", recipientIdentity],
  ]) {
    if (value !== null) {
      rows.push({
        label,
        value,
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
  const regenerationPerTick = Number(visibleEvent.outOfCombatRegenerationPerTick);
  const summary =
    visibleEvent.kind === "status_lifecycle" &&
    visibleEvent.tokenId === "in_combat" &&
    visibleEvent.lifecycle === "expired" &&
    Number.isFinite(regenerationPerTick)
      ? `This agent can regenerate up to ${formatDisplayNumber(regenerationPerTick)} health per tick while it remains out of combat.`
      : lifecycleCopy !== null
        ? lifecycleCopy.summary
        : String(
            visibleEvent.lifecycleToken?.accessibleName ??
              visibleEvent.token?.accessibleName ??
              semanticCopy.summary,
          );
  return createSemanticDescriptor({
    kind: "event",
    id: `semantic-event:${semanticOwnerKey || "unclassified"}`,
    title,
    tone: "information",
    accent: "none",
    summary,
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
 *   renderPolicy: "live_once" | "replay_animated" | "replay_static",
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
 *   connectorRoot: SVGElement,
 *   routeRoot: SVGElement,
 *   eventNodes: Map<
 *     string,
 *     {
 *       group: SVGElement,
 *       connector: SVGElement | null,
 *       underlay: SVGElement | null,
 *       event: JsonRecord,
 *     }
 *   >,
 *   animationSpecs: readonly AnimationSpec[],
 *   nodeCount: number,
 *   persistentNodeCount: number,
 *   motionMode: "normal" | "reduced" | "off",
 *   renderPolicy: "live_once" | "replay_animated" | "replay_static",
 *   retainTransientOnSettle: boolean,
 * }} PainterInstallation
 */

/**
 * Own retained SVG for one controller's authorized combat explanations.
 *
 * install appends three owned roots and returns their mutable installation record.
 * settle keeps persistent or explicitly retained explanations; clear removes only
 * these roots. reproject updates supported geometry in place so browser animations
 * keep their element identity. The controller owns actual animation handles/clocks.
 */
export class SvgChoreographyPainter {
  /**
   * Build and append one bounded SVG installation from an authorized plan.
   *
   * plan supplies ordered events, identity, timing and resource bounds. surface gives
   * ownerDocument, foreground layer, optional routeLayer (defaults to layer) and
   * viewportKey. options supplies motionMode, renderPolicy, settled and persistentOnly;
   * retainTransientOnSettle is honored only when initially settled.
   *
   * Skip nonspatial/unknown rows and, in persistent-only mode, nonpersistent rows.
   * Return a mutable installation with its roots, event map, frozen animation specs
   * and counted nodes. Specs are not started here. Exceeding planned node/animation
   * bounds throws RangeError before the roots are appended. DOM/descriptor errors
   * propagate. The caller owns clear/settle and must remove an old authority before
   * installing a replacement.
   *
   * @param {JsonRecord} plan
   * @param {JsonRecord} surface
   * @param {PainterOptions} options
   * @returns {PainterInstallation}
   */
  install(plan, surface, options) {
    const root = svgElement(surface.ownerDocument, "g", {
      class: "combat-choreography",
      role: "group",
      "aria-label": "Authorized combat event summaries",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-render-policy": options.renderPolicy,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    const routeRoot = svgElement(surface.ownerDocument, "g", {
      class: "combat-choreography-routes",
      role: "group",
      "aria-label": "Authorized combat event routes",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-render-policy": options.renderPolicy,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    const connectorRoot = svgElement(surface.ownerDocument, "g", {
      class: "combat-choreography-connectors",
      "aria-hidden": "true",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-render-policy": options.renderPolicy,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    /** @type {Map<
     *   string,
     *   {
     *     group: SVGElement,
     *     connector: SVGElement | null,
     *     underlay: SVGElement | null,
     *     event: JsonRecord,
     *   }
     * >} */
    const eventNodes = new Map();
    /** @type {AnimationSpec[]} */
    const animationSpecs = [];

    for (const event of plan.events) {
      if (
        !event.spatial ||
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
      const { group, connector, underlay } = rendered;
      eventNodes.set(event.eventId, { group, connector, underlay, event });
      if (connector) {
        connectorRoot.append(connector);
      }
      if (underlay) {
        routeRoot.append(underlay);
      }
      root.append(group);
    }
    const nodeCount =
      root.querySelectorAll("*").length +
      connectorRoot.querySelectorAll("*").length +
      routeRoot.querySelectorAll("*").length +
      3;
    const persistentNodeCount = Array.from(eventNodes.values())
      .filter(({ event }) => event.persistent)
      .reduce(
        (count, { group, connector, underlay }) =>
          count +
          group.querySelectorAll("*").length +
          1 +
          (connector ? connector.querySelectorAll("*").length + 1 : 0) +
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
    (surface.routeLayer ?? surface.layer).append(connectorRoot, routeRoot);
    surface.layer.append(root);
    /** @type {PainterInstallation} */
    const installation = {
      root,
      connectorRoot,
      routeRoot,
      eventNodes,
      animationSpecs: Object.freeze(animationSpecs),
      nodeCount,
      persistentNodeCount,
      motionMode: options.motionMode,
      renderPolicy: options.renderPolicy,
      retainTransientOnSettle:
        options.settled && options.retainTransientOnSettle === true,
    };
    if (options.settled) {
      this.settle(installation);
    }
    return installation;
  }

  /**
   * Remove only the three roots owned by installation.
   *
   * installation may be null; _reason is an optional compatibility argument and is
   * unused. Return undefined. This removes DOM but does not cancel animation handles
   * or clear the installation object; the controller owns those resources.
   *
   * @param {PainterInstallation | null} installation
   * @param {string} [_reason]
   */
  clear(installation, _reason) {
    installation?.root?.remove();
    installation?.connectorRoot?.remove();
    installation?.routeRoot?.remove();
  }

  /**
   * Mark this installation settled and remove its nonretained transient nodes.
   *
   * installation is the mutable result of install. Persistent events, or all events
   * when retainTransientOnSettle is true, become fully opaque with settled metadata.
   * Other event groups/connectors/underlays are removed from DOM and eventNodes.
   * Return undefined; animation handles and original count/spec fields are unchanged.
   *
   * @param {PainterInstallation} installation
   */
  settle(installation) {
    installation.root.dataset.state = "settled";
    installation.connectorRoot.dataset.state = "settled";
    installation.routeRoot.dataset.state = "settled";
    for (const [eventId, record] of installation.eventNodes) {
      const group = record.group;
      const persistent = group.dataset.persistent === "true";
      if (persistent || installation.retainTransientOnSettle) {
        group.dataset.settled = "true";
        group.setAttribute("opacity", "1");
        if (record.connector) {
          record.connector.dataset.settled = "true";
          record.connector.setAttribute("opacity", "1");
        }
        if (record.underlay) {
          record.underlay.dataset.settled = "true";
          record.underlay.setAttribute("opacity", "1");
        }
      } else {
        group.remove();
        record.connector?.remove();
        record.underlay?.remove();
        installation.eventNodes.delete(eventId);
      }
    }
  }

  /**
   * Apply a new layout to retained event elements without restarting clocks.
   *
   * installation is active, plan describes the same authorized content with updated
   * geometry, and surface provides the new viewportKey. Update viewport metadata and
   * supported event geometry; remove existing events that are now absent/nonspatial.
   * No new event subtree or animation is installed here. Return undefined. The caller
   * owns identity compatibility and any rebuild needed for changed content.
   *
   * @param {PainterInstallation} installation
   * @param {JsonRecord} plan
   * @param {JsonRecord} surface
   */
  reproject(installation, plan, surface) {
    installation.root.dataset.viewportKey = surface.viewportKey;
    installation.connectorRoot.dataset.viewportKey = surface.viewportKey;
    installation.routeRoot.dataset.viewportKey = surface.viewportKey;
    const nextEvents = /** @type {JsonRecord[]} */ (plan.events);
    const nextById = new Map(nextEvents.map((event) => [event.eventId, event]));
    for (const [eventId, record] of installation.eventNodes) {
      const next = nextById.get(eventId);
      if (!next?.spatial) {
        record.group.remove();
        record.connector?.remove();
        record.underlay?.remove();
        installation.eventNodes.delete(eventId);
        continue;
      }
      this.#updateGeometry(record.group, record.connector, record.underlay, next);
      record.event = next;
    }
  }

  /**
   * Create one event subtree and append its animation descriptions.
   *
   * ownerDocument creates SVG; event is an admitted spatial plan row; plan supplies
   * phases; options supplies motion/render/settlement choices; animationSpecs is the
   * caller's mutable output array. Return {group, connector, underlay}, with null
   * unused layers, or null for unsupported/cooldown-start rows. Register matching
   * semantic owners and layout metadata. Reduced motion rescales phase times; settled
   * and off modes add no generic fade specs. No Web Animation is started here.
   *
   * @param {Document} ownerDocument
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   * @returns {{
   *   group: SVGElement,
   *   connector: SVGElement | null,
   *   underlay: SVGElement | null,
   * } | null}
   */
  #renderEvent(ownerDocument, event, plan, options, animationSpecs) {
    if (event.kind === "semantic_pulse" && event.cueSemantic === "cooldown_started") {
      return null;
    }
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
    const connector = svgElement(ownerDocument, "g", {
      class: `combat-connector-effect combat-connector-effect--${cssIdentifier(event.kind)}`,
      "aria-hidden": "true",
      opacity: options.settled || options.motionMode === "off" ? 1 : 0,
    });
    copyEventMetadata(group, connector);
    const underlay = event.route
      ? svgElement(ownerDocument, "g", {
          class: `combat-route-effect combat-route-effect--${cssIdentifier(event.kind)}`,
          "aria-hidden": "true",
          opacity: options.settled || options.motionMode === "off" ? 1 : 0,
        })
      : null;
    if (underlay) {
      copyEventMetadata(group, underlay);
      assignLayoutKey(underlay, event.routeLayoutKey ?? event.route?.layoutKey);
      if (Number.isInteger(event.routeLane ?? event.route?.lane)) {
        underlay.dataset.lane = String(event.routeLane ?? event.route.lane);
      }
    }

    if (event.kind === "activation") {
      this.#renderActivation(
        ownerDocument,
        group,
        connector,
        underlay,
        event,
        plan,
        options,
        animationSpecs,
      );
    } else if (event.kind === "net_health") {
      this.#renderNet(ownerDocument, group, connector, event);
    } else if (event.kind === "regeneration") {
      this.#renderRegeneration(ownerDocument, group, connector, event);
    } else if (event.kind === "status_lifecycle") {
      this.#renderLifecycle(ownerDocument, group, connector, event);
    } else if (event.kind === "semantic_pulse") {
      this.#renderSemanticPulse(
        ownerDocument,
        group,
        connector,
        event,
        plan,
        options,
        animationSpecs,
      );
    } else {
      return null;
    }
    this.#syncRouteBridgeGaps(underlay, event.route);
    this.#registerEventExplanation(group, underlay, event);
    this.#applySpatialDisposition(group, event);
    const renderedConnector = connector.childElementCount > 0 ? connector : null;

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
      if (renderedConnector) {
        targets.push({ element: renderedConnector, part: "connector" });
      }
      if (event.kind === "activation" && (underlay || activationHasImpact)) {
        group.setAttribute("opacity", "1");
      }
      const duration = phaseEnd - phaseStart;
      if (!(duration > 0)) {
        return { group, connector: renderedConnector, underlay };
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
    return { group, connector: renderedConnector, underlay };
  }

  /**
   * Paint an accepted activation as a route, target impact or source-local cue.
   *
   * ownerDocument creates nodes in group, connector and optional underlay. event owns
   * anchors, route, paintParts, token and allocation metadata. plan supplies timing;
   * options selects motion/settlement; append specs to animationSpecs when needed.
   * Normal unsettled routes may add a moving particle. Charge labels use public IDs;
   * source-local Mage Burst gets its registered wave. Disabled/missing geometry is
   * omitted. Return undefined; do not infer a target, route or game outcome.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement} connector
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #renderActivation(
    ownerDocument,
    group,
    connector,
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
      const markerCount =
        event.tokenId === "warrior_charge" &&
        Array.isArray(event.route.markerProgresses)
          ? event.route.markerProgresses.length
          : 1;
      const arrows = Array.from({ length: markerCount }, (_, markerIndex) =>
        svgElement(ownerDocument, "path", {
          class: "combat-route__arrow",
          d: "M -11 -6 L 2 0 L -11 6 L -7 0 Z",
          "data-marker-index": markerIndex,
        }),
      );
      underlay.append(hitPath, path, ...arrows);
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
        assignLayoutPlacement(
          ownership,
          event.ownershipLayoutKey,
          event.ownershipBounds,
          event.ownershipDisposition,
          event.ownershipCueCollisionFree,
        );
        const ownershipLabel = `${formatAgentIdentity(event.sourcePublicAgentId, event.sourceIdentity)} → ${formatAgentIdentity(event.targetPublicAgentId, event.recipientIdentity)}`;
        ownership.append(
          svgElement(ownerDocument, "line", {
            class: "combat-route__ownership-leader",
            "aria-hidden": "true",
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
            textLength: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeOwnership.labelLength,
            lengthAdjust: "spacingAndGlyphs",
          }),
        );
        const label = ownership.lastElementChild;
        if (label) {
          label.textContent = ownershipLabel;
        }
        underlay.append(ownership);
      }
      if (options.motionMode === "normal" && !options.settled) {
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
      const impact = this.#appendImpact(
        ownerDocument,
        group,
        event.impactCue ?? event.route.end,
        event,
      );
      if (impact) {
        assignLayoutPlacement(
          impact,
          event.impactLayoutKey,
          event.impactBounds,
          event.impactDisposition,
          event.impactCueCollisionFree,
        );
        appendAllocatedLeader(
          ownerDocument,
          connector,
          "combat-cue__leader combat-cue__leader--impact",
          event.impactLeader,
        );
      }
      this.#animateImpact(impact, event, plan, options, animationSpecs);
      this.#updateActivationGeometry(group, connector, underlay, event);
      return;
    }
    if (event.presentationKind === "target_only_impact") {
      const impact = this.#appendImpact(
        ownerDocument,
        group,
        event.impactCue ?? event.target,
        event,
      );
      if (impact) {
        assignLayoutPlacement(
          impact,
          event.impactLayoutKey,
          event.impactBounds,
          event.impactDisposition,
          event.impactCueCollisionFree,
        );
        appendAllocatedLeader(
          ownerDocument,
          connector,
          "combat-cue__leader combat-cue__leader--impact",
          event.impactLeader,
        );
      }
      this.#animateImpact(impact, event, plan, options, animationSpecs);
      this.#updateActivationGeometry(group, connector, underlay, event);
      return;
    }
    if (!abilityEnabled) {
      return;
    }
    const anchor = event.sourceCue ?? event.source;
    if (!anchor) {
      return;
    }
    const local = svgElement(ownerDocument, "g", {
      class: `combat-local combat-local--${cssIdentifier(event.tokenId)}`,
    });
    local.append(
      svgElement(ownerDocument, "circle", {
        class: "combat-local__hit",
        r:
          (event.sourceBounds?.width ??
            CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.local.width) / 2,
        "aria-hidden": "true",
      }),
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
      const flareExtent =
        CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.mage_burst.flareExtent;
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
          d: `M 0 ${-flareExtent} V -25 M 0 25 V ${flareExtent} M ${-flareExtent} 0 H -25 M 25 0 H ${flareExtent} M -24 -24 L -18 -18 M 18 18 L 24 24 M 24 -24 L 18 -18 M -18 18 L -24 24`,
        }),
      );
    }
    const icon = createSvgIcon(ownerDocument, event.token?.glyphKey ?? "unknown", {
      className: "combat-local__icon",
    });
    setAttributes(icon, { x: -10, y: -10, width: 20, height: 20 });
    local.append(icon);
    group.append(local);
    assignLayoutPlacement(
      local,
      event.sourceLayoutKey,
      event.sourceBounds,
      event.sourceDisposition,
      event.sourceCueCollisionFree,
    );
    appendAllocatedLeader(
      ownerDocument,
      connector,
      "combat-cue__leader combat-cue__leader--source",
      event.sourceLeader,
    );
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
   * Append enabled ability/semantic marks at a supplied impact anchor.
   *
   * ownerDocument creates SVG under group. target is a screen point or null, and event
   * supplies token/paint choices. Return the new impact group, or null when the anchor
   * is absent or both pieces are disabled. Shapes use registered footprints and the
   * presentation-only impactTransform; no event or anchor is inferred.
   *
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
    impact.append(
      svgElement(ownerDocument, "circle", {
        class: "combat-impact__hit",
        r: abilityEnabled ? 22 : 12,
        "aria-hidden": "true",
      }),
    );
    if (abilityEnabled) {
      const compact =
        event.tokenId === "basic_damage" || event.tokenId === "basic_heal";
      impact.append(
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
      const flareExtent =
        CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.holy_word.flareExtent;
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
          d: `M 0 ${-flareExtent} V -21 M 0 21 V ${flareExtent} M ${-flareExtent} 0 H -21 M 21 0 H ${flareExtent}`,
        }),
      );
    } else if (abilityEnabled && event.tokenId === "hunter_trap") {
      const flareExtent =
        CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.hunter_trap.flareExtent;
      impact.append(
        svgElement(ownerDocument, "path", {
          class: "combat-trap__lattice",
          d: "M -17 -17 H 17 V 17 H -17 Z M -17 -6 H 17 M -17 6 H 17 M -6 -17 V 17 M 6 -17 V 17",
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-trap__flare",
          d: `M 0 ${-flareExtent} ${flareExtent} 0 0 ${flareExtent} ${-flareExtent} 0 Z`,
        }),
      );
    } else if (abilityEnabled && event.tokenId === "rogue_poison") {
      const dagger = createSvgIcon(ownerDocument, "activation-poison", {
        className: "combat-poison__dagger",
      });
      setAttributes(dagger, { x: -24, y: 2, width: 18, height: 18 });
      impact.append(
        dagger,
        svgElement(ownerDocument, "circle", {
          class: "combat-poison__splash",
          cx: 13,
          cy: -13,
          r: 3,
        }),
      );
    } else if (abilityEnabled && event.tokenId === "warrior_charge") {
      const flareExtent =
        CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.warrior_charge.flareExtent;
      impact.append(
        svgElement(ownerDocument, "path", {
          class: "combat-charge__impact",
          d: "M -22 0 H -13 M 13 0 H 22 M 0 -22 V -13 M 0 13 V 22",
        }),
        svgElement(ownerDocument, "path", {
          class: "combat-ultimate__flare combat-charge__flare",
          d: `M ${-flareExtent} -11 L -18 -7 M 18 7 L ${flareExtent} 11 M ${-flareExtent} 11 L -18 7 M 18 -7 L ${flareExtent} -11`,
        }),
      );
    }
    group.append(impact);
    setAttributes(impact, { transform: impactTransform(event, target) });
    return impact;
  }

  /**
   * Append an impact fade spec for a positive authored display interval.
   *
   * impact may be null; event/plan supply impact and phase-end milliseconds. options
   * selects normal/reduced/off and settled state; animationSpecs is mutated only when
   * an animation is needed. Reduced motion rescales the interval. Null, settled, off
   * or nonpositive intervals do nothing. Return undefined; no animation starts here.
   *
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
   * Bind matching semantic tooltips to a visible event and optional route underlay.
   *
   * group is the keyboard-focusable semantic owner; underlay may be null. event already
   * contains authorized facts and filter pieces. Choose activation, health or general
   * explanation accordingly, set accessible labels, and register both hit surfaces.
   * Return undefined. Identity/data validation errors propagate; no hidden lookup occurs.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #registerEventExplanation(group, underlay, event) {
    const explanationEvent = paintAwareExplanationEvent(event);
    const explanation =
      event.kind === "activation" && paintPartEnabled(event, "ability")
        ? explainActivation(explanationEvent)
        : event.kind === "net_health" &&
            (paintPartEnabled(event, "battleText") ||
              paintPartEnabled(event, "recipientText"))
          ? explainNetHealth(explanationEvent)
          : explainChoreographyEvent(explanationEvent);
    setAttributes(group, {
      role: "img",
      tabindex: "0",
      "aria-label": explanation.title,
    });
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
   * Paint selected health-change labels at an allocated recipient cue.
   *
   * ownerDocument creates nodes in group and connector. event supplies recipient,
   * netDelta/outcome, optional public identity, paintParts and allocator geometry.
   * Missing recipient does nothing. Add only enabled recipient/battle text, keep a hit
   * surface and update its bounds/leader through updateNetGeometry. Primary pointer
   * down on the hit region does not trigger a parent action. Return undefined.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement} connector
   * @param {JsonRecord} event
   */
  #renderNet(ownerDocument, group, connector, event) {
    if (!event.recipient) {
      return;
    }
    const effectEnabled = paintPartEnabled(event, "effect");
    const battleTextEnabled = paintPartEnabled(event, "battleText");
    const recipientTextEnabled = paintPartEnabled(event, "recipientText");
    const cueGeometryEnabled =
      effectEnabled || battleTextEnabled || recipientTextEnabled;
    group.dataset.netDelta = String(event.netDelta);
    const hit = svgElement(ownerDocument, "rect", {
      class: "combat-net__hit",
      "aria-hidden": "true",
    });
    hit.addEventListener("pointerdown", (pointerEvent) => {
      if (pointerEvent.button === 0) {
        pointerEvent.stopPropagation();
      }
    });
    group.append(hit);
    if (cueGeometryEnabled) {
      connector.append(
        svgElement(ownerDocument, "line", {
          class: "combat-cue__leader",
          "aria-hidden": "true",
        }),
      );
    }
    if (recipientTextEnabled) {
      const recipientLabel = svgElement(ownerDocument, "text", {
        class: "combat-net__recipient",
      });
      const fullRecipientLabel = formatAgentIdentity(
        event.recipientPublicAgentId,
        event.recipientIdentity,
      );
      recipientLabel.textContent = fullRecipientLabel;
      group.dataset.recipientLabel = fullRecipientLabel;
      group.append(recipientLabel);
    }
    if (battleTextEnabled) {
      const label = svgElement(ownerDocument, "text", {
        class: "combat-net__label",
      });
      label.textContent = netLabel(event.netDelta, event.outcome);
      group.append(label);
    }
    this.#updateNetGeometry(group, connector, event);
  }

  /**
   * Paint a recorded regeneration result beside its disclosed recipient.
   *
   * ownerDocument creates nodes in group/connector. event must supply recipient and
   * finite value; otherwise do nothing. Respect effect/battleText filters, use the
   * shared healing plus, and retain an allocated leader to the exact endpoint. Text
   * compression preserves full text content. Return undefined; no healer or movement
   * route is inferred.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement} connector
   * @param {JsonRecord} event
   */
  #renderRegeneration(ownerDocument, group, connector, event) {
    if (!event.recipient || !Number.isFinite(event.value)) {
      return;
    }
    const effectEnabled = paintPartEnabled(event, "effect");
    const battleTextEnabled = paintPartEnabled(event, "battleText");
    const cue = svgElement(ownerDocument, "g", {
      class: "combat-regeneration",
    });
    assignLayoutPlacement(
      cue,
      event.cueLayoutKey,
      event.cueBounds,
      event.cueDisposition,
      event.cueCollisionFree,
    );
    const cueWidth = Number(event.cueBounds?.width ?? 0);
    const cueHeight = Number(event.cueBounds?.height ?? 0);
    const hit = svgElement(ownerDocument, "rect", {
      class: "combat-regeneration__hit",
      x: -cueWidth / 2,
      y: -cueHeight / 2,
      width: cueWidth,
      height: cueHeight,
    });
    hit.addEventListener("pointerdown", (pointerEvent) => {
      if (pointerEvent.button === 0) {
        pointerEvent.stopPropagation();
      }
    });
    cue.append(hit);
    if (effectEnabled) {
      const plus = semanticImpactGlyph(ownerDocument, "healing");
      plus.classList.add("combat-regeneration__plus");
      cue.append(
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
      const visibleValue = `+${formatCompactDisplayNumber(event.value)}`;
      value.textContent = visibleValue;
      constrainCompactText(value, visibleValue, cueWidth - 8, 6);
      cue.append(value);
    }
    group.dataset.value = String(event.value);
    if (effectEnabled || battleTextEnabled) {
      connector.append(
        svgElement(ownerDocument, "line", {
          class: "combat-cue__leader",
          "aria-hidden": "true",
        }),
      );
    }
    group.append(cue);
    const position = event.cue ?? event.recipient;
    setAttributes(cue, {
      transform: `translate(${position.x} ${position.y})`,
    });
    this.#updateCueLeader(connector, event);
  }

  /**
   * Paint a filtered status change at its allocated recipient location.
   *
   * ownerDocument creates nodes in group/connector; event supplies recipient, token,
   * lifecycle, paintParts and layout. Missing recipient does nothing. Render the status
   * and change symbol, or the distinct break/death-clear mark, only when enabled.
   * Keep the leader tied to the supplied recipient and stop primary pointerdown on
   * the hit region. Return undefined; lifecycle meaning comes from the plan.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement} connector
   * @param {JsonRecord} event
   */
  #renderLifecycle(ownerDocument, group, connector, event) {
    if (!event.recipient) {
      return;
    }
    const visibleEvent = paintAwareExplanationEvent(event);
    const effectEnabled = paintPartEnabled(event, "effect");
    const breakEnabled = effectEnabled;
    const lifecycle = svgElement(ownerDocument, "g", {
      class: "combat-lifecycle",
    });
    assignLayoutPlacement(
      lifecycle,
      event.cueLayoutKey,
      event.cueBounds,
      event.cueDisposition,
      event.cueCollisionFree,
    );
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
      lifecycle.append(statusIcon);
      if (
        visibleEvent.lifecycle !== "trap_broken" &&
        visibleEvent.lifecycle !== "cleared_by_death"
      ) {
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
        const changeIcon = createSvgIcon(
          ownerDocument,
          visibleEvent.lifecycleToken?.glyphKey ?? "unknown",
          { className: "combat-lifecycle__change-icon" },
        );
        setAttributes(changeIcon, { x: -6, y: -6, width: 12, height: 12 });
        change.append(changeIcon);
        lifecycle.append(change);
      }
    }
    if (breakEnabled && visibleEvent.lifecycle === "cleared_by_death") {
      const sweep = svgElement(ownerDocument, "g", {
        class: "combat-lifecycle__death-sweep",
      });
      sweep.append(
        svgElement(ownerDocument, "path", {
          class: "combat-lifecycle__death-sweep-cut",
          d: "M -22 10 L 22 -10",
        }),
      );
      lifecycle.append(sweep);
    }
    if (breakEnabled && visibleEvent.lifecycle === "trap_broken") {
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
    if (lifecycle.childElementCount > 0) {
      connector.append(
        svgElement(ownerDocument, "line", {
          class: "combat-cue__leader",
          "aria-hidden": "true",
        }),
      );
    }
    group.append(lifecycle);
    const position = event.cue ?? event.recipient;
    setAttributes(lifecycle, {
      transform: `translate(${position.x} ${position.y})`,
    });
    this.#updateCueLeader(connector, event);
  }

  /**
   * Paint one admitted pulse, life-state ring, team wave or death HUD.
   *
   * ownerDocument creates nodes in group and connector; event owns anchor/semantic and
   * layout. plan/options supply timing and motion; animationSpecs receives any ring
   * specs. Cooldown-start or absent-anchor rows do nothing. Dispatch special cases to
   * their shared painters and otherwise draw a compact pulse at cue or anchor. Return
   * undefined. Team corners are display positions, not simulator body locations.
   *
   * @param {Document} ownerDocument
   * @param {SVGElement} group
   * @param {SVGElement} connector
   * @param {JsonRecord} event
   * @param {JsonRecord} plan
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   */
  #renderSemanticPulse(
    ownerDocument,
    group,
    connector,
    event,
    plan,
    options,
    animationSpecs,
  ) {
    if (event.cueSemantic === "cooldown_started") {
      return;
    }
    if (!event.anchor) {
      return;
    }
    if (event.cueSemantic === "death_announcement") {
      this.#renderDeathAnnouncement(ownerDocument, group, event);
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
    appendAllocatedLeader(
      ownerDocument,
      connector,
      "combat-cue__leader combat-cue__leader--semantic",
      event.cueLeader,
    );
    if (event.cueSemantic === "respawn_wave_occurred") {
      this.#renderRespawnWave(ownerDocument, group, event);
      return;
    }
    const pulse = svgElement(ownerDocument, "g", {
      class: `combat-semantic-pulse combat-semantic-pulse--${cssIdentifier(event.cueSemantic)}`,
      "data-semantic": event.cueSemantic,
    });
    assignLayoutPlacement(
      pulse,
      event.cueLayoutKey,
      event.cueBounds,
      event.cueDisposition,
      event.cueCollisionFree,
    );
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
    if (event.cueSemantic === "cooldown_ready") {
      pulse.append(
        svgElement(ownerDocument, "path", {
          class: "combat-semantic-pulse__mark",
          d: "M -7 0 L -2 6 L 8 -7",
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
    const anchor = event.cue ?? event.anchor;
    setAttributes(pulse, {
      transform: `translate(${anchor.x} ${anchor.y})`,
    });
  }

  /**
   * Paint an outward death or respawn ring at the authorized successor anchor.
   *
   * ownerDocument creates nodes under group; event supplies the anchor and phase.
   * plan supplies fallback timing, options selects motion/settlement, and animationSpecs
   * receives a ring-radius spec only for normal unsettled positive-duration motion.
   * The DOM radius is already the final size, so reduced/off/settled views never show
   * an inward intermediate state. Return undefined.
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
   * Append one persistent team-wave banner at its planned screen location.
   *
   * ownerDocument creates nodes under group. event supplies team identity, label,
   * allocation and cue/anchor. Use the shared fixed footprint; the banner is the event
   * hit surface and its text has no separate scientific owner. Return undefined.
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
    assignLayoutPlacement(
      wave,
      event.cueLayoutKey,
      event.cueBounds,
      event.cueDisposition,
      event.cueCollisionFree,
    );
    wave.append(
      svgElement(ownerDocument, "rect", {
        class: "combat-respawn-wave__panel",
        x: -CHOREOGRAPHY_PAINT_FOOTPRINTS.respawnWave.panelWidth / 2,
        y: -CHOREOGRAPHY_PAINT_FOOTPRINTS.respawnWave.panelHeight / 2,
        width: CHOREOGRAPHY_PAINT_FOOTPRINTS.respawnWave.panelWidth,
        height: CHOREOGRAPHY_PAINT_FOOTPRINTS.respawnWave.panelHeight,
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
    const anchor = event.cue ?? event.anchor;
    setAttributes(wave, {
      transform: `translate(${anchor.x} ${anchor.y})`,
    });
  }

  /**
   * Append the bounded team death HUD with full public identities and tooltips.
   *
   * ownerDocument creates nodes under group. event supplies panel dimensions/anchor,
   * label, prewrapped textRows and matching members with contributor evidence. Each
   * victim receives a focused explanation; missing attribution stays neutral as planned.
   * Return undefined. This uses the existing transition clock and no world-space body
   * position or actor-input permission is added.
   *
   * @param {Document} ownerDocument @param {SVGElement} group @param {JsonRecord} event
   */
  #renderDeathAnnouncement(ownerDocument, group, event) {
    const card = svgElement(ownerDocument, "g", {
      class: `combat-death-announcement combat-death-announcement--${event.teamId === null ? "unattributed" : `team-${event.teamId === 1 ? "a" : "b"}`}`,
      "data-team-side": event.teamSide,
      transform: `translate(${event.anchor.x - event.panelWidth / 2} ${event.anchor.y - event.panelHeight / 2})`,
    });
    card.append(
      svgElement(ownerDocument, "rect", {
        class: "combat-death-announcement__panel",
        width: event.panelWidth,
        height: event.panelHeight,
        rx: 7,
      }),
    );
    const heading = svgElement(ownerDocument, "text", {
      class: "combat-death-announcement__heading",
      x: 10,
      y: 18,
    });
    heading.textContent = String(event.label).toUpperCase();
    card.append(heading);
    event.textRows.forEach(
      (
        /** @type {{lines: readonly string[], y: number}} */ row,
        /** @type {number} */ memberIndex,
      ) => {
        const member = event.members[memberIndex];
        const victim = svgElement(ownerDocument, "g", {
          class: "combat-death-announcement__victim",
          role: "img",
          tabindex: "0",
          "aria-label": `${member.title} — Kill Contributors`,
        });
        victim.append(
          svgElement(ownerDocument, "rect", {
            class: "combat-death-announcement__victim-hit",
            x: 6,
            y: row.y - 12,
            width: event.panelWidth - 12,
            height: row.lines.length * 14 + 6,
          }),
        );
        const label = svgElement(ownerDocument, "text", {
          class: "combat-death-announcement__agent",
          x: 10,
          y: row.y,
        });
        row.lines.forEach((line, index) => {
          const span = svgElement(ownerDocument, "tspan", {
            x: 10,
            y: row.y + index * 14,
          });
          span.textContent = line + (index < row.lines.length - 1 ? " " : "");
          label.append(span);
        });
        victim.append(label);
        registerTooltipOwner(
          victim,
          explainDeathAnnouncement({
            ...event,
            eventId: `${event.eventId}:victim:${memberIndex}`,
            label: "Agent Died",
            members: [member],
          }),
        );
        card.append(victim);
      },
    );
    group.append(card);
  }

  /**
   * Replace crossing backplates while preserving the continuous route hit path.
   *
   * underlay may be null and route may omit bridgeGaps. Remove prior backplates, then
   * add circles only for finite positive-gap allocator records before the visible path.
   * Invalid gap rows are skipped. Return undefined; no crossings are discovered here.
   *
   * @param {SVGElement | null} underlay
   * @param {JsonRecord | null | undefined} route
   */
  #syncRouteBridgeGaps(underlay, route) {
    if (!underlay) {
      return;
    }
    for (const bridge of underlay.querySelectorAll(".combat-route__bridge-backplate")) {
      bridge.remove();
    }
    const visibleRoute = underlay.querySelector(".combat-route__path");
    if (!(visibleRoute instanceof SVGElement) || !Array.isArray(route?.bridgeGaps)) {
      return;
    }
    for (const bridgeGap of route.bridgeGaps) {
      if (
        !bridgeGap ||
        !Number.isFinite(bridgeGap.at?.x) ||
        !Number.isFinite(bridgeGap.at?.y) ||
        !Number.isFinite(bridgeGap.gap) ||
        bridgeGap.gap <= 0
      ) {
        continue;
      }
      const bridge = svgElement(underlay.ownerDocument, "circle", {
        class: "combat-route__bridge-backplate",
        "aria-hidden": "true",
        cx: bridgeGap.at.x,
        cy: bridgeGap.at.y,
        r: bridgeGap.gap / 2,
        "data-bridge-with-layout-key": bridgeGap.withLayoutKey,
        "data-bridge-gap": bridgeGap.gap,
      });
      visibleRoute.before(bridge);
    }
  }

  /**
   * Update supported retained event geometry from a compatible new plan row.
   *
   * group, optional connector and optional underlay are existing SVG nodes; event
   * contains new allocated points/route metadata. Update activation, health, status,
   * regeneration and pulse geometry plus crossing backplates. Existing animation
   * objects keep their elements. Return undefined; content and authority changes need
   * a separate installation rather than this geometry update.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} connector
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #updateGeometry(group, connector, underlay, event) {
    this.#applySpatialDisposition(group, event);
    if (underlay) {
      assignLayoutKey(underlay, event.routeLayoutKey ?? event.route?.layoutKey);
      if (Number.isInteger(event.routeLane ?? event.route?.lane)) {
        underlay.dataset.lane = String(event.routeLane ?? event.route.lane);
      }
    }
    if (event.kind === "activation") {
      this.#updateActivationGeometry(group, connector, underlay, event);
    } else if (event.kind === "net_health") {
      this.#updateNetGeometry(group, connector, event);
    } else if (event.kind === "regeneration") {
      const cue = group.querySelector(".combat-regeneration");
      if (cue instanceof SVGElement && event.recipient) {
        assignLayoutPlacement(
          cue,
          event.cueLayoutKey,
          event.cueBounds,
          event.cueDisposition,
          event.cueCollisionFree,
        );
        const position = event.cue ?? event.recipient;
        setAttributes(cue, {
          transform: `translate(${position.x} ${position.y})`,
        });
        this.#updateCueLeader(connector, event);
      }
    } else if (event.kind === "status_lifecycle") {
      const lifecycle = group.querySelector(".combat-lifecycle");
      if (lifecycle instanceof SVGElement && event.recipient) {
        assignLayoutPlacement(
          lifecycle,
          event.cueLayoutKey,
          event.cueBounds,
          event.cueDisposition,
          event.cueCollisionFree,
        );
        const position = event.cue ?? event.recipient;
        setAttributes(lifecycle, {
          transform: `translate(${position.x} ${position.y})`,
        });
        this.#updateCueLeader(connector, event);
      }
    } else if (event.kind === "semantic_pulse") {
      const pulse = group.querySelector(
        ".combat-semantic-pulse, .combat-lifecycle-ring, .combat-respawn-wave",
      );
      if (pulse && event.anchor) {
        const lifecycleRing = pulse.matches(".combat-lifecycle-ring");
        const cue = lifecycleRing ? event.anchor : (event.cue ?? event.anchor);
        pulse.setAttribute("transform", `translate(${cue.x} ${cue.y})`);
        if (!lifecycleRing) {
          assignLayoutPlacement(
            pulse,
            event.cueLayoutKey,
            event.cueBounds,
            event.cueDisposition,
            event.cueCollisionFree,
          );
        }
      }
      if (
        event.cueSemantic !== "agent_died" &&
        event.cueSemantic !== "agent_respawned"
      ) {
        syncAllocatedLeader(
          connector,
          ".combat-cue__leader--semantic",
          event.cueLeader,
        );
      }
    }
    this.#syncRouteBridgeGaps(underlay, event.route);
  }

  /**
   * Reproject route, impact, local glyph and Charge label within existing nodes.
   *
   * group/connector/underlay belong to one event; event supplies the new geometry.
   * Update path/hit data, reconcile direction-arrow count, move allocation-owned labels
   * and leaders, and update an existing particle's offset path. Return undefined.
   * No action endpoint is reconstructed and animation progress is not restarted.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} connector
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   */
  #updateActivationGeometry(group, connector, underlay, event) {
    const path = underlay?.querySelector(".combat-route__path");
    const hitPath = underlay?.querySelector(".combat-route__hit");
    const ownership = underlay?.querySelector(".combat-route__ownership");
    if (path && event.route) {
      path.setAttribute("d", event.route.path);
    }
    if (hitPath && event.route) {
      hitPath.setAttribute("d", event.route.path);
    }
    if (underlay && event.route) {
      const chargeMarkerProgresses =
        event.tokenId === "warrior_charge" &&
        Array.isArray(event.route.markerProgresses)
          ? event.route.markerProgresses.filter(
              (/** @type {unknown} */ progress) =>
                typeof progress === "number" &&
                Number.isFinite(progress) &&
                progress >= 0 &&
                progress <= 1,
            )
          : null;
      const markerProgresses = chargeMarkerProgresses ?? [undefined];
      const arrows = [...underlay.querySelectorAll(".combat-route__arrow")];
      while (arrows.length > markerProgresses.length) {
        arrows.pop()?.remove();
      }
      const insertionPoint = underlay.querySelector(
        ".combat-route__ownership, .combat-route__particle",
      );
      while (arrows.length < markerProgresses.length) {
        const arrow = svgElement(underlay.ownerDocument, "path", {
          class: "combat-route__arrow",
        });
        underlay.insertBefore(arrow, insertionPoint);
        arrows.push(arrow);
      }
      const compact = event.route.markerVariant === "compact";
      for (const [markerIndex, arrow] of arrows.entries()) {
        const marker = routeMarkerPose(event.route, markerProgresses[markerIndex]);
        setAttributes(arrow, {
          "data-marker-index": markerIndex,
          "data-marker-variant": compact ? "compact" : "full",
          d: compact
            ? "M -6 -3 L 2 0 L -6 3 L -4 0 Z"
            : "M -11 -6 L 2 0 L -11 6 L -7 0 Z",
          transform: `translate(${marker.x} ${marker.y}) rotate(${marker.degrees})`,
        });
      }
    }
    if (ownership && event.route) {
      this.#updateChargeOwnershipGeometry(ownership, event);
    }
    const impact = group.querySelector(".combat-impact");
    if (impact && (event.impactCue || event.route?.end || event.target)) {
      const anchor = event.impactCue ?? event.route?.end ?? event.target;
      setAttributes(impact, {
        transform: impactTransform(event, anchor),
      });
      assignLayoutPlacement(
        impact,
        event.impactLayoutKey,
        event.impactBounds,
        event.impactDisposition,
        event.impactCueCollisionFree,
      );
    }
    const local = group.querySelector(".combat-local");
    const anchor = event.sourceCue ?? event.source;
    if (local && anchor) {
      setAttributes(local, {
        transform: `translate(${anchor.x} ${anchor.y})`,
      });
      assignLayoutPlacement(
        local,
        event.sourceLayoutKey,
        event.sourceBounds,
        event.sourceDisposition,
        event.sourceCueCollisionFree,
      );
    }
    syncAllocatedLeader(connector, ".combat-cue__leader--impact", event.impactLeader);
    syncAllocatedLeader(connector, ".combat-cue__leader--source", event.sourceLeader);
    const particle = underlay?.querySelector(".combat-route__particle");
    if (particle instanceof SVGElement && event.route) {
      particle.style.offsetPath = `path("${event.route.path}")`;
    }
  }

  /**
   * Keep the Charge public-ID label attached to its planned route anchor.
   *
   * ownership is the existing label element and event supplies ownership cue/anchor,
   * optional allocated leader and layout metadata. Hide it when either endpoint is
   * absent. Prefer the allocator's leader; otherwise draw a local connector only when
   * the label is displaced by more than four pixels. Return undefined. Labels never
   * use this geometry to infer source or target identity.
   *
   * @param {Element} ownership
   * @param {JsonRecord} event
   */
  #updateChargeOwnershipGeometry(ownership, event) {
    const cue = event.ownershipCue;
    const anchor = event.ownershipAnchor;
    const rendered = cue && anchor;
    ownership.setAttribute(
      "data-spatial-disposition",
      rendered
        ? String(
            event.ownershipSpatialDisposition ??
              event.ownershipDisposition ??
              "recipient_stack",
          )
        : "absent",
    );
    assignLayoutPlacement(
      ownership,
      event.ownershipLayoutKey,
      event.ownershipBounds,
      event.ownershipDisposition,
      event.ownershipCueCollisionFree,
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
    if (isAllocatedLeader(event.ownershipLeader)) {
      syncAllocatedLeader(
        ownership,
        ".combat-route__ownership-leader",
        event.ownershipLeader,
        cue,
      );
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
   * Place existing health-change hit/text nodes inside their allocated cue.
   *
   * group/connector belong to the event. event supplies recipient, cue bounds/position,
   * lane and optional leader. Without recipient do nothing. Invalid bounds hide the
   * hit box; missing cue position uses the existing recipient/lane fallback. Compress
   * long labels without changing text, then update the leader. Return undefined.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} connector
   * @param {JsonRecord} event
   */
  #updateNetGeometry(group, connector, event) {
    const hit = group.querySelector(".combat-net__hit");
    const label = group.querySelector(".combat-net__label");
    const recipientLabel = group.querySelector(".combat-net__recipient");
    if (!event.recipient) {
      return;
    }
    assignLayoutPlacement(
      group,
      event.cueLayoutKey,
      event.cueBounds,
      event.cueDisposition,
      event.cueCollisionFree,
    );
    if (hit) {
      const bounds = event.cueBounds;
      if (
        Number.isFinite(bounds?.left) &&
        Number.isFinite(bounds?.top) &&
        Number.isFinite(bounds?.right) &&
        Number.isFinite(bounds?.bottom) &&
        bounds.right >= bounds.left &&
        bounds.bottom >= bounds.top
      ) {
        setAttributes(hit, {
          visibility: "visible",
          x: bounds.left,
          y: bounds.top,
          width: bounds.right - bounds.left,
          height: bounds.bottom - bounds.top,
        });
      } else {
        hit.setAttribute("visibility", "hidden");
      }
    }
    const x = event.cue?.x ?? event.recipient.x;
    const y = event.cue?.y ?? event.recipient.y - 32 - event.lane * 18;
    if (recipientLabel) {
      setAttributes(recipientLabel, {
        x,
        y: y - 10,
      });
      constrainCompactText(
        recipientLabel,
        recipientLabel.textContent ?? "",
        Number(event.cueBounds?.width ?? 0) - 8,
        8,
      );
    }
    if (label) {
      setAttributes(label, {
        x,
        y: y + 6,
      });
      constrainCompactText(
        label,
        label.textContent ?? "",
        Number(event.cueBounds?.width ?? 0) - 8,
        7,
      );
    }
    this.#updateCueLeader(connector, event);
  }

  /**
   * Publish the plan's display disposition and clear stale hidden attributes.
   *
   * group is retained SVG; event supplies cue, impact or source disposition in that
   * priority order, defaulting to rendered. Set data-spatial-disposition and remove
   * visibility/aria-hidden. Return undefined. This helper records layout state; it
   * does not suppress a row or grant new spatial authority.
   *
   * @param {SVGElement} group
   * @param {Record<string, any>} event
   */
  #applySpatialDisposition(group, event) {
    const disposition = String(
      event.cueDisposition ??
        event.impactDisposition ??
        event.sourceDisposition ??
        "rendered",
    );
    group.dataset.spatialDisposition = disposition;
    group.removeAttribute("visibility");
    group.removeAttribute("aria-hidden");
  }

  /**
   * Update the connector linking a displaced cue to its disclosed recipient.
   *
   * connector may be null; event supplies cueLeader or recipient/cue points. Prefer a
   * valid allocated leader; otherwise update the existing line from recipient to cue.
   * When points are absent, remove its x endpoints. Return undefined without creating
   * new scientific anchors or moving the recipient.
   *
   * @param {SVGElement | null} connector
   * @param {JsonRecord} event
   */
  #updateCueLeader(connector, event) {
    const leader = connector?.querySelector(".combat-cue__leader");
    const recipient = event.recipient;
    const cue = event.cue;
    if (leader && isAllocatedLeader(event.cueLeader)) {
      syncAllocatedLeader(connector, ".combat-cue__leader", event.cueLeader);
      return;
    }
    if (!leader || !recipient || !cue) {
      leader?.removeAttribute("x1");
      leader?.removeAttribute("x2");
      return;
    }
    setAttributes(leader, {
      visibility: "visible",
      x1: recipient.x,
      y1: recipient.y,
      x2: cue.x,
      y2: cue.y,
    });
  }
}

/**
 * Create a detached SVG element with supplied presentation attributes.
 *
 * ownerDocument owns the node, tagName selects its SVG tag, and attributes defaults
 * to an empty object. Return the element after setAttributes; browser DOM errors
 * propagate. The caller decides where to append it.
 *
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
 * Apply a presentation attribute map to an existing element.
 *
 * Remove names whose value is null, undefined or false; stringify all other values.
 * Return undefined. This mutates DOM and performs no value/authority sanitization.
 *
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
 * Attach an integer value as the role-specific internal slot dataset field.
 *
 * group is an SVG node and role names source/target/etc. Nonintegers do nothing;
 * no range/roster validation is performed. This metadata never supplies display IDs.
 *
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
 * Attach a nonempty string value to the role-specific presentation-key field.
 *
 * group is SVG and role names the relationship. Other values do nothing; old values
 * are not removed. The caller already owns disclosure and key validity.
 *
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
 * Attach nonempty string value as element's data-layout-key.
 *
 * Other values do nothing and do not clear an older key. This is display metadata.
 *
 * @param {Element} element
 * @param {unknown} value
 */
function assignLayoutKey(element, value) {
  if (typeof value === "string" && value.length > 0) {
    element.setAttribute("data-layout-key", value);
  }
}

/**
 * Publish allocator metadata on its visible semantic owner.
 *
 * element receives a nonempty layoutKey, finite bounds edges, optional nonempty
 * disposition and Boolean collisionFree. Missing/invalid pieces are skipped rather
 * than removing old attributes. Return undefined. Bounds describe the reserved cue
 * rectangle; decorative leaders may extend outside it. No layout is solved here.
 *
 * @param {Element} element
 * @param {unknown} layoutKey
 * @param {unknown} bounds
 * @param {unknown} disposition
 * @param {unknown} collisionFree
 */
function assignLayoutPlacement(element, layoutKey, bounds, disposition, collisionFree) {
  assignLayoutKey(element, layoutKey);
  if (bounds && typeof bounds === "object") {
    const rectangle = /** @type {Record<string, unknown>} */ (bounds);
    for (const edge of ["left", "top", "right", "bottom"]) {
      const value = rectangle[edge];
      if (Number.isFinite(value)) {
        element.setAttribute(`data-layout-${edge}`, String(value));
      }
    }
  }
  if (typeof disposition === "string" && disposition.length > 0) {
    element.setAttribute("data-layout-disposition", disposition);
  }
  if (typeof collisionFree === "boolean") {
    element.setAttribute("data-layout-collision-free", String(collisionFree));
  }
}

/**
 * Create and append a line for valid allocator-supplied endpoints.
 *
 * ownerDocument creates the node under parent with className. leader must contain
 * finite start/end; otherwise return null. origin defaults to {x:0,y:0} and converts
 * absolute screen coordinates into the parent's local coordinates. Return the line.
 *
 * @param {Document} ownerDocument
 * @param {Element} parent
 * @param {string} className
 * @param {unknown} leader
 * @param {{x: number, y: number}} [origin]
 */
function appendAllocatedLeader(
  ownerDocument,
  parent,
  className,
  leader,
  origin = { x: 0, y: 0 },
) {
  if (!isAllocatedLeader(leader)) {
    return null;
  }
  const element = svgElement(ownerDocument, "line", {
    class: className,
    "aria-hidden": "true",
  });
  setAllocatedLeaderGeometry(element, leader, origin);
  parent.append(element);
  return element;
}

/**
 * Update or hide an existing selected leader under parent.
 *
 * parent may be null; selector identifies an existing SVG element. Invalid leader
 * hides it; absent elements are not created. A non-line is replaced with an SVG line
 * retaining its class. origin defaults to zero for coordinate conversion. Return
 * undefined; a failed SVG replacement type check throws TypeError.
 *
 * @param {Element | null} parent
 * @param {string} selector
 * @param {unknown} leader
 * @param {{x: number, y: number}} [origin]
 */
function syncAllocatedLeader(parent, selector, leader, origin = { x: 0, y: 0 }) {
  if (parent === null) {
    return;
  }
  let element = parent.querySelector(selector);
  if (!(element instanceof SVGElement)) {
    return;
  }
  if (!isAllocatedLeader(leader)) {
    element.setAttribute("visibility", "hidden");
    return;
  }
  if (element.localName !== "line") {
    const replacement = svgElement(element.ownerDocument, "line", {
      class: element.getAttribute("class"),
      "aria-hidden": "true",
    });
    if (!(replacement instanceof SVGElement)) {
      throw new TypeError("Allocated leader replacement is not SVG geometry.");
    }
    element.replaceWith(replacement);
    element = replacement;
  }
  element.setAttribute("visibility", "visible");
  setAllocatedLeaderGeometry(/** @type {SVGElement} */ (element), leader, origin);
}

/**
 * Return the supplied start/end references for a valid leader, or an empty array.
 *
 * leader needs four finite numeric coordinates. The array is new, but its point
 * objects are not copied or frozen by this helper.
 *
 * @param {unknown} leader
 * @returns {Array<{x: number, y: number}>}
 */
function allocatedLeaderPoints(leader) {
  if (!isAllocatedLeader(leader)) {
    return [];
  }
  const candidate = /** @type {Record<string, any>} */ (leader);
  return [candidate.start, candidate.end];
}

/**
 * Write a valid leader as local line endpoints and inspection metadata.
 *
 * element is SVG; leader contains finite start/end; origin is a caller-validated
 * finite point subtracted from both. Set x/y endpoints, record JSON points and remove
 * path d. Invalid leader endpoints throw TypeError. Return undefined; inputs are unchanged.
 *
 * @param {SVGElement} element
 * @param {Record<string, any>} leader
 * @param {{x: number, y: number}} origin
 */
function setAllocatedLeaderGeometry(element, leader, origin) {
  const points = allocatedLeaderPoints(leader).map(({ x, y }) => ({
    x: x - origin.x,
    y: y - origin.y,
  }));
  const first = points[0];
  const last = points.at(-1);
  if (!first || !last) {
    throw new TypeError("Allocated leader geometry requires two finite points.");
  }
  element.setAttribute("data-leader-points", JSON.stringify(points));
  element.removeAttribute("d");
  setAttributes(element, {
    x1: first.x,
    y1: first.y,
    x2: last.x,
    y2: last.y,
  });
}

/**
 * Return whether value provides finite start/end x/y numbers.
 *
 * Extra fields are ignored. This is a shape check, not layout or authority validation.
 *
 * @param {unknown} value
 * @returns {value is {start: {x: number, y: number}, end: {x: number, y: number}}}
 */
function isAllocatedLeader(value) {
  if (value === null || typeof value !== "object") {
    return false;
  }
  const leader = /** @type {Record<string, any>} */ (value);
  return (
    Number.isFinite(leader.start?.x) &&
    Number.isFinite(leader.start?.y) &&
    Number.isFinite(leader.end?.x) &&
    Number.isFinite(leader.end?.y)
  );
}

/**
 * Copy every data-* attribute from source to target.
 *
 * Both are owned SVG nodes. Other attributes and absent source fields are not copied
 * or removed. Return undefined; the source metadata must already be safe to disclose.
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
 * Return a CSS suffix for a lowercase letter/digit/underscore string.
 *
 * Replace underscores with hyphens. Any other value returns unknown; no HTML or
 * CSS selector text is evaluated.
 *
 * @param {unknown} value
 */
function cssIdentifier(value) {
  return typeof value === "string" && /^[a-z0-9_]+$/.test(value)
    ? value.replaceAll("_", "-")
    : "unknown";
}

/**
 * Create a detached nonnumeric damage, healing or neutral recipient mark.
 *
 * ownerDocument creates SVG. value damage gives a minus, healing gives a plus, and
 * all other values give a neutral diamond. Return the group; no magnitude or outcome
 * is inferred from this display token.
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
 * Return a screen translation and display-only scale for an impact glyph.
 *
 * event token selects bounded Trap/Charge ornaments; four or more coincident Basic
 * routes use a smaller Basic mark. anchor supplies the already authorized screen
 * point. The transform does not alter source/recipient identity or route endpoints.
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
 * Return the event's semantic DOM phase label.
 *
 * event kind/semantic selects regeneration, pulse, outcome, target-only impact or
 * activation. This label is for presentation inspection, not simulator phase logic.
 *
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
  if (event.kind === "activation" && event.presentationKind === "target_only_impact") {
    return "impact";
  }
  return "activation";
}

/**
 * Format a compact signed health-change label without hiding a tiny nonzero value.
 *
 * delta is the recorded finite net change; outcome unchanged or exact zero returns
 * HP unchanged. Other values use NET plus/minus; a magnitude rounded to zero is
 * shown as <0.01. This formats the input and does not recompute combat resolution.
 *
 * @param {number} delta
 * @param {string} outcome
 */
function netLabel(delta, outcome) {
  if (outcome === "unchanged" || delta === 0) {
    return "HP unchanged";
  }
  const magnitude = formatCompactDisplayNumber(Math.abs(delta));
  const visibleMagnitude = magnitude === "0" ? "<0.01" : magnitude;
  return delta < 0 ? `NET −${visibleMagnitude}` : `NET +${visibleMagnitude}`;
}

/**
 * Compress long visible SVG text without changing its full text content.
 *
 * element is a text owner, text supplies its Unicode character count, maximumLength
 * is a pixel width and naturalCharacterLimit is the uncompressed length threshold.
 * Set textLength/lengthAdjust only when both limits call for compression; otherwise
 * remove them. Return undefined. Semantic tooltip text remains complete.
 *
 * @param {Element} element
 * @param {string} text
 * @param {number} maximumLength
 * @param {number} naturalCharacterLimit
 */
function constrainCompactText(element, text, maximumLength, naturalCharacterLimit) {
  if ([...text].length > naturalCharacterLimit && maximumLength > 0) {
    setAttributes(element, {
      textLength: maximumLength,
      lengthAdjust: "spacingAndGlyphs",
    });
    return;
  }
  element.removeAttribute("textLength");
  element.removeAttribute("lengthAdjust");
}

/**
 * Return whether the event's named paint part is enabled.
 *
 * event without an object paintParts uses legacy all-on behavior. Otherwise only
 * an exact true at part enables it; missing keys fail closed. No filter is mutated.
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
 * Return a shallow explanatory view matching the visible paint grammar.
 *
 * event is unchanged. When activation ability paint is hidden, omit its token and
 * use the impact semantic as eventType. When health recipient text is hidden, clear
 * recipientPublicAgentId in the view. Refresh/reapply lifecycle variants use the
 * ordinary applied explanation. Otherwise return event itself. This is display-copy
 * selection, not authorization or deletion of the original scientific identity.
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
    ["refreshed", "reapplied", "trap_broken_and_reapplied"].includes(event.lifecycle)
  ) {
    return {
      ...event,
      eventType: "status_applied",
      lifecycle: "applied",
      lifecycleToken: resolveVisualToken("lifecycle", "applied", event),
    };
  }
  return event;
}

/**
 * Return opacity keyframes for the planned event and motion mode.
 *
 * event.persistent decides whether the final opacity remains one, except normal
 * net-health fades always end at zero. motionMode is normal/reduced/off; the caller
 * decides whether to schedule these frames. No browser animation starts here.
 *
 * @param {JsonRecord} event
 * @param {"normal" | "reduced" | "off"} motionMode
 * @returns {Keyframe[]}
 */
function eventKeyframes(event, motionMode) {
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
 * Package a deterministic animation description without starting it.
 *
 * element is the target, keyframes supplies browser frames, options gives browser
 * timing, and plan/event/part form its stable ID. Return a frozen spec; options is
 * frozen in place while element/keyframes remain shared references. The controller
 * creates and owns the eventual browser animation.
 *
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
