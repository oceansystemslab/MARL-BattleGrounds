import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  joinReplayTransportAndTimelineV1,
  joinTransportAndAuthorizedPresentationV1,
} from "../src/authorized-presentation-normalizer.js";
import {
  isReplayAgentRecipientIdentityRotation,
  isReplayAgentRecipientRotation,
  replayAgentRecipientRotationIdentity,
} from "../src/replay-recipient-rotation.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @type {readonly ("replay_no_shared_obs_agent_pov" | "replay_shared_obs_agent_pov")[]} */
const REPLAY_AGENT_KINDS = Object.freeze([
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
]);

/** @param {"replay_no_shared_obs_agent_pov" | "replay_shared_obs_agent_pov"} kind */
async function joinedReplayAgent(kind) {
  const source = fixture.pairs[kind];
  const joined = await joinTransportAndAuthorizedPresentationV1(
    source.transport,
    source.presentation,
  );
  return joinReplayTransportAndTimelineV1(joined, source.timeline);
}

/**
 * @param {NonNullable<ReturnType<typeof replayAgentRecipientRotationIdentity>>} source
 * @param {Readonly<Record<string, unknown>>} changes
 */
function changedIdentity(source, changes) {
  return Object.freeze({ ...source, ...changes });
}

/** @param {unknown} joined */
function requiredRotationIdentity(joined) {
  const identity = replayAgentRecipientRotationIdentity(joined);
  if (identity === null) {
    throw new TypeError("Expected a Replay Agent rotation identity.");
  }
  return identity;
}

test("Replay Agent recipient identity is extracted only from a branded pair", async () => {
  for (const kind of REPLAY_AGENT_KINDS) {
    const joined = await joinedReplayAgent(kind);
    const identity = requiredRotationIdentity(joined);
    assert.equal(Object.isFrozen(identity), true);
    assert.equal(
      identity.recipientPublicAgentId,
      joined.presentation.authority.recipient_public_agent_id,
    );
    assert.equal(
      identity.recipientPresentationKey,
      joined.presentation.authority.recipient_presentation_key,
    );
    assert.equal(JSON.parse(identity.scope).length, 23);
    assert.equal(replayAgentRecipientRotationIdentity({ ...joined }), null);
  }
});

test("Replay Agent recipient rotation requires one unchanged replay scope", async () => {
  for (const kind of REPLAY_AGENT_KINDS) {
    const identity = requiredRotationIdentity(await joinedReplayAgent(kind));
    const next = changedIdentity(identity, {
      recipientPublicAgentId: "agent-slot-1",
      recipientPresentationKey:
        "pov_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    });
    assert.equal(isReplayAgentRecipientIdentityRotation(identity, next), true);

    const scope = JSON.parse(identity.scope);
    for (let index = 0; index < scope.length; index += 1) {
      const changedScope = [...scope];
      changedScope[index] =
        typeof changedScope[index] === "boolean"
          ? !changedScope[index]
          : typeof changedScope[index] === "number"
            ? changedScope[index] + 1
            : `${changedScope[index]}-changed`;
      assert.equal(
        isReplayAgentRecipientIdentityRotation(
          identity,
          changedIdentity(next, { scope: JSON.stringify(changedScope) }),
        ),
        false,
        `scope slot ${index} must be a reset boundary`,
      );
    }
    assert.equal(
      isReplayAgentRecipientIdentityRotation(
        identity,
        changedIdentity(next, {
          recipientPublicAgentId: identity.recipientPublicAgentId,
        }),
      ),
      false,
    );
    assert.equal(
      isReplayAgentRecipientIdentityRotation(
        identity,
        changedIdentity(next, {
          recipientPresentationKey: identity.recipientPresentationKey,
        }),
      ),
      false,
    );
  }
});

test("Replay Agent preference rotation rejects same-recipient and leaf crossings", async () => {
  const noShared = await joinedReplayAgent("replay_no_shared_obs_agent_pov");
  const shared = await joinedReplayAgent("replay_shared_obs_agent_pov");
  assert.equal(isReplayAgentRecipientRotation(noShared, noShared), false);
  assert.equal(isReplayAgentRecipientRotation(shared, shared), false);
  assert.equal(isReplayAgentRecipientRotation(noShared, shared), false);
  assert.equal(isReplayAgentRecipientRotation(shared, noShared), false);
  assert.equal(isReplayAgentRecipientRotation(null, shared), false);
});
