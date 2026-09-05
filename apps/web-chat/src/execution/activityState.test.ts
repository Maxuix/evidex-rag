import { describe, expect, it } from "vitest";
import { applyActivity, disconnectActivity, emptyActivity, parseActivityEvent, parseActivitySnapshot } from "./activityState";
import { eventFixture, RUN_ID, snapshotFixture, sourceFixture, stepFixture } from "./activityFixtures";

describe("execution activity protocol and ordering", () => {
  it("accepts late updates to separate parallel calls without rolling a newer step back", () => {
    let state = applyActivity(emptyActivity(RUN_ID), RUN_ID, 1, eventFixture(stepFixture({ seq: 3 })), 1000);
    state = applyActivity(state, RUN_ID, 1, eventFixture(stepFixture({ step_id: "step_2", ordinal: 2, seq: 2, name: "keyword_search" })), 2000);
    expect(Object.keys(state.attempts[1].steps)).toHaveLength(2);
    expect(state.attempts[1].elapsedMs).toBe(3200);
    expect(state.attempts[1].receivedAt).toBe(1000);
    expect(applyActivity(state, RUN_ID, 1, eventFixture(stepFixture({ seq: 1 })))).toBe(state);
    expect(state.attempts[1].incomplete).toBe(true);
  });
  it("rejects other runs and older attempts, retains disconnected records", () => {
    const initial = applyActivity(emptyActivity(RUN_ID), RUN_ID, 1, eventFixture());
    const retried = applyActivity(initial, RUN_ID, 1, eventFixture(stepFixture(), 2));
    expect(applyActivity(retried, RUN_ID, 1, eventFixture())).toBe(retried);
    expect(applyActivity(retried, RUN_ID, 1, { ...eventFixture(), run_id: "another" })).toBe(retried);
    expect(disconnectActivity(retried, RUN_ID).attempts[1].steps).toEqual(initial.attempts[1].steps);
    expect(disconnectActivity(retried, RUN_ID).attempts[2].incomplete).toBe(true);
  });
  it("rejects unsupported versions, unknown payload fields, unsafe IDs and impossible ordering", () => {
    const event = eventFixture(stepFixture({ sources: [sourceFixture] }));
    expect(parseActivityEvent(event)).toEqual(event);
    expect(parseActivityEvent({ ...event, version: "chat_activity_v2" })).toBeNull();
    expect(parseActivityEvent({ ...event, step: { ...event.step, reasoning: "private" } })).toBeNull();
    expect(parseActivityEvent({ ...event, step: { ...event.step, sources: [{ ...sourceFixture, document_id: null }] } })).toBeNull();
    expect(parseActivityEvent({ ...event, seq: 9 })).toBeNull();
    expect(parseActivitySnapshot(snapshotFixture([stepFixture(), stepFixture()]))).toBeNull();
    expect(parseActivitySnapshot(snapshotFixture([stepFixture({ ended_offset_ms: 20 })]))).toBeNull();
  });
  it("bounds retained live rows and reports dropped records", () => {
    let state = emptyActivity(RUN_ID);
    for (let n = 1; n <= 1030; n++) state = applyActivity(state, RUN_ID, 1, eventFixture(stepFixture({ step_id: `step_${n}`, ordinal: n, seq: n })));
    expect(Object.keys(state.attempts[1].steps)).toHaveLength(1024);
    expect(state.attempts[1].omitted).toBe(6);
  });
});
