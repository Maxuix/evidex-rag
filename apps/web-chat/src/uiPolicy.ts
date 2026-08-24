/** Frontend interaction policy; backend pagination limits live in Python. */
export const UI_POLICY = Object.freeze({
  knowledgeBasePageSize: 100,
  documentPageSize: 100,
  chunkPageSize: 100,
  indexingJobPageSize: 100,
  chatPageSize: 50,
  runPollInitialMs: 1200,
  runPollJitterMs: 350,
  runPollRetryMs: 2200,
  graphPollMs: 2000,
  indexingPollMs: 1800,
});
