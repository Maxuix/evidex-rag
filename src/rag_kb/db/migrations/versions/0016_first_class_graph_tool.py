"""Upgrade ChatRun snapshots to the first-class Graph Relations tool.

Revision ID: 0016_first_class_graph_tool
Revises: 0015_remove_retired_chat_state
Create Date: 2026-08-24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0016_first_class_graph_tool"
down_revision: str | None = "0015_remove_retired_chat_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_V1_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v3', "
    "'budget', jsonb_build_object("
    "'max_model_rounds', 8, 'max_graph_calls', 2))"
)
_V2_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v2', "
    "'budget', jsonb_build_object('max_model_rounds', 8))"
)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
          status_row RECORD;
          run_row RECORD;
          event_row RECORD;
          known_profiles TEXT[];
          total INTEGER;
        BEGIN
          SELECT count(*) INTO total FROM chat_run;
          RAISE NOTICE 'chat_run rows before v3 migration: %', total;
          FOR status_row IN
            SELECT status, count(*) AS n
              FROM chat_run
             GROUP BY status
             ORDER BY status
          LOOP
            RAISE NOTICE 'chat_run status % : %', status_row.status, status_row.n;
          END LOOP;

          known_profiles := ARRAY[
            'exact_vector_v1', 'exact_vector_v2',
            'hybrid_fts_rrf_v1', 'hybrid_fts_rrf_v2',
            'graph_augmented_v1',
            'graphiti_edge_augmented_v1',
            'graphiti_path_augmented_v2', 'graphiti_path_augmented_v3',
            'adaptive_graphiti_v1', 'adaptive_graphiti_v2'
          ];

          FOR run_row IN
            SELECT id, agent_configuration, agent_trace, retrieval_strategy
              FROM chat_run
          LOOP
            IF (
              jsonb_typeof(run_row.agent_configuration) <> 'object'
              OR run_row.agent_configuration->>'version' <> 'native_tool_calling_agent_v2'
              OR jsonb_typeof(run_row.agent_configuration->'budget') <> 'object'
              OR (run_row.agent_configuration - ARRAY['version', 'budget']::text[])
                 <> '{}'::jsonb
              OR jsonb_typeof(run_row.agent_configuration->'budget'->'max_model_rounds')
                 <> 'number'
              OR run_row.agent_configuration->'budget' ?| ARRAY[
                   'model_rounds', 'retrieval_calls',
                   'calculation_calls', 'evidence_refs']
            ) THEN
              RAISE EXCEPTION 'unknown agent configuration shape in chat_run %',
                run_row.id;
            END IF;
            IF run_row.agent_trace IS NOT NULL AND (
              jsonb_typeof(run_row.agent_trace) <> 'object'
              OR run_row.agent_trace->>'version' <> 'native_tool_calling_agent_v2'
              OR (run_row.agent_trace - ARRAY[
                   'version', 'events', 'budget', 'usage', 'outcome']::text[])
                 <> '{}'::jsonb
              OR jsonb_typeof(run_row.agent_trace->'events') <> 'array'
              OR jsonb_typeof(run_row.agent_trace->'budget') <> 'object'
              OR jsonb_typeof(run_row.agent_trace->'budget'->'max_model_rounds')
                 <> 'number'
              OR run_row.agent_trace->'budget' ?| ARRAY[
                   'model_rounds', 'retrieval_calls',
                   'calculation_calls', 'evidence_refs']
              OR run_row.agent_trace->'budget' <> run_row.agent_configuration->'budget'
              OR jsonb_typeof(run_row.agent_trace->'usage') <> 'object'
              OR run_row.agent_trace->>'outcome'
                 NOT IN ('answered', 'partial', 'refused')
            ) THEN
              RAISE EXCEPTION 'unknown agent trace shape in chat_run %', run_row.id;
            END IF;
            IF (
              jsonb_typeof(run_row.retrieval_strategy) <> 'object'
              OR (
                run_row.retrieval_strategy->>'profile_version' IS NOT NULL
                AND NOT (
                  run_row.retrieval_strategy->>'profile_version' = ANY(known_profiles)
                )
              )
            ) THEN
              RAISE EXCEPTION 'unknown retrieval snapshot shape in chat_run %',
                run_row.id;
            END IF;
            IF run_row.agent_trace IS NOT NULL THEN
              FOR event_row IN
                SELECT value
                  FROM jsonb_array_elements(run_row.agent_trace->'events')
              LOOP
                IF event_row.value->>'tool' = 'graphiti_supplement' THEN
                  IF event_row.value->>'retrieval_lane' = 'graphiti_supplement' THEN
                    IF (
                      event_row.value->>'route_reason_code'
                         NOT IN ('cross_document_relation_gap', 'entity_alias_gap',
                                 'relation_chain_gap')
                      OR event_row.value->>'route_result_code'
                         NOT IN ('admitted', 'no_new_evidence', 'not_configured',
                                 'not_ready', 'runtime_unavailable', 'rejected')
                    ) THEN
                      RAISE EXCEPTION 'unknown Graphiti event shape in chat_run %',
                        run_row.id;
                    END IF;
                  ELSIF event_row.value->>'retrieval_lane' IS NOT NULL THEN
                    RAISE EXCEPTION
                      'unknown Graphiti lane in chat_run %', run_row.id;
                  END IF;
                END IF;
              END LOOP;
            END IF;
          END LOOP;
        END $$
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM pg_constraint
             WHERE conrelid = 'public.chat_run'::regclass
               AND conname = 'ck_chat_run_ck_chat_run_agent_trace_v2'
          ) THEN
            ALTER TABLE chat_run
              DROP CONSTRAINT ck_chat_run_ck_chat_run_agent_trace_v2;
          ELSE
            ALTER TABLE chat_run DROP CONSTRAINT ck_chat_run_agent_trace_v2;
          END IF;
        END $$
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM pg_constraint
             WHERE conrelid = 'public.chat_run'::regclass
               AND conname = 'ck_chat_run_ck_chat_run_agent_configuration_v2'
          ) THEN
            ALTER TABLE chat_run
              DROP CONSTRAINT ck_chat_run_ck_chat_run_agent_configuration_v2;
          ELSE
            ALTER TABLE chat_run
              DROP CONSTRAINT ck_chat_run_agent_configuration_v2;
          END IF;
        END $$
        """
    )
    op.execute(
        """
        UPDATE chat_run
           SET agent_configuration = jsonb_build_object(
                 'version', 'native_tool_calling_agent_v3',
                 'budget', jsonb_build_object(
                   'max_model_rounds', COALESCE(
                     (agent_configuration->'budget'->>'max_model_rounds')::integer,
                     8
                   ),
                   'max_graph_calls', 2
                 )
               )
        """
    )
    op.execute(
        """
        UPDATE chat_run
           SET retrieval_strategy = CASE
                 WHEN retrieval_strategy->>'profile_version'
                      IN ('adaptive_graphiti_v1', 'adaptive_graphiti_v2')
                 THEN jsonb_build_object(
                   'profile_version', 'adaptive_graphiti_v3',
                   'strategy', 'exact_vector',
                   'top_k', retrieval_strategy->'top_k',
                   'rerank_mode', retrieval_strategy->'rerank_mode',
                   'router', 'native_agent_graph_tool_v1',
                   'augmentation', 'graphiti_path_v3',
                   'graph_edge_limit', 16,
                   'graph_source_chunk_target', 12,
                   'graph_source_chunk_limit', 16,
                   'graph_call_timeout_seconds', 90
                 )
                 ELSE retrieval_strategy
               END
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
          run_row RECORD;
          event_row RECORD;
          graph_call_index INTEGER;
          new_events JSONB;
          new_event JSONB;
          new_reason TEXT;
          new_result TEXT;
          invocation_source TEXT;
        BEGIN
          FOR run_row IN
            SELECT id, agent_trace
              FROM chat_run
             WHERE agent_trace IS NOT NULL
          LOOP
            graph_call_index := 0;
            new_events := '[]'::jsonb;
            FOR event_row IN
              SELECT value
                FROM jsonb_array_elements(run_row.agent_trace->'events')
            LOOP
              new_event := event_row.value;
              IF new_event->>'tool' = 'graphiti_supplement' THEN
                graph_call_index := graph_call_index + 1;
                invocation_source := CASE
                  WHEN new_event->>'tool_call_id' LIKE 'guard\\_%' THEN 'legacy_guard'
                  ELSE 'agent'
                END;
                IF new_event->>'retrieval_lane' = 'graphiti_supplement' THEN
                  new_reason := CASE new_event->>'route_reason_code'
                    WHEN 'cross_document_relation_gap' THEN 'cross_document_relation'
                    WHEN 'entity_alias_gap' THEN 'entity_alias'
                    WHEN 'relation_chain_gap' THEN 'relation_chain'
                    ELSE NULL
                  END;
                  new_result := CASE new_event->>'route_result_code'
                    WHEN 'admitted' THEN 'admitted'
                    WHEN 'no_new_evidence' THEN 'no_evidence'
                    WHEN 'not_configured' THEN 'not_ready'
                    WHEN 'not_ready' THEN 'not_ready'
                    WHEN 'runtime_unavailable' THEN 'unavailable'
                    WHEN 'rejected' THEN 'rejected'
                    ELSE NULL
                  END;
                  IF new_reason IS NULL OR new_result IS NULL THEN
                    RAISE EXCEPTION 'unmappable Graphiti event in chat_run %',
                      run_row.id;
                  END IF;
                  new_event := new_event || jsonb_build_object(
                    'tool', 'search_graph_relations',
                    'retrieval_lane', 'graph_relations',
                    'route_reason_code', new_reason,
                    'route_result_code', new_result,
                    'call_index', graph_call_index,
                    'invocation_source', invocation_source
                  );
                ELSE
                  -- Lane-less rejected argument event kept historical but
                  -- renamed; route fields stay absent.
                  new_event := new_event || jsonb_build_object(
                    'tool', 'search_graph_relations',
                    'retrieval_lane', 'graph_relations',
                    'call_index', graph_call_index,
                    'invocation_source', invocation_source
                  );
                END IF;
              END IF;
              new_events := new_events || jsonb_build_array(new_event);
            END LOOP;
            UPDATE chat_run
               SET agent_trace = jsonb_build_object(
                     'version', 'native_tool_calling_agent_v3',
                     'events', new_events,
                     'budget', (
                       SELECT agent_configuration->'budget'
                         FROM chat_run
                        WHERE id = run_row.id
                     ),
                     'usage', run_row.agent_trace->'usage',
                     'outcome', run_row.agent_trace->>'outcome'
                   )
             WHERE id = run_row.id;
          END LOOP;
        END $$
        """
    )
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=_V1_DEFAULT,
    )
    _create_v3_constraints()


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_chat_run_agent_trace_v3"), "chat_run", type_="check"
    )
    op.drop_constraint(
        op.f("ck_chat_run_agent_configuration_v3"), "chat_run", type_="check"
    )
    op.execute(
        """
        DO $$
        DECLARE
          run_row RECORD;
          event_row RECORD;
        BEGIN
          FOR run_row IN
            SELECT id, agent_configuration, agent_trace, retrieval_strategy
              FROM chat_run
          LOOP
            IF run_row.retrieval_strategy->>'profile_version'
               = 'adaptive_graphiti_v3' THEN
              IF (
                (run_row.retrieval_strategy->>'graph_edge_limit')::integer <> 16
                OR (run_row.retrieval_strategy->>'graph_source_chunk_target')::integer
                   <> 12
                OR (run_row.retrieval_strategy->>'graph_source_chunk_limit')::integer
                   <> 16
                OR (run_row.retrieval_strategy->>'graph_call_timeout_seconds')::integer
                   <> 90
              ) THEN
                RAISE EXCEPTION
                  'v3 Graph parameters cannot be represented by v2 snapshots';
              END IF;
            END IF;
            IF run_row.agent_trace IS NOT NULL THEN
              FOR event_row IN
                SELECT value
                  FROM jsonb_array_elements(run_row.agent_trace->'events')
              LOOP
                IF event_row.value->>'retrieval_lane' = 'graph_relations' THEN
                  IF (
                    event_row.value->>'invocation_source' = 'agent'
                    AND event_row.value->>'call_index' IS NOT NULL
                    AND (event_row.value->>'call_index')::integer > 1
                  ) THEN
                    RAISE EXCEPTION
                      'v2 snapshots cannot represent a second Graph call';
                  END IF;
                  IF (
                    event_row.value->>'invocation_source'
                       NOT IN ('agent', 'legacy_guard')
                    OR event_row.value->>'duration_ms' IS NOT NULL
                    OR event_row.value->>'candidate_count' IS NOT NULL
                    OR event_row.value->>'path_count' IS NOT NULL
                    OR event_row.value->>'hydrated_chunk_count' IS NOT NULL
                    OR event_row.value->>'returned_chunk_count' IS NOT NULL
                    OR event_row.value->>'hop1_count' IS NOT NULL
                    OR event_row.value->>'hop2_count' IS NOT NULL
                    OR event_row.value->>'hop3_count' IS NOT NULL
                    OR (
                      event_row.value->>'route_reason_code' IS NOT NULL
                      AND event_row.value->>'route_reason_code'
                         NOT IN ('relation_chain', 'entity_alias',
                                 'cross_document_relation')
                    )
                    OR (
                      event_row.value->>'route_result_code' IS NOT NULL
                      AND event_row.value->>'route_result_code'
                         NOT IN ('admitted', 'no_evidence', 'not_ready',
                                 'timeout', 'unavailable', 'rejected')
                    )
                  ) THEN
                    RAISE EXCEPTION
                      'v3 Graph trace cannot be represented by v2 snapshots';
                  END IF;
                END IF;
              END LOOP;
            END IF;
          END LOOP;
        END $$
        """
    )
    op.execute(
        """
        UPDATE chat_run
           SET agent_configuration = jsonb_build_object(
                 'version', 'native_tool_calling_agent_v2',
                 'budget', jsonb_build_object(
                   'max_model_rounds', COALESCE(
                     (agent_configuration->'budget'->>'max_model_rounds')::integer,
                     8
                   )
                 )
               )
        """
    )
    op.execute(
        """
        UPDATE chat_run
           SET retrieval_strategy = CASE
                 WHEN retrieval_strategy->>'profile_version'
                      = 'adaptive_graphiti_v3'
                 THEN jsonb_build_object(
                   'profile_version', 'adaptive_graphiti_v2',
                   'strategy', 'exact_vector',
                   'top_k', retrieval_strategy->'top_k',
                   'rerank_mode', retrieval_strategy->'rerank_mode',
                   'router', 'native_agent_path_guard_v2',
                   'augmentation', 'graphiti_path_v3'
                 )
                 ELSE retrieval_strategy
               END
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
          run_row RECORD;
          event_row RECORD;
          new_events JSONB;
          new_event JSONB;
          new_reason TEXT;
          new_result TEXT;
        BEGIN
          FOR run_row IN
            SELECT id, agent_trace
              FROM chat_run
             WHERE agent_trace IS NOT NULL
          LOOP
            new_events := '[]'::jsonb;
            FOR event_row IN
              SELECT value
                FROM jsonb_array_elements(run_row.agent_trace->'events')
            LOOP
              new_event := event_row.value;
              IF new_event->>'retrieval_lane' = 'graph_relations' THEN
                new_reason := CASE new_event->>'route_reason_code'
                  WHEN 'cross_document_relation' THEN 'cross_document_relation_gap'
                  WHEN 'entity_alias' THEN 'entity_alias_gap'
                  WHEN 'relation_chain' THEN 'relation_chain_gap'
                  ELSE NULL
                END;
                new_result := CASE new_event->>'route_result_code'
                  WHEN 'admitted' THEN 'admitted'
                  WHEN 'no_evidence' THEN 'no_new_evidence'
                  WHEN 'not_ready' THEN 'not_ready'
                  WHEN 'timeout' THEN 'runtime_unavailable'
                  WHEN 'unavailable' THEN 'runtime_unavailable'
                  WHEN 'rejected' THEN 'rejected'
                  ELSE NULL
                END;
                IF (
                  new_event->>'route_reason_code' IS NOT NULL
                  AND new_reason IS NULL
                ) OR (
                  new_event->>'route_result_code' IS NOT NULL
                  AND new_result IS NULL
                ) THEN
                  RAISE EXCEPTION 'unmappable Graph event in chat_run %',
                    run_row.id;
                END IF;
                new_event := new_event
                  - ARRAY['call_index', 'invocation_source', 'duration_ms',
                          'candidate_count', 'path_count',
                          'hydrated_chunk_count', 'returned_chunk_count',
                          'hop1_count', 'hop2_count', 'hop3_count']
                  || jsonb_build_object(
                       'tool', 'graphiti_supplement',
                       'retrieval_lane', 'graphiti_supplement',
                       'route_reason_code', new_reason,
                       'route_result_code', new_result
                     );
              END IF;
              new_events := new_events || jsonb_build_array(new_event);
            END LOOP;
            UPDATE chat_run
               SET agent_trace = jsonb_build_object(
                     'version', 'native_tool_calling_agent_v2',
                     'events', new_events,
                     'budget', jsonb_build_object(
                       'max_model_rounds', COALESCE(
                         (run_row.agent_trace->'budget'->>'max_model_rounds')::integer,
                         8
                       )
                     ),
                     'usage', run_row.agent_trace->'usage',
                     'outcome', run_row.agent_trace->>'outcome'
                   )
             WHERE id = run_row.id;
          END LOOP;
        END $$
        """
    )
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=_V2_DEFAULT,
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_configuration_v2"),
        "chat_run",
        "(jsonb_typeof(agent_configuration) = 'object' "
        "AND (agent_configuration - ARRAY['version', 'budget']::text[]) = '{}'::jsonb "
        "AND agent_configuration->>'version' = 'native_tool_calling_agent_v2' "
        "AND jsonb_typeof(agent_configuration->'budget') = 'object' "
        "AND (agent_configuration->'budget') ? 'max_model_rounds' "
        "AND ((agent_configuration->'budget') - 'max_model_rounds'::text) "
        "= '{}'::jsonb "
        "AND CASE WHEN "
        "jsonb_typeof(agent_configuration->'budget'->'max_model_rounds') "
        "= 'number' AND (agent_configuration->'budget'->>'max_model_rounds') "
        "~ '^(0|[1-9][0-9]*)$' THEN "
        "(agent_configuration->'budget'->>'max_model_rounds')::integer "
        "BETWEEN 1 AND 12 ELSE FALSE END "
        "AND pg_column_size(agent_configuration) <= 4096) IS TRUE",
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v2"),
        "chat_run",
        "agent_trace IS NULL OR (("
        "jsonb_typeof(agent_trace) = 'object' "
        "AND (agent_trace - ARRAY['version', 'events', 'budget', 'usage', "
        "'outcome']::text[]) = '{}'::jsonb "
        "AND agent_trace->>'version' = 'native_tool_calling_agent_v2' "
        "AND jsonb_typeof(agent_trace->'events') = 'array' "
        "AND jsonb_array_length(agent_trace->'events') <= 32 "
        "AND jsonb_typeof(agent_trace->'budget') = 'object' "
        "AND (agent_trace->'budget') ? 'max_model_rounds' "
        "AND ((agent_trace->'budget') - 'max_model_rounds'::text) = '{}'::jsonb "
        "AND agent_trace->'budget' = agent_configuration->'budget' "
        "AND jsonb_typeof(agent_trace->'usage') = 'object' "
        "AND agent_trace->>'outcome' IN ('answered', 'partial', 'refused') "
        "AND pg_column_size(agent_trace) <= 65536) IS TRUE)",
    )


def _create_v3_constraints() -> None:
    op.create_check_constraint(
        op.f("ck_chat_run_agent_configuration_v3"),
        "chat_run",
        "(jsonb_typeof(agent_configuration) = 'object' "
        "AND (agent_configuration - ARRAY['version', 'budget']::text[]) = '{}'::jsonb "
        "AND agent_configuration->>'version' = 'native_tool_calling_agent_v3' "
        "AND jsonb_typeof(agent_configuration->'budget') = 'object' "
        "AND (agent_configuration->'budget') ? 'max_model_rounds' "
        "AND (agent_configuration->'budget') ? 'max_graph_calls' "
        "AND ((agent_configuration->'budget') - "
        "ARRAY['max_model_rounds', 'max_graph_calls']::text[]) = '{}'::jsonb "
        "AND CASE WHEN "
        "jsonb_typeof(agent_configuration->'budget'->'max_model_rounds') "
        "= 'number' AND (agent_configuration->'budget'->>'max_model_rounds') "
        "~ '^(0|[1-9][0-9]*)$' THEN "
        "(agent_configuration->'budget'->>'max_model_rounds')::integer "
        "BETWEEN 1 AND 12 ELSE FALSE END "
        "AND CASE WHEN "
        "jsonb_typeof(agent_configuration->'budget'->'max_graph_calls') "
        "= 'number' AND (agent_configuration->'budget'->>'max_graph_calls') "
        "~ '^(0|[1-9][0-9]*)$' THEN "
        "(agent_configuration->'budget'->>'max_graph_calls')::integer "
        "BETWEEN 1 AND 2 ELSE FALSE END "
        "AND pg_column_size(agent_configuration) <= 4096) IS TRUE",
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        "agent_trace IS NULL OR (("
        "jsonb_typeof(agent_trace) = 'object' "
        "AND (agent_trace - ARRAY['version', 'events', 'budget', 'usage', "
        "'outcome']::text[]) = '{}'::jsonb "
        "AND agent_trace->>'version' = 'native_tool_calling_agent_v3' "
        "AND jsonb_typeof(agent_trace->'events') = 'array' "
        "AND jsonb_array_length(agent_trace->'events') <= 32 "
        "AND jsonb_typeof(agent_trace->'budget') = 'object' "
        "AND (agent_trace->'budget') ? 'max_model_rounds' "
        "AND (agent_trace->'budget') ? 'max_graph_calls' "
        "AND ((agent_trace->'budget') - "
        "ARRAY['max_model_rounds', 'max_graph_calls']::text[]) = '{}'::jsonb "
        "AND agent_trace->'budget' = agent_configuration->'budget' "
        "AND jsonb_typeof(agent_trace->'usage') = 'object' "
        "AND agent_trace->>'outcome' IN ('answered', 'partial', 'refused') "
        "AND pg_column_size(agent_trace) <= 65536) IS TRUE)",
    )