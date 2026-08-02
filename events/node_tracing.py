# events/node_tracing.py
# Wraps LangGraph node functions so every node emits a start/complete/error
# event over the existing 'agent.status' Kafka topic — this is what lets a
# failure trace back to the exact node and decision, not just which of the
# 4 agents was running. No new topics: everything rides in agent.status's
# `data` payload.

import time
import traceback

from events.agent_events import emit_agent_status


def traced_node(agent: str, node_name: str, watch=None):
    """
    Decorator for a LangGraph node function `fn(state) -> dict`.

    Emits agent.status events:
      - '<node_name>:started'   before the node runs
      - '<node_name>:completed' after it returns, with the `watch`-listed
        keys from its output plus elapsed_ms
      - '<node_name>:error'     if it raises, with a truncated traceback —
        then re-raises so pipeline behavior is unchanged; this only adds
        observability.
    """
    watch = watch or []

    def decorator(fn):
        def wrapped(state):
            emit_agent_status(agent, f'{node_name}:started', {'node': node_name})
            start = time.time()
            try:
                result = fn(state)
            except Exception as e:
                emit_agent_status(agent, f'{node_name}:error', {
                    'node': node_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()[-2000:],
                    'elapsed_ms': round((time.time() - start) * 1000, 1),
                })
                raise

            details = {
                'node': node_name,
                'elapsed_ms': round((time.time() - start) * 1000, 1),
            }
            for key in watch:
                if key in result:
                    details[key] = result[key]

            emit_agent_status(agent, f'{node_name}:completed', details)
            return result

        return wrapped

    return decorator


def node_tracer(agent: str):
    """
    Returns a small helper bound to one agent name, so a graph builder can
    replace `graph.add_node(name, fn)` with `tracer(graph, name, fn)`
    without repeating the agent name at every call site.
    """
    def add_traced_node(graph, name, fn, watch=None):
        graph.add_node(name, traced_node(agent, name, watch=watch)(fn))

    return add_traced_node
