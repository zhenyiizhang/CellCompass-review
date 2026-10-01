from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..utils.llm_providers import normalize_llm_provider
from ..tools.context_compaction import (
    compact_history,
    estimate_tokens,
    get_context_policy,
    is_context_overflow_error,
    project_compacted_history,
    record_compaction_stats,
    record_microcompact_stats,
    should_compact,
)

_LEGACY_ASSISTANT_REASONING_CONTENT = (
    "Legacy assistant message predates provider reasoning_content capture."
)
_LEGACY_TOOL_REASONING_CONTENT = (
    "Legacy tool-call message predates provider reasoning_content capture."
)


@dataclass
class PreparedPrompt:
    prompt_messages: List[BaseMessage]
    persisted_history: List[BaseMessage] | None = None
    history_updates: List[BaseMessage] = field(default_factory=list)
    compaction_meta: Dict[str, Any] = field(default_factory=dict)


class RuntimeMiddleware:
    def __init__(self, state: Dict[str, Any], llm, event_sink=None):
        self.state = state
        self.llm = llm
        self.event_sink = event_sink

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_sink:
            try:
                self.event_sink(event_type, payload)
            except Exception:
                pass

    @staticmethod
    def _sanitize_dangling_tool_results(messages: List[BaseMessage]) -> tuple[List[BaseMessage], int]:
        seen_tool_call_ids = set()
        seen_tool_response_ids = set()
        sanitized: List[BaseMessage] = []
        removed = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", None) or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        seen_tool_call_ids.add(str(tc["id"]))
                sanitized.append(msg)
                continue
            if isinstance(msg, ToolMessage):
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
                if tool_call_id and tool_call_id not in seen_tool_call_ids:
                    removed += 1
                    continue
                if tool_call_id:
                    seen_tool_response_ids.add(tool_call_id)
                sanitized.append(msg)
                continue
            if isinstance(msg, HumanMessage):
                additional = dict(getattr(msg, "additional_kwargs", {}) or {})
                if additional.get("synthetic_tool_media"):
                    tool_call_id = str(additional.get("tool_call_id") or "")
                    if not tool_call_id or tool_call_id not in seen_tool_response_ids:
                        removed += 1
                        continue
                sanitized.append(msg)
                continue
            sanitized.append(msg)
        return sanitized, removed

    def _needs_reasoning_content_for_tool_calls(self) -> bool:
        provider = normalize_llm_provider(
            self.state.get("llm_provider")
            or (self.state.get("initial_config") or {}).get("llm_provider")
            or (self.state.get("runtime_config") or {}).get("provider")
            or ""
        )
        if provider not in {"xiaomi", "deepseek"}:
            return False
        extra_body = dict(getattr(self.llm, "extra_body", {}) or {})
        thinking = extra_body.get("thinking")
        if not isinstance(thinking, dict):
            return False
        return str(thinking.get("type") or "").strip().lower() == "enabled"

    def _reasoning_content_provider(self) -> str:
        provider = normalize_llm_provider(
            self.state.get("llm_provider")
            or (self.state.get("initial_config") or {}).get("llm_provider")
            or (self.state.get("runtime_config") or {}).get("provider")
            or ""
        )
        return provider if provider in {"xiaomi", "deepseek"} else ""

    def _ensure_tool_call_reasoning_content(self, messages: List[BaseMessage]) -> tuple[List[BaseMessage], int]:
        """Satisfy thinking-mode providers that require reasoning traces.

        Some OpenAI-compatible thinking APIs reject assistant history messages
        without `reasoning_content`. New responses keep the real field via the
        llm_factory round-trip patch; resumed legacy histories may lack it, so
        add a neutral protocol placeholder only for those messages.
        """

        if not self._needs_reasoning_content_for_tool_calls():
            return messages, 0

        provider = self._reasoning_content_provider()
        patched: List[BaseMessage] = []
        added = 0
        for msg in messages:
            additional = dict(getattr(msg, "additional_kwargs", {}) or {})
            raw_tool_calls = additional.get("tool_calls")
            has_tool_calls = bool(getattr(msg, "tool_calls", None) or []) or bool(raw_tool_calls or [])
            if not isinstance(msg, AIMessage):
                patched.append(msg)
                continue
            if "reasoning_content" in additional:
                patched.append(msg)
                continue
            additional["reasoning_content"] = (
                _LEGACY_TOOL_REASONING_CONTENT if has_tool_calls else _LEGACY_ASSISTANT_REASONING_CONTENT
            )
            additional["_cytobridge_reasoning_content_provider"] = provider
            patched.append(
                AIMessage(
                    content=getattr(msg, "content", ""),
                    additional_kwargs=additional,
                    tool_calls=list(getattr(msg, "tool_calls", []) or []),
                    id=getattr(msg, "id", None),
                    name=getattr(msg, "name", None),
                    response_metadata=dict(getattr(msg, "response_metadata", {}) or {}),
                )
            )
            added += 1
        return patched, added

    @staticmethod
    def _conversation_messages_for_budget(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Exclude the per-turn static system prompt from compaction budgeting."""
        if messages and isinstance(messages[0], SystemMessage):
            return list(messages[1:])
        return list(messages)

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            number = int(float(value))
        except Exception:
            return None
        return number if number > 0 else None

    def _usage_calibrated_budget_tokens(self, estimated_tokens: int) -> tuple[int, str]:
        """Use provider-reported usage as the primary context-pressure signal.

        Some transports, especially stateful sidecars, report prompt tokens
        inconsistently, so total_tokens is preferred when present. The local
        estimate remains a floor because tool results may have been appended
        after the last model response.
        """
        usage = dict(self.state.get("llm_context_usage") or {})
        total = self._positive_int(usage.get("total_tokens") or usage.get("budget_tokens"))
        prompt = self._positive_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        usage_tokens = total or prompt
        if usage_tokens is None:
            return int(estimated_tokens), "estimate"
        calibrated = max(int(estimated_tokens), int(usage_tokens))
        source = "usage_total_tokens" if total is not None else "usage_prompt_tokens"
        if calibrated != int(usage_tokens):
            source = f"{source}+estimate_floor"
        return calibrated, source

    def _record_compacted_budget_usage(self, after_tokens: int, *, source: str) -> None:
        self.state["llm_context_usage"] = {
            "budget_tokens": int(after_tokens),
            "total_tokens": int(after_tokens),
            "prompt_tokens": int(after_tokens),
            "source": source,
        }

    def prepare_messages(self, prompt_messages: List[BaseMessage]) -> PreparedPrompt:
        policy = get_context_policy(self.state.get("context_policy"))
        sanitized_messages, removed_dangling = self._sanitize_dangling_tool_results(list(prompt_messages))
        sanitized_messages, added_reasoning = self._ensure_tool_call_reasoning_content(sanitized_messages)
        persisted_history = list(sanitized_messages[1:]) if removed_dangling and sanitized_messages else None
        if removed_dangling:
            self._emit(
                "status",
                {
                    "agent": "cytobridge",
                    "message": f"Recovered history by dropping {removed_dangling} dangling tool result message(s).",
                },
            )
        if added_reasoning:
            self._emit(
                "status",
                {
                    "agent": "cytobridge",
                    "message": (
                        f"Patched {added_reasoning} legacy tool-call message(s) with provider-required "
                        "reasoning_content placeholders."
                    ),
                },
            )
        projected_messages, projection_meta = project_compacted_history(
            sanitized_messages,
            preserve_leading_system=True,
        )
        if not policy.get("enabled", True):
            return PreparedPrompt(
                prompt_messages=projected_messages,
                persisted_history=persisted_history,
                compaction_meta={"projection_meta": projection_meta},
            )
        budget_messages = (
            self._conversation_messages_for_budget(projected_messages)
            if projection_meta.get("has_boundary")
            else projected_messages
        )
        estimated_before = estimate_tokens(budget_messages)
        before, budget_source = self._usage_calibrated_budget_tokens(estimated_before)
        context_window = int(policy.get("context_window", 256000))
        effective_context_window = int(policy.get("effective_context_window") or context_window)
        context_window_cap = int(policy.get("context_window_cap", effective_context_window))
        allow_large_context_window = bool(policy.get("allow_large_context_window", False))
        if not should_compact(
            before,
            effective_context_window,
            float(policy.get("trigger_ratio", 0.82)),
        ):
            return PreparedPrompt(
                prompt_messages=projected_messages,
                persisted_history=persisted_history,
                compaction_meta={
                    "mode": "none",
                    "before_tokens": before,
                    "estimated_before_tokens": estimated_before,
                    "budget_token_source": budget_source,
                    "after_tokens": before,
                    "context_window": context_window,
                    "effective_context_window": effective_context_window,
                    "context_window_cap": context_window_cap,
                    "allow_large_context_window": allow_large_context_window,
                    "projection_meta": projection_meta,
                },
            )
        compacted, compact_meta = compact_history(
            sanitized_messages,
            keep_last_turns=int(policy.get("keep_last_turns", 6)),
            llm=self.llm,
            llm_enabled=bool(policy.get("llm_compact_enabled", True)),
            llm_input_max_chars=int(policy.get("llm_compact_input_max_chars", 24000)),
            short_history_max_tool_chars=int(policy.get("max_tool_chars", 1500)),
            preserve_leading_system=True,
            state=self.state,
            trigger="auto",
            force_full=budget_source != "estimate",
            microcompact_enabled=bool(policy.get("microcompact_enabled", True)),
            context_window=effective_context_window,
            trigger_ratio=float(policy.get("trigger_ratio", 0.82)),
        )
        compact_meta = {
            **dict(compact_meta),
            "before_tokens": before,
            "estimated_before_tokens": estimated_before,
            "budget_token_source": budget_source,
            "context_window": context_window,
            "effective_context_window": effective_context_window,
            "context_window_cap": context_window_cap,
            "allow_large_context_window": allow_large_context_window,
        }
        compacted, removed_after_compaction = self._sanitize_dangling_tool_results(compacted)
        after = estimate_tokens(self._conversation_messages_for_budget(compacted))
        history_updates = list(compact_meta.get("history_updates") or [])
        micro_meta = dict(compact_meta.get("microcompact") or {})
        if not history_updates and str(compact_meta.get("mode") or "none") == "none":
            return PreparedPrompt(
                prompt_messages=compacted,
                persisted_history=persisted_history,
                history_updates=[],
                compaction_meta=dict(compact_meta),
            )
        if micro_meta.get("mode") == "microcompact" and not history_updates:
            persisted_micro_history = (
                list(compacted[1:])
                if compacted and isinstance(compacted[0], SystemMessage)
                else list(compacted)
            )
            self.state["compaction_stats"] = record_microcompact_stats(
                self.state.get("compaction_stats"),
                before_tokens=int(micro_meta.get("before_tokens", before)),
                after_tokens=int(micro_meta.get("after_tokens", after)),
            )
            self._record_compacted_budget_usage(after, source="estimate_after_microcompact")
            self._emit(
                "context_microcompacted",
                {
                    "agent": "cytobridge",
                    "before_tokens": int(micro_meta.get("before_tokens", before)),
                    "after_tokens": int(micro_meta.get("after_tokens", after)),
                    "tokens_saved": int(micro_meta.get("tokens_saved", 0)),
                    "compacted_tool_messages": int(micro_meta.get("compacted_tool_messages", 0)),
                    "cleared_media_messages": int(micro_meta.get("cleared_media_messages", 0)),
                    "reason": "runtime_v2_pre_invoke_threshold",
                },
            )
            return PreparedPrompt(
                prompt_messages=compacted,
                persisted_history=persisted_micro_history,
                history_updates=[],
                compaction_meta={**dict(compact_meta), "budget_token_source": budget_source},
            )
        stats_before_full = self.state.get("compaction_stats")
        if micro_meta.get("mode") == "microcompact":
            stats_before_full = record_microcompact_stats(
                stats_before_full,
                before_tokens=int(micro_meta.get("before_tokens", before)),
                after_tokens=int(micro_meta.get("after_tokens", after)),
            )
        self.state["compaction_stats"] = record_compaction_stats(
            stats_before_full,
            before_tokens=int(compact_meta.get("before_tokens", before)),
            after_tokens=int(compact_meta.get("after_tokens", after)),
            reason="runtime_v2_pre_invoke_threshold",
            mode=str(compact_meta.get("mode", "rule") or "rule"),
        )
        self._record_compacted_budget_usage(after, source="estimate_after_full_compact")
        self._emit(
            "context_compacted",
            {
                "agent": "cytobridge",
                "before_tokens": int(compact_meta.get("before_tokens", before)),
                "after_tokens": int(compact_meta.get("after_tokens", after)),
                "reason": "runtime_v2_pre_invoke_threshold",
                "mode": compact_meta.get("mode", "rule"),
                "removed_messages": compact_meta.get("removed_messages"),
                "preserved_tail_count": compact_meta.get("preserved_tail_count"),
                "dropped_dangling_tool_results": removed_dangling + removed_after_compaction,
                "summary_text": compact_meta.get("summary_text", ""),
                "summary_core": compact_meta.get("summary_core", ""),
                "microcompact": micro_meta,
            },
        )
        # Persist only the compact boundary/summary/rehydrate messages. The
        # preserved tail is stored inside the boundary metadata and rehydrated by
        # project_compacted_history(), so storing it twice would grow snapshots
        # again and duplicate recent turns on the next projection.
        persisted_compacted_history = list(history_updates)
        return PreparedPrompt(
            prompt_messages=compacted,
            persisted_history=persisted_compacted_history,
            history_updates=history_updates,
            compaction_meta={**dict(compact_meta), "budget_token_source": budget_source},
        )

    @staticmethod
    def is_context_error(exc: Exception) -> bool:
        return is_context_overflow_error(exc)
