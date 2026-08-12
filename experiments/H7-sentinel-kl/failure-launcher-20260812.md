# H7 sentinel-KL launcher failure — 2026-08-12

**Status:** launcher failure; non-result; original attempt consumed. This is
immutable engineering evidence for the one authorization tagged
`h7-authorized-20260812`. It is classified
`measurement_failed` / `non_evaluable`, with `publication_valid: false` and
`profile_cluster_count: 115`. It supplies no K3, KL, fairness, robustness,
efficacy, controller, model, or publication result.

## Authorized invocation

| Field | Value |
| --- | --- |
| Authorization commit | `dab18a96e1cddc5d65cf7b584d5f09b5d2c4c858` |
| Authorization tag | `h7-authorized-20260812` |
| Modal app | `ap-8ffhsYgqERSLniCGF7vaW4` |
| App creation | `2026-08-12 17:41:47.776958 IST` (epoch `1786536707.776958`) |
| Working directory | `C:\Kaustubh\ast-asr-worktrees\fair-cispo-tiny` |
| Command | `uvx modal run scripts/modal_h7_sentinel.py` |
| Shell elapsed time | `3.4 s` |
| Literal error | `'charmap' codec can't encode character '\u2713' in position 0: character maps to <undefined>` |

## Environment and root cause boundary

The local launcher environment was Modal `1.5.3`, uv `0.11.28`, and uvx
Python `3.12.3`. Its effective `sys.stdout.encoding`, `sys.stderr.encoding`,
and preferred encoding were `cp1252`; `sys.flags.utf8_mode` was `0`. Windows'
active code page was `437`, even though PowerShell/the console reported UTF-8.
The `✓` character in the literal error is therefore not encodable by the actual
uvx launcher streams.

The Modal `1.5.3` local ordering is exact: `_init_local_app_new` completed,
then Rich `_step_completed_text` attempted to render `✓`, and the launcher
raised before `_create_all_objects`. The app object therefore exists, while its
layout objects, function IDs, and class IDs remain empty. This is local
launcher/console evidence, not remote-execution evidence. It neither attributes
the error to module/image construction nor proves that no transient image
preparation occurred.

## Remote-state inspection

Immediately after the failed command, app
`ap-8ffhsYgqERSLniCGF7vaW4` was `APP_STATE_EPHEMERAL`, with `stopped_at: null`.
The app had empty layout objects, function IDs, class IDs, task history, logs,
and containers. The observed counts were `tasks: 0`, `history: []`, `logs:
empty`, and `containers: []`. Function, call, container, and image IDs are all
`null`.

A later read-only app-list observation found the same app naturally transitioned
to `stopped` at `2026-08-12 17:44:48+05:30`, still with `tasks: 0`. No explicit
stop action was issued. This later state does not overwrite the immediate
`APP_STATE_EPHEMERAL` / `stopped_at: null` snapshot above.

The H7 reserved run root
`/artifacts/profile-h7-fixed-policy-sentinel-kl-s2028-20260812` was absent.
Consequently there is no source manifest, input-lock replay, observed
image/function execution, model load, feature extraction, scoring forward pass,
`terminal_decision.json`, `failure.json`, or output artifact. Image start is
classified `unresolved_no_id_observed`: the inspection does not prove that no
transient preparation occurred, only that no image ID or remote execution
artifact was observed.

## Terminal boundary

The original H7 name is consumed and may not be retried, reused, deleted, or
recovered implicitly. This record authorizes no explicit stop or mutation of
the old app; a natural backend state transition may occur independently. A new
invocation requires the separately locked `recovery-protocol.md`, a new
reviewed code commit/tag and implementation-bound authorization, a new app, and
new r1 run/output names. This failure record itself authorizes none of those
actions.
