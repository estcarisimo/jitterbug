# Online mode

The back ends emit `StreamingEvent` objects of three kinds: `change_point`, `verdict`
(provisional or final) and, from `OnlineJitterbug` with
`streaming.revise_change_points`, `revision` (the last change point moved from
`revised_from` to `start_epoch`). `StateChangeTracker` emits `StateChangeEvent` objects.
See [Online mode](../ONLINE_MODE.md) for the method and the measurements.

::: jitterbug.streaming.online_analyzer

::: jitterbug.streaming.window_backend

::: jitterbug.streaming.verdict

::: jitterbug.streaming.state_changes

::: jitterbug.streaming.replay
