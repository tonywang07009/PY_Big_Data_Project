# Review Protocol

## Language Rule

- Persisted documents are English.
- Questions to the human are Traditional Chinese.
- Human decisions are summarized in English in `agent_doc/decisions/decision_log.md`.

## Diver / Counter Model

Every micro-task uses two review roles.

### Diver

The Diver proposes the task approach and produces the first artifact.

Required notes:

- objective
- input
- output
- method
- assumptions
- risks
- reproducibility notes

### Counter

The Counter challenges the Diver output before the task passes.

Required checks:

- fit with QFD and downstream needs
- runtime stability and rerun safety
- human maintainability
- leakage risk
- artifact existence and path correctness

## Task Verdict

Every task receives:

```json
{
  "task_id": "T0-A",
  "verdict": "PASS | CONDITIONAL | FAIL",
  "fit_score": 1,
  "stability_score": 1,
  "maintainability_score": 1,
  "counter_notes": []
}
```

Scores use a 1-5 scale. Any score below 3 is `FAIL`; any score equal to 3 is `CONDITIONAL`.

## Human Brainstorming Trigger

Start a Traditional Chinese brainstorming discussion when:

- data is missing or semantically unclear
- multiple model or preprocessing choices are valid
- a QFD requirement conflicts with implementation cost
- a task is `CONDITIONAL` or `FAIL`
- the next choice affects scientific validity or maintainability

Brainstorming format:

1. Context
2. Problem
3. Options
4. Tradeoffs
5. Recommended default
6. Human decision
7. Recorded outcome

