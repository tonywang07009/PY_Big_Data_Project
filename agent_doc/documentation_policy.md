# Documentation Policy

## Naming Principles

Names should be readable, specific, and low-friction for humans.

- Use `agent_doc/` for agent-facing governance and routing documents.
- Use `source_materials/` for original references and proposal files.
- Use `pipeline_tools/` for executable pipeline modules.
- Use `model_outputs/` for machine-readable artifacts.
- Use `reports/` for human-facing deliverables.
- Keep `data/raw/` for read-only source data.

## Document Routing

| Path | Purpose |
|---|---|
| `project.md` | short system overview and router |
| `agent_doc/project_brief.md` | project scope and source summary |
| `agent_doc/qfd_matrix.md` | research-to-engineering mapping |
| `agent_doc/review_protocol.md` | Diver/Counter and brainstorming rules |
| `agent_doc/testing_workflow.md` | task, gate, and system testing policy |
| `agent_doc/gates/` | gate-specific implementation discussion |
| `agent_doc/decisions/` | human-approved decisions |
| `agent_doc/agent_memory/` | agent execution logs |
| `reports/` | final human-facing reports |

## Length Policy

- Keep `project.md` short enough to read as a router.
- Put detailed implementation rules in the smallest relevant `agent_doc/` file.
- Avoid duplicating the same rule across many files; route to the source of truth instead.

