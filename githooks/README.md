# Committed Git Hooks

This directory holds the repo's pre-commit / pre-push hooks. Unlike
`.git/hooks/`, this directory IS committed so every clone gets the
same enforcement.

## Install on first clone

```bash
./scripts/install-hooks.sh
```

That one-shot command points `core.hooksPath` at this directory. All
subsequent `git commit` / `git push` operations invoke the scripts
below.

## Hook dispatcher (multiplex via directory)

Git only calls one executable per hook phase. When multiple checks
need to run, we use a dispatcher pattern: `./githooks/pre-commit` is
an executable shim that iterates files in `./githooks/pre-commit/`
and runs each in alpha order. Scripts use numeric prefixes (`10-`,
`20-`, `50-`) to control ordering.

## Current hooks

### `pre-commit/`
- **`50-agent-mail.py`** — MCP Agent Mail guard. Coordinates with
  other active swarm agents via the mcp-agent-mail service before
  allowing a commit. Blocks commits that touch files another agent
  is currently holding a reservation on. Can be bypassed by setting
  `AGENT_MAIL_GUARD_MODE=warn` (development mode only).

### `pre-push/`
- **`50-agent-mail.py`** — MCP Agent Mail guard. Pre-push sibling of
  the pre-commit hook. Warns (or blocks, per `AGENT_MAIL_GUARD_MODE`)
  on pushes that would conflict with remote swarm state.

## Adding new hooks

1. Create `./githooks/<phase>/<NN>-<name>.sh` (or `.py`).
2. `chmod +x` the file (git on some platforms won't execute
   non-executable hooks).
3. Commit it.
4. Existing clones pick it up on next `git commit` / `git push`
   once they've run `./scripts/install-hooks.sh` once.

## Why committed hooks

Agents and contributors cloning fresh get the same discipline
automatically. Without this pattern, `.git/hooks/` is local-only
and a fresh clone has zero enforcement.

Tracked under br-frankenpandas-pa2y.
