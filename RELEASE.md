# Releasing disko

`disko` is the head of a three-repo release chain. One command runs the
exhaustive tests, bumps the version, publishes to PyPI **through CI**, and then
releases the two downstream repos in the order their dependencies require:

```console
$ uv run invoke release --dry-run     # print the plan, change nothing
$ uv run invoke release               # prompts once, then runs everything
$ uv run invoke release --level minor --yes
```

```
1. test       uv sync + uv lock --check → flake8 fatal gate → pytest on
              3.11, 3.12 and 3.13 → CLI smoke for all four entry points →
              uv build
2. bump       pyproject version + CHANGES.md `## Unreleased` → `## <ver> (<date>)`
3. publish    commit + tag v<version> + push → .github/workflows/package.yaml
              builds and publishes to PyPI → wait for that run to go green
4. tools      invoke -r ~/tart-telescope/tools release --disko-version=<version>
              (exact pin, image tests, tag → GHCR, wait for the image)
5. tart_cargo invoke -r ~/tart-telescope/tart_cargo release --tools-tag=<new tag>
              (poll GHCR, bump the base image, tests, tag → PyPI)
```

Stages 4 and 5 are *not* optional garnish: the tools image pip-installs disko
from PyPI, and tart_cargo's cab image version must exist before it is
referenced. Each stage waits for the previous one's artifact to actually
appear, which is why the chain takes a while (see [Timing](#timing)).

- tools release details: `../tart-telescope/tools/RELEASE.md`
- tart_cargo release details: `../tart-telescope/tart_cargo/RELEASE.md`

## Prerequisites

| Need | Why |
|---|---|
| `uv` and `uv run invoke …` | `invoke` and `plumbum` are in the `dev` dependency group, so the project venv always carries them |
| push rights to `tmolteno/disko`, `tart-telescope/tools`, `tart-telescope/tart_cargo` | the chain tags and pushes in all three |
| `gh` CLI, authenticated | `publish` waits for the package.yaml run; without it pass `--no-wait` |
| `PYPI_API_TOKEN` secret in the disko repo | package.yaml's publish step (already configured) |
| `invoke` + `plumbum` on your interpreter | the two downstream task files import them (installed globally today) |
| downstream checkouts | default `~/tart-telescope/{tools,tart_cargo}`; override with `DISKO_TOOLS_REPO` / `DISKO_TART_CARGO_REPO` |
| `docker` (optional) | tools' tests build the image; its release can be run with `--skip-tests` by whoever cannot |

## The tasks

```console
$ uv run invoke --list
  bump      Bump the pyproject version and open a CHANGES.md section for it.
  publish   Commit the bump, tag vX.Y.Z and push: CI publishes to PyPI.
  release   Full release: test -> bump -> CI publish -> tools -> tart_cargo.
  status    Show where every repo in the chain currently stands.
  test      Exhaustive local tests — everything a release must pass first.
```

### `test` — the gate

Runs `uv sync --dev`, `uv lock --check` (lock agrees with pyproject), the
flake8 **fatal** gate (`--select=E9,F63,F7,F82`), the informational style count
(baseline 59; never fails the run), `pytest` on **each** of 3.11/3.12/3.13
(matching `.github/workflows/testing.yaml`), then CLI smoke tests:

```text
disko --version
disko --ms test_data/test.ms --nside 32 --dir <tmp> --HDF smoke.h5 --SVG
disko_draw <tmp>/smoke.h5 --SVG <tmp>/smoke.svg
disko --ms test_data/test.ms --nside 4 --dir <tmp> --PDF        # issue #12 defaults
disko --ms test_data/test.ms --nside 4 --dir <tmp> --no-regularization --HDF …
disko_svd --help ; disko_bayes --help
```

and finally `uv build` (the distributions package.yaml will rebuild). `--quick`
drops to one interpreter and skips the build.

> **CI note:** `testing.yaml` has been disabled since 2026-06-18, so
> `invoke test` is the only test gate before a release. Run it before you tag
> anything, including manual releases.

### `bump` — version + changelog

`--level patch|minor|major` (default `patch`) rewrites `version = "…"` in
`pyproject.toml` and converts the `## Unreleased` heading in `CHANGES.md` to
`## <version> (<YYYY-MM-DD>)`, leaving a fresh empty `## Unreleased` above it.
`--dry-run` prints the new version and the commits that would land in it.

### `publish` — tag push, CI does the upload

Refuses if the tag already exists or if tracked files other than
`pyproject.toml`/`CHANGES.md` are modified (untracked scratch is only
reported). Then: `git add` both files → `Release <version>` commit → annotated
`v<version>` tag → `git push --follow-tags`. `.github/workflows/package.yaml`
triggers on that tag, runs `uv build` and uploads with
`pypa/gh-action-pypi-publish` using the `PYPI_API_TOKEN` secret.

`--wait` (default) polls `gh run list --workflow package.yaml` until the run
on that tag completes successfully — required when chaining, because stage 4
installs disko from PyPI. `--no-wait` skips it; `--yes` skips the prompt.

### `status` — where the chain stands

Prints disko's version and last tag, and for each downstream repo its last
tag, tracked-modification count and whether `tasks.py` exists.

## Versioning and changelog conventions

- Semantic-ish: `patch` for fixes, `minor` for behaviour changes (issue #12
  changed defaults → a minor bump was the right call), `major` for breaks.
- Tags are `vMAJOR.MINOR.PATCH` and **are** the publish trigger — never push
  one by hand unless you intend to publish that exact commit.
- `CHANGES.md` keeps everything under `## Unreleased` while work lands; the
  bump turns that section into the release section. Write entries as you go,
  not at release time.

## Safety

| Guard | Behaviour |
|---|---|
| prompts | `release` asks once, then passes consent to the stages; `publish` asks when run alone; `--yes` bypasses |
| `--dry-run` | `release`, `bump` print the full plan and exit without touching anything |
| clean tree | only `pyproject.toml` and `CHANGES.md` may be modified at publish time |
| no re-tag | publishing an existing tag is refused |
| stage waits | PyPI → GHCR → downstream, each verified by polling before the next stage |

## Manual fallback

If you cannot use the tasks (broken venv, no plumbum), the chain is four
commands per repo:

```console
$ uv run pytest && uv run flake8 disko --select=E9,F63,F7,F82 --count
# edit pyproject.toml version; move the CHANGES.md Unreleased section
$ git commit -am "Release X.Y.Z"
$ git tag -a vX.Y.Z -m "Release X.Y.Z" && git push --follow-tags origin master
$ gh run list --workflow package.yaml --watch        # wait for PyPI
# then release tools, then tart_cargo — see their RELEASE.md files
```

## Timing

- `test`: ~1–3 min normally, longer the first time on a new interpreter
  (uv downloads 3.12/3.13).
- `publish` wait: up to 30 min (usually ~1 min).
- **tools**: image build + `make test-all`, then up to 60 min for GHCR — the
  amd64 leg runs on `ubuntu-latest` but arm64 runs on a self-hosted runner.
- **tart_cargo**: recipe dry-runs (seconds) + publish wait.

## After the release

1. `uv run invoke status` — all three should show the new tag, clean.
2. Confirm the package: <https://pypi.org/project/disko/>.
3. Confirm the image tag exists (stage 4 waits for this, but if you used
   `--no-wait`: `docker manifest inspect ghcr.io/tart-telescope/tools:<tag>`).
4. Confirm tart_cargo's `genesis/tart-cargo-base.yml` points at that image and
   its release is on PyPI.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `publish workflow failed: <url>` | the tag is already pushed — **do not re-tag**. Fix on a branch, bump again, release the new version. |
| `gh CLI is required to wait…` | install `gh`, or pass `--no-wait` and watch CI yourself |
| `… has no tasks.py — cannot run invoke release` | downstream checkout missing/out of date: `git pull` in that repo (or set the `DISKO_*` env vars to the right paths) |
| `working tree has unrelated changes` | commit or stash; untracked files are only reported, never blocking |
| prompt does not appear / hangs | `release` reads from the terminal; use `--yes` in non-interactive runs |
| 3.12/3.13 step is slow | uv is fetching those interpreters once; it caches |
