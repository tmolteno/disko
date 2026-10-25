"""
Release automation for disko: invoke tasks driven by plumbum commands.

    uv run invoke --list
    uv run invoke test                  # exhaustive local test run
    uv run invoke test --quick          # one interpreter, no package build
    uv run invoke bump --level minor --dry-run
    uv run invoke publish --yes         # commit + tag + push -> CI publishes
    uv run invoke release               # the whole chain (prompts first)
    uv run invoke status

`release` runs five stages, in this order:

    1. test       exhaustive local tests: uv lock --check, the flake8
                   fatal-error gate, the full pytest matrix on every
                   supported interpreter (3.11/3.12/3.13, matching
                   .github/workflows/testing.yaml), CLI smoke tests for
                   every entry point including the issue #12 defaults and
                   the opt-in unregularized path, and a package build.
    2. bump       pyproject version, and CHANGES.md's `## Unreleased`
                   section becomes `## <version> (<date>)`.
    3. publish    commit + annotated tag v<version> + push. The tag is what
                   triggers .github/workflows/package.yaml, which builds the
                   distributions and publishes them to PyPI — no local
                   twine. The task then waits for that workflow run to go
                   green, because the next stage's image installs disko
                   from PyPI and would otherwise race the publish.
    4. tools      `invoke -r <tools repo> release` — tags the tools image,
                   which builds and pushes ghcr.io/tart-telescope/tools.
    5. tart_cargo `invoke -r <tart_cargo repo> release` — points the
                   stimulus base image at that new tag and publishes the
                   tart_cargo package.

Stages 4 and 5 need a tasks.py in their own repos (both ship one). Repo
locations default to ~/tart-telescope/{tools,tart_cargo} and can be
overridden with DISKO_TOOLS_REPO / DISKO_TART_CARGO_REPO.

Nothing pushes without a confirmation prompt unless --yes is given, and
--dry-run prints the full plan without touching anything.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from invoke import Exit, task
from plumbum import colors, local

# ---------------------------------------------------------------- locations

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGES = ROOT / "CHANGES.md"
TEST_MS = ROOT / "test_data" / "test.ms"

# Same interpreters .github/workflows/testing.yaml builds against.
PYTHON_MATRIX = ("3.11", "3.12", "3.13")

TOOLS_REPO = Path(os.environ.get("DISKO_TOOLS_REPO", Path.home() / "tart-telescope" / "tools"))
CARGO_REPO = Path(os.environ.get("DISKO_TART_CARGO_REPO", Path.home() / "tart-telescope" / "tart_cargo"))

UV = local["uv"]
GIT = local["git"]


def _paint(style, text: str) -> str:
    """Apply a plumbum style portably.

    plumbum 2 dropped ``colors.bold("text")`` in favour of ``text | style``,
    and that form degrades to plain text when stdout is not a tty — exactly
    what we want in captured CI logs.
    """
    return text | style


def _bold(text: str) -> str:
    return _paint(colors.bold, text)


def _light_blue(text: str) -> str:
    return _paint(colors.light_blue, text)


def _green(text: str) -> str:
    return _paint(colors.green, text)


def _yellow(text: str) -> str:
    return _paint(colors.yellow, text)


_LEVELS = ("patch", "minor", "major")


# ---------------------------------------------------------------- helpers

def _run(cmd, **kwargs):
    """Run a plumbum command from the repo root, echoing it, failing on error."""
    print(_light_blue(f"+ {cmd}"), flush=True)
    kwargs.setdefault("cwd", str(ROOT))  # the chain may be invoked from anywhere
    return cmd.run(**kwargs)


def _capture(cmd):
    """Run a plumbum command quietly from the repo root -> (rc, out, err)."""
    kwargs = {"cwd": str(ROOT)}
    return cmd.run(retcode=None, **kwargs)


def _banner(text):
    print(_bold(f"\n=== {text}"), flush=True)


def _git(*args, cwd=ROOT):
    rc, out, err = local["git"][list(args)].run(retcode=None, cwd=str(cwd))
    if rc != 0:
        raise Exit(f"git {' '.join(args)} failed in {cwd}: {err.strip()}")
    return out.strip()


# ------------------------------------------------------------ version logic

_VERSION_RE = re.compile(r'^version = "(?P<v>[^"]+)"', re.M)


def read_version() -> str:
    match = _VERSION_RE.search(PYPROJECT.read_text())
    if not match:
        raise Exit(f"no version = \"...\" line in {PYPROJECT}")
    return match.group("v")


def write_version(version: str) -> None:
    text = PYPROJECT.read_text()
    PYPROJECT.write_text(_VERSION_RE.sub(f'version = "{version}"', text, count=1))


def bump_version(version: str, level: str) -> str:
    if level not in _LEVELS:
        raise Exit(f"--level must be one of {_LEVELS}, got {level!r}")
    major, minor, patch = (int(part) for part in version.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def last_tag() -> str:
    rc, out, _ = _capture(GIT["describe", "--tags", "--abbrev=0"])
    return out.strip() if rc == 0 else ""


def commits_since(tag: str) -> list[str]:
    """Subject lines since `tag`; without a tag, the ten most recent ones."""
    ref = [f"{tag}..HEAD"] if tag else ["-10"]
    rc, out, _ = _capture(GIT["log", *ref, "--pretty=%s"])
    return [line for line in out.splitlines() if line.strip()] if rc == 0 else []


# ------------------------------------------------------------- CHANGES.md

def _changes_for_release(version: str) -> None:
    """Turn `## Unreleased` into `## <version> (<today>)`, leaving a fresh
    empty Unreleased section behind. If there is no Unreleased section (the
    tart_cargo-style per-version layout), synthesise one from git history."""
    today = _dt.date.today().isoformat()
    text = CHANGES.read_text()

    if re.search(r"^## Unreleased\s*$", text, re.M):
        text = re.sub(
            r"^## Unreleased\s*$",
            f"## Unreleased\n\n## {version} ({today})",
            text,
            count=1,
            flags=re.M,
        )
    else:
        bullets = "\n".join(f"- {line}" for line in commits_since(last_tag()))
        section = f"## {version} ({today})\n\n{bullets or '- See git history.'}\n"
        head, _, rest = text.partition("\n")
        text = f"{head}\n\n{section}\n{rest.lstrip()}\n"

    CHANGES.write_text(text)


# ------------------------------------------------------------------ guard

def _assert_clean_for_publish() -> None:
    """Only the two files a release edits may be *modified*.

    Untracked files (local tooling, scratch output) are reported but do not
    block: they are not part of what the tag points at.
    """
    allowed = {PYPROJECT, CHANGES}
    unexpected, untracked = [], []
    for line in _git("status", "--porcelain").splitlines():
        if not line.strip():
            continue
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if line.startswith("??"):
            untracked.append(line)
        elif (ROOT / path) not in allowed:
            unexpected.append(line)
    if untracked:
        print(_yellow("  untracked, not part of the release:"))
        for line in untracked:
            print(_yellow(f"    {line}"))
    if unexpected:
        raise Exit(
            "working tree has unrelated changes; commit or stash them first:\n  "
            + "\n  ".join(unexpected)
        )


def _confirm(message: str, assume_yes: bool) -> None:
    if assume_yes:
        print(_yellow(f"{message} (--yes: proceeding)"))
        return
    answer = input(f"{message} [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise Exit("aborted")


# ------------------------------------------------------------------- tasks

@task(help={"quick": "single interpreter and no package build"})
def test(c, quick=False):
    """Exhaustive local tests — everything a release must pass first."""
    _banner("test: environment")
    _run(UV["sync", "--dev"])
    _run(UV["lock", "--check"])  # uv.lock agrees with pyproject.toml

    _banner("test: lint (fatal findings only)")
    _run(
        UV[
            "run", "flake8", "disko", "--count",
            "--select=E9,F63,F7,F82", "--show-source", "--statistics",
        ]
    )
    rc, out, _ = _capture(
        UV[
            "run", "flake8", "disko", "--count", "--exit-zero",
            "--max-complexity=10", "--max-line-length=127", "--statistics",
        ]
    )
    total = (out.strip().splitlines() or ["?"])[-1]
    print(_yellow(f"  style findings (informational, not fatal): {total}"))

    interpreters = PYTHON_MATRIX[:1] if quick else PYTHON_MATRIX
    for version in interpreters:
        _banner(f"test: pytest on python {version}")
        _run(UV["run", "--python", version, "pytest"])

    _banner("test: CLI smoke tests")
    tmp = Path(tempfile.mkdtemp(prefix="disko-release-"))
    try:
        _run(UV["run", "disko", "--version"])
        # README example: imaging + HDF + SVG + drawing
        _run(
            UV[
                "run", "disko", "--ms", str(TEST_MS), "--nside", "32",
                "--dir", str(tmp), "--HDF", "smoke.h5", "--SVG",
            ]
        )
        _run(
            UV[
                "run", "disko_draw", str(tmp / "smoke.h5"),
                "--SVG", str(tmp / "smoke.svg"),
            ]
        )
        # issue #12 defaults: no algorithm flag -> Tikhonov alpha=0.02,
        # no tiling flag -> healpix (the README's shortest command)
        _run(UV["run", "disko", "--ms", str(TEST_MS), "--nside", "4", "--dir", str(tmp), "--PDF"])
        # issue #12 opt-in unregularized path
        _run(
            UV[
                "run", "disko", "--ms", str(TEST_MS), "--nside", "4",
                "--dir", str(tmp), "--no-regularization", "--HDF", "smoke_nr.h5",
            ]
        )
        # the other two entry points still start
        _run(UV["run", "disko_svd", "--help"])
        _run(UV["run", "disko_bayes", "--help"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not quick:
        _banner("test: package build")
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        _run(UV["build"])

    print(_green("\ntest: OK"))


@task(
    help={
        "level": "patch | minor | major (default: patch)",
        "dry_run": "show what would change without writing anything",
    }
)
def bump(c, level="patch", dry_run=False):
    """Bump the pyproject version and open a CHANGES.md section for it."""
    _banner("bump")
    current = read_version()
    new = bump_version(current, level)
    tag = last_tag()

    print(f"  version {current} -> {new} ({level})")
    print(f"  last tag: {tag or '(none)'}")
    for line in commits_since(tag)[:15]:
        print(f"    - {line}")

    if dry_run:
        print(_yellow("  dry run: no files written"))
        return

    write_version(new)
    _changes_for_release(new)
    print(_green(f"  bumped to {new}; see {CHANGES.name}"))


@task(
    help={
        "yes": "skip the confirmation prompt",
        "wait": "wait for the PyPI publish workflow to go green (default)",
    }
)
def publish(c, yes=False, wait=True):
    """Commit the bump, tag vX.Y.Z and push: CI publishes to PyPI.

    .github/workflows/package.yaml triggers on the v*.*.* tag and runs
    `uv build` + pypa/gh-action-pypi-publish, so this never runs twine
    locally. Waiting matters: the tools image built in the next stage
    installs disko from PyPI and would otherwise race the publish.
    """
    _banner("publish")
    version = read_version()
    tag = f"v{version}"

    if tag in _git("tag").splitlines():
        raise Exit(f"{tag} already exists — run `invoke bump` first")
    _assert_clean_for_publish()

    _confirm(f"Release {version}: commit, tag {tag} and push (CI publishes to PyPI)?", yes)

    _run(GIT["add", str(PYPROJECT), str(CHANGES)])
    if _capture(GIT["diff", "--cached", "--quiet"])[0] != 0:
        _run(GIT["commit", "-m", f"Release {version}"])
    else:
        print("  bump already committed; tagging the current commit")
    _run(GIT["tag", "-a", tag, "-m", f"Release {version}"])
    _run(GIT["push", "--follow-tags", "origin", "HEAD"])

    if wait:
        _wait_for_publish(tag)
    else:
        print(_yellow("  not waiting for CI; verify the publish run yourself"))


def _wait_for_publish(tag: str, timeout_s: int = 1800, poll_s: int = 20) -> None:
    # plumbum's local.which() raises when a program is missing and hands back
    # a path (not a command) when it exists, so detect with shutil.which and
    # build the command with local[...].
    if shutil.which("gh") is None:
        raise Exit("gh CLI is required to wait for the PyPI publish (or pass --no-wait)")
    gh = local["gh"]

    print(_light_blue(f"  waiting for the package.yaml run on {tag} ..."))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc, out, _ = _capture(
            gh[
                "run", "list", "--repo", "tmolteno/disko", "--workflow", "package.yaml",
                "--limit", "10", "--json", "headBranch,status,conclusion,url",
            ]
        )
        if rc == 0 and out.strip() and out.strip() != "[]":
            for run in json.loads(out):
                if run.get("headBranch") == tag and run.get("status") == "completed":
                    if run.get("conclusion") == "success":
                        print(_green(f"  PyPI publish OK: {run['url']}"))
                        return
                    raise Exit(f"publish workflow failed: {run['url']}")
        time.sleep(poll_s)
    raise Exit(f"timed out after {timeout_s}s waiting for the publish run on {tag}")


def _chain(repo: Path, args: list[str], label: str) -> None:
    if not (repo / "tasks.py").exists():
        raise Exit(f"{repo} has no tasks.py — cannot run `invoke release` for {label}")
    _banner(f"chain: {label} ({repo})")
    _run(local["invoke"]["-r", str(repo), "release", *args])


@task(
    help={
        "level": "patch | minor | major for every stage (default: patch)",
        "yes": "skip every confirmation prompt",
        "wait": "wait for the PyPI publish workflow (default)",
        "dry_run": "print the full plan and exit without running anything",
    }
)
def release(c, level="patch", yes=False, wait=True, dry_run=False):
    """Full release: test -> bump -> CI publish -> tools -> tart_cargo."""
    version = read_version()
    planned = bump_version(version, level)

    if dry_run:
        _banner("release plan (dry run)")
        print(f"  1. test        exhaustive: lock check, lint gate, pytest on {', '.join(PYTHON_MATRIX)}, CLI smoke, uv build")
        print(f"  2. bump        {version} -> {planned} (+ CHANGES.md section)")
        print(f"  3. publish     commit + tag v{planned} + push -> package.yaml -> PyPI, wait for green")
        print(f"  4. tools       invoke -r {TOOLS_REPO} release --disko-version={planned}")
        print(f"  5. tart_cargo  invoke -r {CARGO_REPO} release --tools-tag=<new tools tag>")
        print(_yellow("  dry run: nothing executed"))
        return

    _confirm(
        f"Release disko {version} -> {planned} and chain tools + tart_cargo releases?",
        yes,
    )
    yes = True  # confirmed once; downstream stages inherit the consent

    test(c, quick=False)
    bump(c, level=level, dry_run=False)
    publish(c, yes=yes, wait=wait)

    # tools installs disko from PyPI, so its stage runs only after the
    # publish above went green (unless --no-wait was requested).
    _chain(TOOLS_REPO, [f"--level={level}", "--yes", f"--disko-version={planned}"], "tools")
    tools_tag = _git("describe", "--tags", "--abbrev=0", cwd=TOOLS_REPO)
    _chain(CARGO_REPO, [f"--level={level}", "--yes", f"--tools-tag={tools_tag}"], "tart_cargo")

    print(_green(f"\nrelease {planned} complete: disko {planned}, tools {tools_tag}, tart_cargo released"))


@task
def status(c):
    """Show where every repo in the chain currently stands."""
    _banner("status")
    print(f"  disko      version={read_version()}  last tag={last_tag() or '(none)'}")
    for label, repo in (("tools", TOOLS_REPO), ("tart_cargo", CARGO_REPO)):
        if not repo.exists():
            print(f"  {label:<10} missing: {repo}")
            continue
        tag = _git("describe", "--tags", "--abbrev=0", cwd=repo)
        # tracked modifications only: untracked scratch (.qwen, logs) is normal
        modified = [
            line
            for line in _git("status", "--porcelain", cwd=repo).splitlines()
            if line.strip() and not line.startswith("??")
        ]
        tasks = "tasks.py" if (repo / "tasks.py").exists() else "NO tasks.py"
        print(f"  {label:<10} last tag={tag}  modified={len(modified)}  {tasks}  ({repo})")
