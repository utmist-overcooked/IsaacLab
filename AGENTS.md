# IsaacLab Guidelines

## Breaking API changes

- **Breaking changes require a deprecation first.** Do not remove or rename public API symbols without deprecating them in a prior release.

## API design rules (naming + structure)

- **Group by common prefix for discoverability (autocomplete).**
  - **Classes**: group by domain concept — `ActuatorNetLSTM`, `ActuatorNetMLP` (not `LSTMActuatorNet`, `MLPActuatorNet`).
  - **Methods**: group by noun before modifier — `set_joint_position_target()` (not `set_target_joint_position()`).
- **Method names are `snake_case`.**
- **CLI arguments are `snake_case`.**
- **Prefer nested classes when self-contained.**
  - If a helper type or an enum is only meaningful inside one parent class and doesn't need a public identity, define it as a nested class instead of creating a new top-level class/module.
- **Follow PEP 8 for Python code.**
- **Use modern Python type-hint syntax.**
  - Prefer PEP 604 unions: `x | y`, `x | None`. Do not use `typing.Union` or `typing.Optional`.
- **Use specific type hints for public interfaces.**
  - For torch tensors, annotate with `torch.Tensor`. For Warp arrays, annotate concrete dtypes (e.g., `wp.array(dtype=wp.vec3)`) rather than generic `object`.
  - Prefer consistent parameter names across base/override APIs (e.g., `xforms`, `scales`, `colors`, `materials`).
- **Use Google-style docstrings.**
  - Write clear, concise docstrings that explain what the function does, its parameters, and its return value.
  - Keep argument/return types in function annotations, not inline in docstrings.
  - In `Args:` entries, use `name: description` (not `name (Type): description`).
  - Use Sphinx cross-reference roles for symbol references (e.g. `:class:`, `:meth:`, `:attr:`, `:paramref:`), but keep targets as short as possible.
  - Within the same class/module, prefer short local references (e.g. `:meth:\`set_joint_position_target\``, `:attr:\`num_joints\``) over fully qualified paths.
  - If qualification is needed, prefer public API paths (e.g. `isaaclab.assets.Articulation`) and do not use internal `_src` or private module paths in Sphinx role targets.
- **State SI units for all physical quantities in docstrings.**
  - Use inline `[unit]` notation, e.g. `"""Particle positions [m], shape [particle_count, 3], float."""`.
  - For joint-type-dependent quantities use `[m or rad, depending on joint type]`.
  - For spatial vectors annotate both components, e.g. `[N, N·m]`.
  - For compound arrays list per-component units, e.g. `[0] k_mu [Pa], [1] k_lambda [Pa], ...`.
  - When a parameter's interpretation varies across solvers, document each solver's convention instead of a single unit.
  - Skip non-physical fields (indices, keys, counts, flags).
  - This rule applies to **public API docstrings only**, not test docstrings.
- **Keep the documentation up-to-date.**
  - When adding new files or symbols that are part of the public-facing API, make sure to keep the auto-generated documentation updated by running `./isaaclab.sh -d`.

## Dependencies

- **Avoid adding new required dependencies.** IsaacLab's core should remain lightweight and minimize external requirements.
- **Strongly prefer not adding new optional dependencies.** If additional functionality requires a new package, carefully consider whether the benefit justifies the added complexity and maintenance burden. When possible, implement functionality using existing dependencies, including Warp functions and kernels, NumPy, or the standard library.

## Tooling: prefer `./isaaclab.sh -p` for running, testing, and benchmarking

> **Vulcan note:** every command in this section (and `./isaaclab.sh -d` above) must run
> **inside a Slurm job** (`salloc` for iterating, `sbatch` for batch runs), never on the
> login node — most of it boots Isaac Sim and needs a GPU. Set up the environment first
> with the preamble in the "LOCAL: Running on Vulcan" section at the end of this file.

We use a wrapped python call within `./isaaclab.sh`.

- **Use `./isaaclab.sh -p -c` for inline Python**: When running one-off Python commands, use `./isaaclab.sh -p -c "..."` instead of `python3 -c "..."`.
- **Use `./isaaclab.sh -p`** to run standalone Python scripts without a `pyproject.toml` (e.g., in CI after switching to a branch with no project files).

### Run tests

```bash
# run all tests (extremely heavy, should be avoided).
./isaaclab.sh -t

# run a specific test file by name
./isaaclab.sh -p -m pytest PATH_TO_TEST

# run a specific example test
./isaaclab.sh -p -m pytest PATH_TO_TEST::METHOD
```

### Pre-commit (lint/format hooks)

**CRITICAL: Always run pre-commit hooks BEFORE committing and BEFORE pushing.**

> **Vulcan note:** run `unset PYTHONPATH PIP_CONFIG_FILE` first — pre-commit pip-installs
> its hook environments on first use, and Alliance's manylinux shim breaks those installs
> otherwise. Linting itself is light enough for the login node (it needs internet access,
> which compute nodes only have through the proxy).

Proper workflow:
1. Make your code changes
2. Run `./isaaclab.sh -f` to check ALL files
3. If pre-commit modifies any files (e.g., formatting), review the changes
4. Stage the modified files with `git add`
5. Run `./isaaclab.sh -f` again to ensure all checks pass
6. Only then create your commit with `git commit`
7. Verify pre-commit still passes before pushing — never push commits that haven't been checked

```bash
# Run pre-commit checks on all files
./isaaclab.sh -f
```

**Common mistakes to avoid:**
- Don't commit first and then run pre-commit (requires amending commits)
- Don't push before running pre-commit (pushes broken code to the remote)
- Do run pre-commit before committing and before pushing (clean workflow)

**When reviewing code** (e.g. via a code-reviewer agent), always run `./isaaclab.sh -f` as part of the review to catch formatting or lint issues early.

## Changelog

- **Do not edit `CHANGELOG.rst` or `config/extension.toml` directly.** Each PR adds a fragment file under `source/<package>/changelog.d/`; the changelog and version are compiled by the nightly CI workflow.
- **Add one fragment per touched package.** Pick any short, unique slug for the filename — your branch name (with `/` replaced by `-`) is a good default. The filename suffix declares the bump tier; within a batch the highest tier wins for the package.

  | Filename | Effect |
  |---|---|
  | `source/<pkg>/changelog.d/<slug>.rst` | patch bump |
  | `source/<pkg>/changelog.d/<slug>.minor.rst` | minor bump |
  | `source/<pkg>/changelog.d/<slug>.major.rst` | major bump |
  | `source/<pkg>/changelog.d/<slug>.skip` | no entry, no bump (CI / docs / test-only) |

- Use **past tense** matching the section header: "Added X", "Fixed Y", "Changed Z".
- Place entries under the correct category: `Added`, `Changed`, `Deprecated`, `Removed`, or `Fixed`.
- Avoid internal implementation details users wouldn't understand.
- **For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance.**
  - Example: "Deprecated `Articulation.A` in favor of `Articulation.B`."
- **Breaking changes** belong in `Changed`, prefixed with `**Breaking:**`.
- Use Sphinx cross-reference roles for class/method/module names.

### RST formatting reference

```
Added
^^^^^

* Added :class:`~package.ClassName` to support feature X.

Fixed
^^^^^

* Fixed edge case in :meth:`~package.ClassName.method` where input was
  not validated, causing ``AttributeError`` at runtime.
```

Key formatting rules:
- Category heading: underline with `^` (carets), at least as long as the heading text.
- Entries: `* ` prefix, continuation lines indented by 2 spaces.

See `tools/changelog/test/integration/` for worked examples that double as integration-test fixtures.

## Commit and Pull Request Guidelines

Follow conventional commit message practices.

- **Use feature branches**: All development work should be on branches named `<username>/feature-desc` (e.g., `jdoe/docs-versioning`). Do not commit directly to `main`.
- Keep commits focused and atomic—one logical change per commit.
- Reference related issues in commit messages when applicable.
- **When iterating on PR feedback**, prefer adding new commits over amending existing ones. This avoids force-pushing and lets the reviewer easily verify each change request was addressed.
- **Do not include AI attribution or co-authorship lines** (e.g., "Co-Authored-By: Claude...") in commit messages. Commits should represent human contributions without explicit AI attribution.
- **Commit message format**:
  - Separate subject from body with a blank line
  - Subject: imperative mood, capitalized, ~50 chars, no trailing period
    - Write as a command: "Fix bug" not "Fixed bug" or "Fixes bug"
    - Test: "If applied, this commit will _[your subject]_"
  - Body: wrap at 72 chars, explain _what_ and _why_ (not _how_—the diff shows that)

## File headers and copyright

- New files must use the current year (2026) in the SPDX copyright header:
  ```
  # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```
- Do not change the year in existing file headers.

## Sandbox & Networking

- **Never push to `origin` (`isaac-sim/IsaacLab`).** The `origin` remote is the public upstream repository; this is a personal local copy and is not contributed back upstream.
- (Vulcan: login nodes have direct internet access; compute nodes reach the outside only through the prolog-injected squid proxy.)

## GitHub Actions and CI/CD

- Pin actions by major version tag (e.g. `actions/checkout@v6`). Use the same major version that other workflows in `.github/workflows/` already use — don't introduce a new major version without checking how it's used elsewhere.

## Testing Guidelines

- **Always verify regression tests fail without the fix.** When writing a regression test for a bug fix, temporarily revert the fix and run the test to confirm it fails. Then reapply the fix and verify the test passes. This ensures the test actually covers the bug.

### Debugging Warp kernels

**Do not add `wp.printf` to kernels in production code.** Debug prints in Warp kernels affect performance and can produce noisy test output. Use them only in standalone reproduction scripts during development, and always remove them before committing.

To debug Warp kernel behavior:

1. **Write a standalone reproduction script** and run it directly with `./isaaclab.sh -p -c "..."` or `./isaaclab.sh -p script.py`. This keeps stdout visible and avoids the test framework entirely.
2. **Use high-precision format strings** for floating-point debugging (e.g., `wp.printf("val=%.15e\n", x)`) — the default `%f` format hides values smaller than ~1e-6 that can still affect control flow.
3. **Remove all `wp.printf` calls before committing.**

---

# LOCAL: Running on Vulcan (Alliance HPC)

This checkout lives on the Vulcan cluster. Everything below is local operational
knowledge; it overrides upstream guidance where they conflict.

## Environment preamble (required for EVERY use of this install)

Any shell or job script that runs `./isaaclab.sh` (including `-p`, `-p -c`, tests,
training) needs, in this order:

```bash
module --force purge && module load StdEnv/2023 python/3.12.4
unset PYTHONPATH PIP_CONFIG_FILE   # (1)
cd /project/aip-mtaylor3/zachtang/IsaacLab
source env_isaaclab/bin/activate   # venv lives in-repo; isaaclab.sh auto-detects it
export LD_LIBRARY_PATH=$PWD/.cuda-shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # (2)
export OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=$SCRATCH/.cache HF_HOME=$SCRATCH/hf
mkdir -p $SCRATCH/ovcache/{cache_ov,share_ov,nvidia,nv,omnilogs}   # (3)
```

1. Alliance injects a `_manylinux` shim via `PYTHONPATH` that makes pip reject all
   manylinux wheels, and `PIP_CONFIG_FILE` adds wheelhouse/constraints. This venv is a
   vanilla-PyPI environment (isaacsim/torch-cu128 only exist as manylinux wheels);
   always unset both before activating or pip-installing into it.
2. omni.physx dlopens unversioned `libcuda.so`, which the CVMFS Gentoo loader cannot
   resolve. Without the `.cuda-shim` dir on `LD_LIBRARY_PATH`, PhysX silently falls
   back to CPU simulation ("No CUDA context manager available" in the Kit log) and GPU
   pipelines crash with Warp `ProxyArray` errors — torch/warp still work, so it is
   easy to misdiagnose.
3. `~/.cache/ov`, `~/.cache/nvidia`, `~/.local/share/ov`, `~/.nv`,
   `~/.nvidia-omniverse` are symlinks into `$SCRATCH/ovcache/` so Kit caches cannot
   fill the 50 GB `$HOME`; recreate the targets in each job in case scratch was purged.

## Interactive GUI / editor (viewport streaming)

If the user asks about **running the Isaac Sim GUI, the editor, or visualizing/interacting
with a scene** on Vulcan (as opposed to headless training), see **`RUN.md`** in the repo
root. It has the copy-paste runbook: OnDemand Interactive Desktop (GPU) → `LIVESTREAM=2`
headless render → NVIDIA WebRTC client to `127.0.0.1`, plus the VirtualGL/black-screen
gotchas. Helper scripts: `/scratch/zachtang/isaaclab_setup/gui_env.sh` and
`run_webrtc_client.sh`. (`RUN.md` is local-only and git-excluded — do not upstream.)

## Scheduling jobs

- **At the start of any session where the agent anticipates needing the GPU, ask the
  user first** — before submitting anything. Do not silently default to a pattern.
  The prompt should cover:
  - **`sbatch` vs `salloc`**: does the user want each task submitted as its own batch
    job, or one interactive `salloc` session reserved for the whole work session?
  - **How long** to reserve (a `--time` value), if `salloc` is chosen.
  - **Context on the anticipated GPU usage plan** — what the agent expects to run and
    roughly how often (e.g. "one training run, ~2h, run once" vs "iterating on env
    code, expect ~15 short smoke tests over the next hour"). This is for the user's
    benefit: it tells them how often the GPU is expected to be touched so they can
    judge whether a reserved session or per-task batch jobs makes more sense — see the
    fairshare/overhead tradeoff below.
  - Re-ask if the plan changes significantly mid-session (e.g. scope grows from one
    quick check to many iterations).
- **Never run Python/`./isaaclab.sh` on the login node** — everything goes through
  Slurm (`sbatch`/`salloc`). This overrides upstream's "run `./isaaclab.sh -p -c`
  inline" guidance: the command is right, but it must execute inside a job.
- GPU jobs: `--account=aip-mtaylor3 --gres=gpu:l40s:1 --cpus-per-task=8 --mem=32G`
  and the shortest `--time` that fits (≤3h reaches the most nodes). Never set
  `--partition`; never use bare `gpu:N`. Job output goes to `$SCRATCH/logs/`.
- Working templates (install, GPU smoke test with utilization sampling):
  `/scratch/zachtang/isaaclab_setup/`.
- Cloud assets stream from
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`;
  compute nodes reach it (and PyPI) through the prolog-injected squid proxy.

## Install facts (2026-09-01)

- Branch `release/3.0.0-beta2` (repo default), venv `env_isaaclab/` (git-excluded via
  `.git/info/exclude`), `isaacsim[all,extscache]==6.0.1.0`, torch 2.10.0+cu128,
  installed with `./isaaclab.sh -i 'newton,rl,visualizer'`.
- `mimic`/`teleop` were NOT installed (robomimic needs cmake compilation); add later
  with `./isaaclab.sh -i mimic` inside a job with the preamble above.
- Verified: rsl_rl PPO on `Isaac-Cartpole-v0` (PhysX GPU, headless, 4096 envs) trains
  at ~200k steps/s on one L40S.
