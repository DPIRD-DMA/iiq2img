# Releasing iiq2img

`iiq2img` is published to [PyPI](https://pypi.org/project/iiq2img/) automatically
by GitHub Actions whenever a `v*` tag is pushed. Authentication uses PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), and the
`publish` job waits for a reviewer to approve the `pypi` environment before
uploading. The workflow lives at
[`.github/workflows/publish.yml`](../.github/workflows/publish.yml).

The package version is **derived from the git tag** by
[`setuptools-scm`](https://setuptools-scm.readthedocs.io/) at build time, so
there is no version string to bump in [`pyproject.toml`](../pyproject.toml).
The tag is the source of truth.

## Picking the right version

The tag is the *only* place the version lives, so a typo in `git tag` becomes
the published version. A few habits keep this safe:

- **Check the previous tag** before picking a new one:
  ```sh
  git describe --tags --abbrev=0    # most recent tag
  git tag --sort=-v:refname | head  # last few tags, newest first
  ```
- **Preview what `setuptools-scm` would build right now** (useful when sanity
  checking a tag locally before pushing):
  ```sh
  uvx --from setuptools-scm python -m setuptools_scm
  ```
  After tagging locally but before pushing, this should print the exact
  version you expect (e.g. `0.5.1`). If it prints something like
  `0.5.1.dev3+g1a2b3c4`, your working tree has uncommitted changes or the tag
  isn't on `HEAD` — fix that before pushing.
- **Prefer the GitHub *Releases* UI** over the terminal for cutting tags
  (*Releases → Draft a new release → Choose a tag → Create new tag on
  publish*). The dropdown shows every existing tag right above the input box,
  which makes off-by-one mistakes obvious. It also lets you write release
  notes in the same step.
- **Use the approval gate as a safety net.** The `build` job runs *before*
  the `publish` job, and its logs show the exact version `setuptools-scm`
  derived (e.g. `Successfully built iiq2img-0.5.1.tar.gz`). Always glance at
  that line before clicking *Approve* in the `pypi` environment — it's your
  last chance to catch a typo before anything reaches PyPI.

If you do push a wrong tag and notice before approving the deployment:
**reject** the deployment in the Actions UI, then delete the bad tag locally
and on the remote (see *Troubleshooting* below). Nothing reaches PyPI until
you approve.

## Cutting a release

1. **Update [`CHANGELOG.md`](../CHANGELOG.md)** with the new version and a
   summary of changes, then commit and push to `main`:
   ```sh
   git add CHANGELOG.md
   git commit -m "changelog for v0.5.1"
   git push
   ```
2. **Tag and push** — this is what triggers the release. The tag (minus the
   leading `v`) becomes the PyPI version, so follow
   [semver](https://semver.org/): patch for fixes, minor for additive features,
   major for breaking changes.
   ```sh
   git tag v0.5.1
   git push origin v0.5.1
   ```
3. **Approve the deployment.** Open the run in
   [Actions](https://github.com/DPIRD-DMA/iiq2img/actions) — the `build` job
   runs first, then the `publish` job waits in the `pypi` environment for a
   reviewer to click *Review deployments → Approve and deploy*.
4. **Verify.** Once the workflow goes green:
   ```sh
   pip install --upgrade iiq2img
   iiq2img --help
   ```
   and check the new version on
   [pypi.org/project/iiq2img](https://pypi.org/project/iiq2img/).

## Troubleshooting

- **OIDC / "trusted publisher" error on upload.** The pending publisher on
  PyPI is missing or its fields don't match the workflow exactly (owner, repo,
  workflow filename, environment name are all case-sensitive).
- **`File already exists` from PyPI.** You pushed a tag whose version is
  already published. PyPI does not allow re-uploading — delete the bad tag and
  push a new one with a higher version:
  ```sh
  git tag -d v0.5.1
  git push origin :refs/tags/v0.5.1
  git tag v0.5.2
  git push origin v0.5.2
  ```
- **Version looks like `0.5.2.dev3+g1a2b3c4`.** `setuptools-scm` saw commits
  *after* the latest tag and produced a development version. This only happens
  if the workflow built from a non-tag ref, or if `fetch-depth: 0` /
  `fetch-tags: true` are missing from the checkout step in
  [`publish.yml`](../.github/workflows/publish.yml).
- **Workflow stuck on "Waiting for review".** That's the `pypi` environment
  protection rule — a required reviewer needs to approve the run in the
  Actions UI.

## Yanking a bad release

Releases can't be deleted from PyPI, but they can be
[yanked](https://pypi.org/help/#yanked) from the project page so `pip` won't
install them by default. Then cut a new patch release with the fix.
