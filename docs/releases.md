# CI and crates.io releases

## Regular checks

The CI workflow runs these independent jobs in parallel:

| Job | Platforms | Checks |
|---|---|---|
| Format and lint | Linux | rustfmt, Clippy with/without bundled data, release helper tests |
| Bundled tests | Linux + macOS | Unit, inference and documentation tests, without model downloads |
| External-only tests | Linux + macOS | Build and tests with default features disabled |
| Model and Python parity | Linux + macOS | Original model, every weight bit, Unicode audit, Python references |
| Crate packaging | Linux + macOS | Workspace publish dry-run, archive sizes and offline consumer |

Rust dependencies and build artifacts are cached by toolchain, OS/architecture,
lockfile and job configuration. Model and Unicode reference caches are keyed
by their pinned sources and generation scripts. Python dependencies use pip's
cache; generated Python references additionally include interpreter version,
platform, requirements, generator scripts, model manifest and fixture inputs
in their key. Cache hits skip downloads or reference generation, while model
validation, Unicode table checks, output comparisons and Rust tests still run.

New commits cancel obsolete branch checks. Publishing jobs are serialized and
are not automatically canceled. Multithread throughput benchmarks remain manual.

## One-time authentication

For the first release, add a crates.io API token as the repository Actions
secret **`CARGO_REGISTRY_TOKEN`**, with permission to publish new crates and
versions for `ragisa` and `ragisa-model`. Keep the token in GitHub's secret
settings; no token belongs in the repository or release tag.

After both crates exist, you can use
[crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing):
configure both crates with repository owner **Kilerd**, repository **ragisa**,
and workflow filename **release.yml**, without an environment name. Then remove
the repository token secret. The workflow uses OIDC when that secret is absent,
and obtains a short-lived token only after all checks pass. Initial publication
still requires an API token under crates.io's current rules.

## Prepare and verify a version

1. Set `package.version` in the root `Cargo.toml` to the intended ragisa version.
2. If the model crate changes, bump `model/Cargo.toml` and update the root
   dependency's exact `=version` requirement. An unchanged model crate can
   retain its existing published version.
3. Update `Cargo.lock` and commit the release changes to `main`.
4. Run the **Release** workflow manually on `main`, or use:

```sh
gh workflow run release.yml --ref main
```

Manual runs validate manifest consistency and reuse all CI jobs, including
`cargo publish --workspace --dry-run --locked` and the offline consumer check.
**Manual dispatch never uploads to crates.io**, even when dispatched on a tag.
It requires no publishing credentials.

Local preflight commands are also available:

```sh
python3 .github/scripts/release.py check --tag v0.1.0
cargo publish --workspace --dry-run --locked
python3 tools/check_package.py
```

Use CPython 3.12 or newer for the release helper. The tag must be exactly
`v` followed by the root package version; prerelease tags such as
`v0.2.0-rc.1` are supported.

## Publish from a tag

After configuring authentication and verifying the release commit, push its
version tag, for example:

```sh
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

The tag-triggered workflow validates versions and runs the full CI suite on
that commit, then publishes **ragisa-model first, followed by ragisa**. Cargo
waits for the model to appear in the registry index before proceeding. A
failure publishing the model prevents publishing the parent.

If a run fails after one crate has been published, rerun the failed workflow
jobs on the same tag. Already-published versions are skipped. Yanked target
versions and registry errors fail the release; they are not treated as missing
versions. Crates.io versions are immutable, so use a new version for changed
package contents rather than moving an existing release tag.
