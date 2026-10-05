# CHANGELOG

<!-- version list -->

## v0.2.0 (2026-10-05)

### Bug Fixes

- **mutator**: Spread mutants across rules and operators
  ([`91433b3`](https://github.com/burakkaygusuz/clastogen/commit/91433b35742f23984375ee003978e5f40eab325c))

- **plugin**: Preserve pytest exit status and skip xfailed trials
  ([`1dbea5e`](https://github.com/burakkaygusuz/clastogen/commit/1dbea5e29026bdb0bd618418a5983623f215d277))

- **stats**: Reject min_rate of 1.0 in evaluate_pass_rate
  ([`2890f5d`](https://github.com/burakkaygusuz/clastogen/commit/2890f5db64f8253bdd2419faaafe4cf70b3ae425))

### Build System

- **sdist**: Limit source distribution to package files
  ([`2f15372`](https://github.com/burakkaygusuz/clastogen/commit/2f1537221aa7fb55fe645b6d596fd7aa37f5771d))

### Continuous Integration

- **actions**: Pin actions to commit SHAs and add dependabot
  ([`5908f64`](https://github.com/burakkaygusuz/clastogen/commit/5908f64a85e2b9c7a023f6ead09f1195029e9591))

- **release**: Push releases with a GitHub App token
  ([`e0bb90e`](https://github.com/burakkaygusuz/clastogen/commit/e0bb90ecb63c23d39b7cba10aa06de63339eda97))

### Documentation

- **readme**: Correct SPRT call counts and document baseline noise
  ([`77086fa`](https://github.com/burakkaygusuz/clastogen/commit/77086fae635c3f9ff156f2ec97ab13414fee1276))

### Refactoring

- **core**: Collapse single-module packages into modules
  ([`d5c0f49`](https://github.com/burakkaygusuz/clastogen/commit/d5c0f492daf384d20bf81e3dafd0b5c5c2ceed84))

- **plugin**: Read suppressions from TOML with stdlib tomllib
  ([`3402808`](https://github.com/burakkaygusuz/clastogen/commit/3402808b395f071ca71ad5aa6573e65801491927))

### Breaking Changes

- **plugin**: Suppressions move from .clastogen/suppressions.yaml to .clastogen/suppressions.toml
  using [[suppressions]] tables, and the clastogen[config] extra is removed.


## v0.1.0 (2026-10-05)

- Initial Release
