# Security Policy

## Supported versions

Jitterbug is a research tool maintained on a best-effort basis. Only the latest
release line receives fixes.

| Version | Supported |
| ------- | --------- |
| 2.x     | ✅        |
| 1.x     | ❌        |

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private reporting instead:
[Report a vulnerability](https://github.com/estcarisimo/jitterbug/security/advisories/new).

Include a description of the issue, reproduction steps (a minimal input file and the
command or code that triggers it), and the version and platform you saw it on. Expect
an acknowledgement within about two weeks. This is an academic side project, not a
staffed product, so response times are best-effort.

Confirmed vulnerabilities are fixed in a pull request, released, and then disclosed in
`CHANGELOG.md` under *Security* and in a GitHub security advisory that credits the
reporter (unless they prefer otherwise). Reporters are asked to keep details private
until the fix is released.

## What this tool does with your data

Jitterbug reads Round-Trip Time (RTT) measurements from files you provide (CSV, scamper
JSON) or from an InfluxDB instance you configure, runs change point detection and
statistical tests on them, and writes results to files or the terminal. It does not
require credentials for the file-based workflow, does not send telemetry, and makes no
network requests of its own except:

- **InfluxDB**: only when you use the InfluxDB loader, and only to the server and with
  the token you configure.
- **Installing the optional `bcp` back end** fetches a package from GitHub at install
  time, as with any Python dependency.

## Scope

In scope:

- Path traversal or unintended file writes through the output options of the CLI or the
  exporters.
- Unsafe parsing or deserialisation of input files (CSV, JSON, YAML configuration).
- Leakage of InfluxDB credentials through logs, configuration dumps, or results files.
- Denial of service through crafted input that is disproportionate to its size (for
  example, quadratic blow-ups in the change point detectors on adversarial input).
- Dependency vulnerabilities with a plausible exploitation path in this tool.

Out of scope:

- Vulnerabilities in the third-party detectors (ruptures, bayesian_changepoint_detection);
  report those upstream.
- Findings from automated scanners with no demonstrated impact.

## Automated scanning

CI runs `bandit` over the package and `pip-audit` over the locked dependency set on
every pull request, and GitHub CodeQL scans the code base. Dependency updates are handled by Dependabot. Secret
scanning with push protection is enabled on the repository.
