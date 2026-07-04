# Security Policy

## Supported versions

`pngbend` is pre-1.0 software. Only the latest release and the `main` branch
receive security fixes.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's [private vulnerability reporting][report] for this repository, or
contact the maintainer through the address on the [GitHub profile][profile].

[report]: https://github.com/nsheely/pngbend/security/advisories/new
[profile]: https://github.com/nsheely

Include a description, the affected version or commit, and a reproducer. A
crafted `.png` that triggers the issue is ideal. Expect an acknowledgement
within a few days.

## Scope

`glasspng` decodes untrusted binary input (arbitrary PNG and DEFLATE data), so
the highest-value reports are memory-safety or denial-of-service issues in the
decode path reachable from `glasspng::decode` on a crafted file: panics,
out-of-bounds access, or unbounded CPU/memory use. The decoder caps inflated
output at the IHDR-implied size to defend against decompression bombs; a way
past that cap is in scope.
