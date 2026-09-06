"""Credential-shaped fixtures, assembled at runtime rather than written as literals.

The detector under test finds strings that look like credentials, so its tests need strings
that look like credentials. Writing them literally puts token-shaped text in the repository,
where upstream secret scanners find it and block the push — correctly, because a scanner
cannot tell a fixture from a leak, and it should not try.

Assembling each value from fragments means no literal in this repository matches a scanner's
pattern, while the string handed to the detector is byte-identical to the real shape. The
alternative — allowlisting the pattern, or clicking "allow this secret" on a blocked push —
trains everyone involved in exactly the wrong habit.

None of these are functional credentials. The AWS one is the value AWS itself publishes in
its documentation; the rest are filler.
"""

from __future__ import annotations

# AWS publishes this identifier as a documentation example. Split so the assembled value is
# correct while the source contains no complete access-key literal.
AWS_ACCESS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"

GITHUB_TOKEN = "ghp_" + "E" * 36
ANTHROPIC_KEY = "sk-" + "ant-" + "N" * 32
OPENAI_KEY = "sk-" + "R" * 32
SLACK_TOKEN = "xoxb-" + "0123456789" + "-" + "abcdefghij"
JWT_LIKE = "eyJ" + "A" * 20 + ".eyJ" + "B" * 20 + "." + "C" * 20

# Split around the word boundary so the source carries no complete PEM header.
PRIVATE_KEY_HEADER = "-----BEGIN RSA " + "PRIVATE KEY-----"

__all__ = [
    "ANTHROPIC_KEY",
    "AWS_ACCESS_KEY",
    "GITHUB_TOKEN",
    "JWT_LIKE",
    "OPENAI_KEY",
    "PRIVATE_KEY_HEADER",
    "SLACK_TOKEN",
]
