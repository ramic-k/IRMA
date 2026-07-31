# Security policy

IRMA reads user-supplied files that can carry executable content: a
phonopy.yaml is YAML, machine-learned-potential checkpoints are
pickle-based archives, and `irma mlip env create` installs packages from
PyPI. IRMA scans phonopy.yaml inputs for the known code-execution vector
at every entry point that parses one, and pins known checkpoint hashes
where the backend allows it. These are mitigations, not a sandbox: treat
decks, phonopy.yaml files, bundles, and checkpoints from untrusted
sources with the same caution as any downloaded code.

To report a vulnerability, please use GitHub's private vulnerability
reporting (the Security tab, "Report a vulnerability") rather than a
public issue, so a fix can land before the details are public. Reports
are acknowledged as quickly as the author can manage, normally within a
week.
