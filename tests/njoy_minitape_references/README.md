# NJOY mini reference tapes

Tiny NJOY2016.78 LEAPR output tapes used by `tests/test_writer_flag_tapes.py`
to pin the Card 4 output options byte-for-byte:

* `isabt.endf.gz` — isabt=1: the asymmetric S-tilde form (LASYM raised by 2).
* `ilog.endf.gz`  — ilog=1: ln(S) storage (ENDF LLN=1).

The generating decks live inline in the test file (a 3-alpha x 4-beta,
2-temperature toy spectrum — the same deck as the writer round-trip tests,
with the respective Card 4 flag set). The references were produced by running
unmodified NJOY2016.78 on those decks; IRMA's MF7 output is required to be
byte-identical to them (MF1 is excluded from the comparison: IRMA
deliberately writes a consistent NWD and exact directory record counts where
NJOY does not; see `_patch_mf1_directory_counts` in
`irma/core/endf_writer.py`).

* `coldh_skold.endf.gz` — a miniature ortho-hydrogen deck (6 alpha x 10
  beta, 14 K, diffusion translation + one discrete oscillator + a 12-point
  S(kappa) table, ncold=1 + nsk=2) pinning the cold-hydrogen rotational
  sums and the diffusion translational path byte-for-byte in fast CI
  (`tests/test_coldh_skold_minitape.py`). With ncold != 0 neither NJOY nor
  IRMA runs the Skold step (leapr.f90 `if nsk==2 and ncold==0`); the
  S(kappa) table feeds coldh's spin-correlation factors instead. The
  full-grid 7-temperature expected validation remains in
  `tests/native_LEAPR_NJOY_ENDF_validation/`.

* `skold.endf.gz` — the same deck with ncold=0, so the Skold correction
  runs; pins it byte-for-byte in the same test. Generated with the local
  NJOY2016.79 build (LEAPR unmodified upstream).

* `coldd_ortho.endf.gz` / `coldd_para.endf.gz` — miniature ortho-deuterium
  (ncold=3) and para-deuterium (ncold=4) decks (6 alpha x 10 beta, 19 K,
  diffusion translation + a 12-point S(kappa) table, nsk=0) pinning the
  cold-deuterium side of the coldh kernel — deuteron constants and the
  law=4/5 even/odd-J spin-correlation factors — byte-for-byte in fast CI
  (`tests/test_coldd_minitape.py`). Card 4/5 constants (mat 13/12,
  za 1002, awr 1.9968, spr 3.395, npr 2) and the translational weights
  (twt=.025, c=40., tbeta=.475, no discrete oscillators) follow the
  ENDF/B tsl-ortho-D / tsl-para-D LEAPR evaluations (LANL eval-apr93,
  LA-12639-MS); the grids and toy rho/S(kappa) shapes are those of the
  `coldh_skold` minitape. nsk=0 with ncold>0 also covers the
  Card 17/18-without-Card 19 deck-reading path. Generated with a local
  NJOY2016.78 build whose LEAPR source is unmodified upstream (a local
  patch touches THERMR only) and which regenerates `coldh_skold.endf.gz`
  byte-identically from its deck.

* `two_pass.endf.gz` — a miniature two-pass bound-secondary deck (4 alpha x
  6 beta, 296 K, principal + secondary spectrum blocks, nss=1 b7=0 mss=1)
  pinning the BeO-style mixed-moderator merge in fast CI
  (`tests/test_two_pass_minitape.py`). MF7 is byte-identical except the
  incoherent-elastic SB head record, which NJOY renders with its adaptive
  9-digit no-exponent form (deliberate format divergence — see
  endf_writer.py; compared numerically in the test).

The SHA256 of every reference tape here (and the slow-harness `tsl-*.endf.gz`
tapes) is pinned in `tests/SHA256SUMS.txt` and actively asserted by
`tests/test_reference_manifest.py`, so a silently re-gzipped, truncated, or
swapped reference fails loudly with a named hash mismatch.
