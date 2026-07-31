# MLIP phonon models

`irma mlip` builds a phonon model from a pretrained machine-learned
interatomic potential (MLIP) for any ASE-readable crystal within the
chosen potential's element coverage: no DFT calculation, no
force-constant file, just a structure file and a choice of potential.
The result is a self-contained **bundle** that feeds all three
downstream IRMA tools:
the ENDF-6 tape generator, the `irma.spectra` forward model, and the
NCrystal exporter. On a laptop CPU a build takes seconds to minutes for ordinary cells;
the cost scales with the supercell size and the symmetry-reduced
displacement count.

The approach (pretrained universal interatomic potentials standing in for DFT in a phonopy finite-displacement workflow) follows ORNL's INSPIRED (Han, Savici, Li & Cheng, *Comput. Phys. Commun.* **304**, 109288 (2024), MIT), whose machine-learned-potential tab pioneered it for INS spectra; IRMA's front end generalizes it to all three IRMA outputs and inherits several of its conventions (the default mesh-density rule, the per-potential dtype usage).

The workflow is deliberately two-step. `irma mlip build` computes the
physics (relaxation, finite-displacement force constants, a quick-look
DOS) and writes the bundle; `irma mlip emit` turns a bundle into
ready-to-edit input files. Nothing runs the downstream calculations for
you: you inspect the generated inputs and drive `irma`, `irma spectra`,
or `irma ncrystal` yourself.

```bash
irma mlip env create nequip   # once per potential: provision + register its environment
irma mlip build MgO.cif -o mgo_bundle --potential nequip
irma mlip validate mgo_bundle
irma mlip emit mgo_bundle --to endf,spectra,ncrystal --mat 'Mg=44' --mat 'O=48'
```

!!! note "Why conservative forces are a hard requirement"
    Finite-displacement force constants assume the forces are exact
    gradients of a potential energy surface. Several published "direct
    force" model variants predict forces without that constraint, and the
    published benchmarks (the Matbench Discovery thermal-conductivity
    metric) show what happens to their phonons: imaginary modes
    everywhere and errors an order of magnitude worse than the same
    labs' conservative models. Where a backend offers both variants,
    IRMA selects the conservative one explicitly (ORB's conservative
    builder; upet's `non_conservative=False` on every construction),
    and no IRMA option can switch a model to direct-force prediction.

## The potentials

| `--potential` | default model | training data (level) | license | note |
|---|---|---|---|---|
| `nequip` | `mir-group/NequIP-OAM-L:0.1` | OMat24 + sAlex + MPtrj (PBE/PBE+U) | MIT / CC-BY-4.0 | best all-around in our validation |
| `grace` | `GRACE-2L-OAM` | OMat24 + sAlex + MPtrj (PBE/PBE+U) | **ASL (academic)** | TensorFlow; runs via a dedicated env |
| `orb` | `orb_v3_conservative_inf_omat` | OMat24 (PBE/PBE+U) | Apache-2.0 | |
| `sevennet` | `7net-mf-ompa` | MPtrj + sAlex + OMat24 (PBE/PBE+U) | GPL-3.0 | |
| `mattersim` | `MatterSim-v1.0.0-5M.pth` | Microsoft MD dataset (PBE) | MIT | needs Python >= 3.12 |
| `mace` | `medium-mpa-0` | MPtrj + sAlex (PBE/PBE+U) | MIT | foundation names (`medium-omat-0`, MATPES, MH) are **ASL** |
| `mace-off` | `medium` | SPICE organics (wB97M-D3) | ASL | molecules only (H C N O F P S Cl Br I) |
| `pet-mad` | `pet-mad-s` (newest release, pinned at build) | MAD (**r2SCAN**) | BSD-3-Clause | expect a small stiff offset vs PBE references |
| `dpa3` | `DPA-3.1-3M` | OpenLAM multitask (PBE/PBE+U) | LGPL / CC-BY-4.0 | open baseline; **not for van-der-Waals layered crystals** |

Potentials are never installed by `pip install irma[mlip]`; each brings
its own heavy dependency stack, and the stacks conflict with each
other. Install a potential with `irma mlip env create <potential>`,
which provisions and registers a dedicated environment for it; builds
then dispatch to that environment automatically (see
[Environments](#environments-and-conflicting-dependencies) for the
mechanism, the support matrix, and how to register an environment you
manage yourself). The `[mlip]` extra installs only `ase`, `phonopy`,
and `pyyaml`.

When a build uses an Academic Software License checkpoint (grace, the
newer MACE foundation models, mace-off's named checkpoints), the CLI
prints a license note and records it in the bundle manifest, so the
research-only restriction travels with the results. The license column
reflects what the upstream projects published at the time of writing
(July 2026); verify before any commercial use.

### Choosing a model

`--model` accepts a checkpoint name or a local file, with per-backend
syntax for the extra axes some potentials have:

- **pet-mad** — `NAME@VERSION` pins a released version
  (`--model pet-mad-s@1.5.0`). A bare name (or `@latest`) resolves to the
  newest release **once, at build start**, and the resolved checkpoint is
  pinned into the bundle; a mid-build upstream release cannot split the
  build. The PBE-level siblings (`pet-omat-s`, `pet-oam-l`, ...) are
  selected the same way.
- **dpa3** — `MODEL::HEAD` selects the fitting net of the multitask
  checkpoint (default `MP_traj_v024_alldata_mixu`, the MPtrj PBE+U head;
  `DPA-3.1-3M::Omat24` is the other materials head). A frozen single-task
  `.pth` file needs no head.
- **nequip** — a model-zoo id (`mir-group/NequIP-OAM-XL:0.1`; the
  `mir-group/` prefix is the default) or an already-compiled
  `.nequip.pt2`/`.nequip.pth` artifact. Zoo models are compiled once with
  `nequip-compile` into the cache and reloaded from there (per torch
  compile mode: crossing the torch 2.10 TorchScript/AOTInductor
  boundary triggers one recompilation).
- **mace** — the usual size names plus foundation checkpoints by name
  (`--model medium-omat-0`) or a downloaded `.model` file path.
- **grace** — a foundation-model name from the tensorpotential registry
  (`GRACE-2L-OAM`, `GRACE-1L-OAM`, ...). Local paths are not supported.

!!! tip "Recipe: Egret-1 for organic molecules"
    Rowan's Egret-1 (MIT per its repository; its paper reports the
    strongest molecular vibrational-frequency validation of the models
    we surveyed) loads through the existing
    mace-off wrapper: download `EGRET_1.model` from
    `github.com/rowansci/egret-public` and build with
    `--potential mace-off --model /path/EGRET_1.model`. The checkpoints
    carry no stress output, so `--relax-cell` is unavailable; for
    isolated molecules in a vacuum box that is no loss.

## What a build does

1. **Relax** the structure with the same potential (FIRE, default
   `--fmax 0.01` eV/A; `--relax-cell` adds the cell via a Frechet filter).
   The spacegroup is checked before and after, and a symmetry change is
   reported loudly. `--snap-symmetry [TOL]` then projects the positions
   onto the exact orbits of the spacegroup detected at TOL (bare flag:
   1e-2 A): float32 relaxations routinely land a hair off the ideal
   Wyckoff sites, which silently multiplies the displacement count and
   invalidates symmetry-reduced BORN files; the applied shift is
   recorded in the manifest.
2. **Displace** — phonopy generates the symmetry-reduced displacement set
   for the supercell (default: the smallest diagonal supercell whose repeat counts satisfy
   `ceil(12 A / a_i)` per lattice-vector length; note this is a
   lattice-parameter rule, not a true minimum-image criterion for
   strongly skewed cells; override with `--supercell "n1 n2 n3"` or
   an Lmin).
3. **Forces** — one potential evaluation per displaced supercell, serial
   by default. `--jobs N` runs N spawn workers with one native thread
   each, and for heavy models it is THE performance knob: pick a value
   that divides the displacement count evenly, up to your core count
   (see the performance notes below). Forces are cached under a
   physics fingerprint, so an interrupted build resumes where it
   stopped.
4. **Force constants** — drift correction, symmetrization, and a
   quick-look DOS/imaginary-mode census on a mesh you can override with
   `--mesh`. The DOS grid pitch is `min(0.5 meV, span/200)`; when the
   model has numerically dispersionless bands the linear tetrahedron
   method would drop entirely (isolated molecular modes such as an O–H
   stretch on a small supercell, or every band of a Γ-only mesh, the
   disordered default), the DOS and the emitters' species-projected DOS
   fall back to 1 meV Gaussian smearing automatically and the manifest
   census records `dos_smearing_fallback_mev`. `--dos-smearing MEV`
   forces a smearing width explicitly (the grid refines to resolve it).
5. **Bundle** — everything lands in one directory: `phonopy.yaml` with
   embedded force constants, the relaxed structure, DOS, and a manifest
   with the full provenance (potential; resolved checkpoint identity,
   content-hashed when the model is a local file or compiled artifact;
   package versions; license note if any; all arguments).

`irma mlip validate <bundle>` re-checks a bundle (manifest schema and
required fields, safe file names and symlink containment, file hashes,
embedded force constants, manifest/YAML NAC agreement, and a real
phonopy reload) and prints a summary. It is the documented first action
on a bundle you received rather than built.

!!! warning "A bundle is code-adjacent input"
    Treat a received bundle with the same caution as a script from the
    same source. The manifest's sha256 hashes prove *internal
    consistency* only — that the bundle's files match what its builder
    recorded — and an attacker-built bundle is perfectly
    self-consistent, so the hashes are worthless as evidence of origin
    or good faith. The concrete hazard is `phonopy.yaml`: phonopy parses
    it with PyYAML's **unsafe** loader, which executes `!!python/` tags
    at parse time. IRMA therefore scans every phonopy.yaml and refuses
    files carrying such tags *before* phonopy's parser sees them, at
    every entry point that parses one — `irma mlip validate` and bundle
    loading, the ENDF deck engine, the NCrystal exporter, and the GUI
    file picker. That scan is fail-closed for the known code-execution
    vector, **not** a sandbox: it does not make phonopy's parser safe or
    vouch for the bundle's physics. Validate first, and prefer bundles
    from sources you trust.

!!! note "Force-cache fine print"
    The cache key covers the relaxed geometry, displacement set, and the
    RESOLVED model identity (content hash for local files and compiled
    artifacts). Reruns reuse cached forces only when the relaxation lands
    on bit-identical positions: trivially true for structures that
    relax in zero steps, but float32 potentials that take real FIRE steps
    can land within tolerance yet not bit-identically, and then the
    forces recompute. That costs time, never correctness.

### Born effective charges

Pass `--born PATH` (phonopy BORN format) to embed non-analytical-term
corrections in the bundle. NAC is never read from the working directory,
and the emitted inputs are set up so the downstream tools use the
embedded values. The PET-MAD calculators advertise Born-charge and dielectric outputs,
but IRMA does not yet wire those predictions into NAC; supply
a BORN file from DFPT if LO-TO splitting matters for your material.

## Disordered and amorphous materials

Declare disorder explicitly with `--disordered`; IRMA never switches
behavior on a heuristic (a console hint appears when a large P1 cell
looks disordered, nothing more). The box is then its own supercell by default (an explicit
`--supercell` still wins), the mesh defaults to the Gamma point, and emission switches to the
DOS-driven classic path: incoherent elastic scaled by the total bound
cross section plus a contin-style inelastic deck per species. The
ncrystal target is refused for disordered bundles in this version.

## Emitting IRMA inputs

```bash
irma mlip emit <bundle> --to endf,spectra,ncrystal \
    --mat 'C=31' [--nuclide C=13-C] [--species 'C:b_coh_fm=6.646'] \
    [--principal C] [--temperature 296] [--out-dir DIR] \
    [--inelastic-mode 0|1|2] [--elastic-format mef|sef]
```

- **endf** — one ready-to-run deck per principal scatterer (`iel=10`,
  automatic alpha/beta grids), gated through the same parser the GUI
  uses before anything is written. The elastic convention is the mixed
  elastic format (MEF: both elastic components on every species' tape)
  by default; `--elastic-format sef` selects the single-channel
  convention. `--inelastic-mode` picks the physics level (default 2):
  modes 1/2 are the phonopy-backed directional decks (Card 6g
  `10000 1000 1`, the validation-campaign sampling and the GUI
  form's production default; per-species Debye-Waller
  from the displacement tensors), and mode 0 is the classic isotropic
  path driven by the
  bundle's species-projected DOS, with the principal's spectrum on the
  classic cards and a Card 6e partial spectrum for every other
  species, so each species' elastic W'(T) carries its own lambda. Mode
  0 is DOS-driven and therefore subject to the imaginary-mode gate
  (`--allow-unstable`). Every emitted deck also carries its provenance
  on the tape itself: comment cards 6+ — which the ENDF writer maps
  onto MF1/MT451 free-text DESCRIPTION records — record the bundle
  path and fingerprint, wrapped so no card exceeds the writer's
  66-column mapping (card 1 is the structured ZSYMAM/ALAB/EDATE
  header, cards 2–5 are left blank for you to fill in). A tape built
  from an emitted deck is therefore traceable to its bundle without
  the emit manifest.
- **spectra** — an `irma spectra` YAML with the bundle's phonon model
  wired in, validated against the real config schema.
- **ncrystal** — an exporter YAML for `irma ncrystal`, refused for
  disordered bundles.

Scattering constants come from the built-in nuclear-data table,
generated from `periodictable`'s neutron tables (the
Rauch–Waschkowski/Sears compilation). By default an element uses its
natural-abundance constants while the ENDF ZA identity defaults to the
most abundant isotope; `--nuclide` switches both the identity and the
constants to a specific isotope. `--species` overrides `awr`,
`b_coh_fm`, and `sigma_inc_b` directly; `sigma_bound_b` is always
derived from those (a supplied value acts as a 0.5% consistency
check). The emit manifest records the identity and constants source
for every species.

## Environments and conflicting dependencies

The potential packages cannot all live in one Python environment; the
pins are mutually unsatisfiable, and this is upstream reality rather
than an IRMA choice:

- `mace-torch` pins `e3nn==0.4.4`; `sevenn` and `mattersim` need
  `e3nn>=0.5`; `nequip` needs `e3nn>=0.6`.
- `deepmd-kit` wheels carry a C++ op library ABI-locked to the torch
  they were built against, and dlopen `libmpi` from the `mpich` wheel.
- `tensorpotential` (grace) runs on TensorFlow, not torch.
- `mattersim` and `orb-models` need Python >= 3.12.

The support matrix, as verified on the packages current at the time of
writing:

| potential | pip install | Python | key pins / gotchas |
|---|---|---|---|
| `nequip` | `nequip` | >= 3.10 | `e3nn>=0.6,<0.7`; excludes mace-torch from the same env |
| `grace` | `tensorpotential` | >= 3.9 | TensorFlow `<=2.20` (no torch); heavy runtime, isolate it |
| `orb` | `orb-models` | **>= 3.12** | `torch>=2.8,<3.0` |
| `sevennet` | `sevenn` | >= 3.10 | `e3nn>=0.5`; excludes mace-torch |
| `mattersim` | `mattersim` | **>= 3.12** | `e3nn>=0.5`, `numpy>=2.0`; excludes mace-torch |
| `mace` / `mace-off` | `mace-torch` | >= 3.9 | hard-pins `e3nn==0.4.4`; excludes sevenn, mattersim, nequip |
| `pet-mad` | `upet` | >= 3.11 | tightly-pinned metatensor CalVer stack; `torch>=2.3,<2.14` |
| `dpa3` | `deepmd-kit[torch]` + `mpich` + `huggingface_hub` | >= 3.10 | binaries ABI-locked to the pinned torch; `libmpi` dlopened from the `mpich` wheel |

The version windows reflect the upstream package metadata verified at
the time of writing (July 2026); `env create` itself pins only the
package names plus `ase>=3.23` and picks up whatever the upstreams
currently publish. The two e3nn camps (`mace-torch` versus everything
else that uses e3nn)
were both demonstrated to break live in either direction; this is not a
theoretical conflict. IRMA's answer is per-potential environments with
transparent dispatch:

```bash
irma mlip env create mace     # provision + register a dedicated env
irma mlip env list
irma mlip env remove mace
```

`env create` builds a standard virtual environment (with `uv` when it is
on PATH, otherwise the standard library's `venv` seeded from the running
interpreter; **conda is never required or invoked**), installs a
known-good requirement set, runs a backend-specific import smoke check,
and registers the interpreter in the irma-mlip cache. From then
on, `--potential mace` transparently runs its force calls in that
environment through a small subprocess server; relaxation, serial
builds, and parallel workers all dispatch the same way, and force caches
are shared across environments. The foreign environment needs only
`ase`, the potential package, and its per-backend helpers (the
requirement-set table is exactly what `env create` installs; dpa3 also
needs `mpich` and `huggingface_hub`), never IRMA itself.

Registration can also point at any environment you manage yourself:
the `IRMA_MLIP_PYTHON_<POTENTIAL>` environment variable (for example
`IRMA_MLIP_PYTHON_MACE=/opt/envs/mace/bin/python`) overrides the
registry, and setting it empty disables dispatch for that potential.
Dispatch engages whenever a registered interpreter differs from the
running one: a registered environment wins even if the potential is
also importable locally (unregister it, or set the variable empty, to
force in-process execution). With nothing registered, an importable
potential runs in-process exactly as before.

!!! warning "Debian/Ubuntu system Python"
    The stdlib-`venv` fallback needs the `python3-venv` system package on
    Debian-family distributions. Installing `uv` sidesteps that.

!!! warning "Linux: install the CPU torch wheel"
    On Linux, a bare `pip install torch` gives the CUDA build, and the
    CUDA wheel breaks `nequip-compile` even for CPU targets (a TF32
    flag clash inside torch). For CPU-only MLIP work install the CPU
    wheels explicitly (`pip install torch torchvision --index-url
    https://download.pytorch.org/whl/cpu`) and keep
    torchvision/torchaudio on the same flavor as torch (a mixed pair
    fails at import with "operator torchvision::nms does not exist").
    Verified on Linux: the full pipeline reproduces the macOS phonons
    identically (ZrO2: same freq_max and imaginary census), with the
    same near-linear `--jobs` scaling (serial 197 s -> 46 s at
    `--jobs 9` on a 40-core node).

Everything the front end downloads or builds lives under one cache
directory, `~/.cache/irma-mlip` by default (`IRMA_MLIP_CACHE` overrides
it): checkpoints, compiled nequip artifacts, provisioned environments,
and the environment registry.

## Validation snapshot

Maximum phonon frequency against DFT references, with the imaginary-mode
census on the shared meshes (Ni fcc vs a 4x4x4 Perlmutter reference;
graphite vs the published PBE model behind the IRMA paper, whose
seven-functional spread is 198.2-202.7 meV; wurtzite BeO vs the paper's
4x4x3 model, no NAC on either side). The census values below are backed
by the archived campaign record
[campaign_census_2026-07-17.txt](assets/mlip/campaign_census_2026-07-17.txt),
extracted from the per-bundle build manifests:

| potential | Ni | graphite | BeO | imaginary modes |
|---|---|---|---|---|
| nequip | -1.0% | +0.5% | +0.5% | none anywhere |
| grace | -3.7% | +0.1% | +4.6% | none anywhere |
| orb | -0.6% | +0.2% | +3.3% | small flexural artifacts on graphite |
| sevennet | -2.9% | +0.4% | -2.1% | small flexural artifacts on graphite |
| mattersim | -6.8% | +0.6% | +2.8% | none anywhere |
| pet-mad | -4.6% | +1.5% | +2.6% | none anywhere |
| mace | -3.3% | +2.9% | — | none |
| mace-off | — | -3.5% | — | out of domain on graphite (molecular training set, no interlayer physics) |
| dpa3 | -12.5% | -6.5% | -6.3% | graphite unstable (13 848) |

A fourth, harder case is monoclinic ZrO2 (baddeleyite, P2_1/c, the
phonondb mp-2858 PBEsol reference with its Born charges supplied via
`--born`, same 3x2x2 supercell, NAC on both sides; PBE-trained models
read slightly stiff here because the PBEsol cell is compressed relative
to their own equilibria):

| potential | vs PBEsol ref (98.58 meV) | imaginary |
|---|---|---|
| dpa3 | -0.4% | 4 |
| orb | +1.3% | 0 |
| nequip | +1.9% | 0 |
| sevennet | +2.0% | 0 |
| mace | +3.0% | 0 |
| pet-mad | +3.3% | 3556 |
| grace | +5.9% | 8 |
| mattersim | +6.6% | 48 |

Two lessons from that campaign: ionic crystals genuinely need `--born`
(without NAC even the DFT reference itself shows 92 imaginary modes),
and a BORN file written for the input symmetry can be rejected after
relaxation if the potential drifts the positions off the exact Wyckoff
sites (phonopy then expects one row per atom); supplying per-atom Born
rows sidesteps it.

Points worth carrying into your own material choices: pet-mad's stiff
offset on graphite and BeO is its r2SCAN training reference showing
through, not an error; dpa3 is systematically soft and cannot bind
van-der-Waals layers (both of its materials heads produce thousands of
imaginary modes on graphite), so keep it to three-dimensionally bonded
solids; and an MLIP landing within a couple of percent of one DFT
functional is within the spread of DFT functionals themselves.

## Performance notes

Measured on monoclinic ZrO2 (baddeleyite, 12-atom P2_1/c cell, 324-atom
supercell, 18 symmetry-reduced displacements; Apple-silicon laptop, 16
cores):

- **Worker memory scales with supercell size and architecture.** On a
  432-atom supercell, sevennet and dpa3 each needed ~24 GB *per worker*
  (they run comfortably at `--jobs 1`-`2` there; the other potentials
  are unremarkable). Watch the first wave's memory before committing to
  a large `--jobs` on big cells.
- **`--jobs` is the lever for heavy models.** With single-thread-clean
  workers, the NequIP force loop went from 458 s (serial, pathological
  artifact; see below) to 93.6 s serial and **21.1 s with `--jobs 9`**.
  Choose a jobs value that packs the displacement count into full waves
  (18 displacements: 9 workers = two full waves beats 16 workers = two
  ragged ones, because each extra worker also pays a model load).
- **Serial `--threads` barely matters for compiled artifacts** (nequip):
  IRMA compiles them for one thread, deliberately. AOTInductor bakes the
  compile-time thread configuration into the kernels; an artifact
  compiled at torch's multi-thread default ran **5x slower per call**
  than the single-thread compile (25.0 s vs 5.1 s per 324-atom force
  call) while ignoring every runtime clamp. IRMA therefore compiles
  nequip artifacts in a clamped environment and tags them (`-st1`) so a
  pathological artifact can never be reused from an older cache.
- **Fast models don't care.** MatterSim finishes the same ZrO2 force
  loop in 8-10 s under every configuration; tuning only pays off for
  nequip-class equivariant models on large supercells.
- **`--worker-threads`** exists for machines and backends where
  jobs x threads < cores leaves real headroom, but with single-threaded
  compiled artifacts the measured optimum was 1 thread per worker.

## Troubleshooting

- **Exit codes**: 130 = interrupted (cached displacement forces are
  kept; rerun to resume); 2 = usage or validation problem (bad model string,
  malformed bundle, unstable DOS emission without `--allow-unstable`);
  3 = relaxation did not converge (raise `--nmax`, loosen `--fmax`,
  add `--jitter-cycles 3`, or pass `--force` to record the residual and
  continue); 4 = a missing or
  broken potential dependency; the message names the pip package and,
  when a dedicated environment would fix it, the exact
  `irma mlip env create` command.
- **Imaginary modes in the build summary** usually mean the structure is
  not at a true minimum of that potential or the supercell truncates the
  force constants: tighten `--fmax`, enlarge `--supercell`, or try
  another potential. Small flexural artifacts are common in layered
  materials.
- **A relaxation that stalls above `--fmax` and stays flat** is usually
  an MLIP force-noise kink: the reported force stays finite at the
  energy minimum, so no optimizer can descend further.
  `--jitter-cycles 3` (default 0) kicks the min(3, natoms) highest-force
  atoms (Gaussian, 0.05 A per component, constraint-aware, fixed seed)
  and re-relaxes, each extra cycle with its own `--nmax` budget
  (`steps_taken` reports the cumulative total); a per-step observer
  keeps the lowest-residual frame seen anywhere, and `--snap-symmetry`
  applies to that frame afterwards. On a 302-atom PMMA glass this took
  mattersim from a 0.09 eV/A stall to 0.005-0.01. Residuals below
  ~0.01 eV/A are usually the potential's own noise floor -- pushing
  further does not reduce imaginary modes.
- **deepmd on a newer torch** fails with a CXX11 ABI message at load
  time; the dedicated environment (`irma mlip env create dpa3`) installs
  the matched `deepmd-kit[torch]` pin and is the supported route.
- **A stressless checkpoint** (Egret-1) works for everything except
  `--relax-cell`, which needs the stress tensor.
