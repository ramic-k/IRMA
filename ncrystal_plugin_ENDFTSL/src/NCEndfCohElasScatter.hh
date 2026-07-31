#ifndef NCPlugin_EndfCohElasScatter_hh
#define NCPlugin_EndfCohElasScatter_hh

#include "NCrystal/NCPluginBoilerplate.hh"
#include "NCrystal/interfaces/NCProcImpl.hh"

namespace NCPluginNamespace {

  // Data-driven coherent-elastic sampler. Consumes a TSL evaluation's OWN
  // tabulated cumulative structure-factor table S(E) at the Bragg edges
  // (ENDF MF7/MT2 LTHR=1/3), verbatim -- no crystal structure, no PowderBragg.
  //   sigma_coh(E) = S(E)/E      (histogram / staircase, INT=1; 0 below E_1)
  //   scattering cosine mu = 1 - 2 E_i/E   (elastic: E' = E)
  // This reproduces exactly what NJOY->ACE->MCNP/SCALE sample from the tape.
  class EndfCohElasScatter final : public NC::ProcImpl::ScatterIsotropicMat {
  public:
    // edges_eV: ascending Bragg-edge energies E_i [eV];
    // cumS_eVbarn: cumulative S_i [eV*barn] (same length, non-decreasing).
    EndfCohElasScatter( NC::VectD edges_eV, NC::VectD cumS_eVbarn );

    const char * name() const noexcept override { return NCPLUGIN_NAME_CSTR "CohElas"; }

    NC::CrossSect crossSectionIsotropic( NC::CachePtr&, NC::NeutronEnergy ) const override;
    NC::ScatterOutcomeIsotropic sampleScatterIsotropic( NC::CachePtr&, NC::RNG&,
                                                        NC::NeutronEnergy ) const override;

  private:
    NC::VectD m_edges;   // ascending Bragg-edge energies E_i [eV]
    NC::VectD m_cumS;    // cumulative S_i [eV*barn]
  };

}

#endif
