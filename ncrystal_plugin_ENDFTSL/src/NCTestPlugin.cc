#include "NCTestPlugin.hh"
#include "NCEndfCohElasScatter.hh"

#include "NCrystal/internal/utils/NCMath.hh"
#include "NCrystal/internal/utils/NCMsg.hh"

void NCP::customPluginTest()
{
  NCPLUGIN_MSG("Testing ENDFTSL plugin");

  // Unit-test the data-driven coherent-elastic sampler in isolation: tabulated
  // Bragg edges {1e-3, 2e-3} eV with cumulative S {0.4, 1.0} eV*barn.
  // sigma_coh(E) = S(E)/E, histogram in E (the cumulative value of the largest
  // edge <= E), and 0 below the first edge.
  NCP::EndfCohElasScatter coh( NC::VectD{ 1.0e-3, 2.0e-3 },
                               NC::VectD{ 0.4, 1.0 } );
  NC::CachePtr cp;

  const double xs_below = coh.crossSectionIsotropic( cp, NC::NeutronEnergy{ 0.5e-3 } ).dbl();
  nc_assert_always( xs_below == 0.0 );                       // below the first edge

  const double xs_1 = coh.crossSectionIsotropic( cp, NC::NeutronEnergy{ 1.5e-3 } ).dbl();
  nc_assert_always( NC::floateq( xs_1, 0.4 / 1.5e-3 ) );      // only the first edge below E

  const double xs_2 = coh.crossSectionIsotropic( cp, NC::NeutronEnergy{ 2.5e-3 } ).dbl();
  nc_assert_always( NC::floateq( xs_2, 1.0 / 2.5e-3 ) );      // both edges below E

  // 1/E fall-off above the last edge: at 2x the energy the cross section halves.
  const double xs_2b = coh.crossSectionIsotropic( cp, NC::NeutronEnergy{ 5.0e-3 } ).dbl();
  nc_assert_always( NC::floateq( xs_2b, 1.0 / 5.0e-3 ) );

  NCPLUGIN_MSG("ENDFTSL plugin self-test succeeded (coherent S(E)/E sampler)");
}
