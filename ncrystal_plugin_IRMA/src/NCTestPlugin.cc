#include "NCTestPlugin.hh"

#include "NCrystal/internal/utils/NCMath.hh"
#include "NCrystal/internal/utils/NCMsg.hh"

void NCP::customPluginTest()
{
  NCPLUGIN_MSG("Testing IRMA plugin");

  const std::string base_data = "stdlib::Al_sg225.ncmat";
  std::string testdata = NC::FactImpl::createTextData(base_data)->rawDataCopy();
  testdata += "\n@CUSTOM_IRMA\n";
  testdata += "  pack plugins::IRMA/toy_precomputed.irmapack\n";

  // toy_precomputed.irmapack is baked at 300 K; load at that temperature. The base
  // NCMAT carries no @TEMPERATURE, so NCrystal would otherwise default to 293.15 K
  // and the plugin's temperature-match guard would (correctly) reject the mismatch.
  // All three scatters share the same temperature so the additivity check below
  // compares like with like.
  auto cfg = NC::MatCfg::createFromRawData( std::string(testdata), ";temp=300" );
  auto cfg_inelasonly = NC::MatCfg::createFromRawData(
    std::string(testdata), ";temp=300;comp=inelas" );

  auto scat = NC::createScatter( cfg );
  auto scat_inelasonly = NC::createScatter( cfg_inelasonly );
  auto scat_base_noinelas = NC::createScatter( base_data + ";inelas=0;temp=300" );

  bool found_positive_plugin_xs = false;
  for ( auto& wlval : NC::linspace(0.5,3.0,8) ) {
    auto wl = NC::NeutronWavelength( wlval );
    auto xs_total = scat.crossSectionIsotropic( wl );
    auto xs_plugin = scat_inelasonly.crossSectionIsotropic( wl );
    auto xs_base_noinelas = scat_base_noinelas.crossSectionIsotropic( wl );
    NCPLUGIN_MSG( "xs @ "<<wl<<" : total="<<xs_total
                  <<" plugin-inelas="<<xs_plugin
                  <<" base-noinelas="<<xs_base_noinelas );
    nc_assert_always( xs_total.dbl() >= 0.0 );
    nc_assert_always( xs_plugin.dbl() >= 0.0 );
    // Exact additivity (total == plugin-inelas + base-noinelas) holds ONLY because
    // toy_precomputed.irmapack is inelastic-only: the plugin takes over just the
    // inelastic channel, leaving the base elastic untouched. If the toy pack ever
    // gains elastic tensors the plugin would also disable the base elastic, and
    // this must become a per-channel decomposition.
    nc_assert_always( NC::floateq( xs_total.dbl(),
                                   xs_plugin.dbl() + xs_base_noinelas.dbl() ) );
    if ( xs_plugin.dbl() > 0.0 )
      found_positive_plugin_xs = true;
  }

  nc_assert_always( found_positive_plugin_xs );
  NCPLUGIN_MSG("IRMA plugin self-test succeeded");
}
