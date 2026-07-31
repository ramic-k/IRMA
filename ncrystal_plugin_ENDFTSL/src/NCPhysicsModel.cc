#include "NCPhysicsModel.hh"
#include "NCEndfCohElasScatter.hh"

#include "NCrystal/core/NCException.hh"
#include "NCrystal/factories/NCFactImpl.hh"
#include "NCrystal/internal/elincscatter/NCElIncScatter.hh"
#include "NCrystal/internal/sab/NCScatKnlData.hh"
#include "NCrystal/internal/sab/NCSABUtils.hh"
#include "NCrystal/internal/sabscatter/NCSABScatter.hh"
#include "NCrystal/internal/utils/NCString.hh"
#include "NCrystal/interfaces/NCProcImpl.hh"

#include <cmath>
#include <map>
#include <sstream>

namespace {

  constexpr const char * endftsl_magic = "ENDFTSLPACK_TEXT_V1";
  constexpr const char * custom_section_name = "ENDFTSL";

  std::string trim( std::string s )
  {
    auto is_ws = []( char c ) { return c==' ' || c=='\t' || c=='\r' || c=='\n'; };
    while ( !s.empty() && is_ws(s.front()) )
      s.erase(s.begin());
    while ( !s.empty() && is_ws(s.back()) )
      s.pop_back();
    return s;
  }

  std::vector<double> parseDoubles( const std::string& text,
                                    const std::string& fieldname )
  {
    std::vector<double> out;
    std::istringstream is(text);
    std::string token;
    while ( is >> token ) {
      double value = 0.0;
      if ( !NC::safe_str2dbl( token, value ) )
        NCRYSTAL_THROW2(BadInput,"Invalid numeric value '"<<token
                        <<"' in ENDFTSL pack field "<<fieldname);
      out.push_back(value);
    }
    return out;
  }

  double parseRequiredDouble( const std::map<std::string,std::string>& fields,
                              const std::string& key )
  {
    auto it = fields.find(key);
    if ( it == fields.end() )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack is missing required field "<<key);
    double value = 0.0;
    if ( !NC::safe_str2dbl( it->second, value ) )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack field "<<key
                      <<" has invalid numeric value '"<<it->second<<"'");
    return value;
  }

  struct PackData {
    std::string backend;
    std::string sabRepresentation = "scaled_sym_sab";
    double temperatureK = 0.0;
    double boundXS = 0.0;
    double elementMassAMU = 0.0;
    double elasticMSD = -1.0;
    double elasticIncohXS = -1.0;
    double elasticScale = 1.0;
    NC::VectD cohEdgesEV;     // ENDF MF7/MT2 LTHR=1/3 Bragg edge energies [eV]
    NC::VectD cohCumS;        // cumulative S_i [eV*barn]
    NC::VectD alphaGrid;
    NC::VectD betaGrid;
    NC::VectD sabValues;

    bool providesIncoherentElastic() const
    {
      return elasticMSD > 0.0 && elasticIncohXS >= 0.0;
    }

    bool providesCoherentElastic() const
    {
      return !cohEdgesEV.empty();
    }
  };

  PackData loadPack( const std::string& path )
  {
    auto textData = NC::FactImpl::createTextData( path );
    std::istringstream is( textData->rawDataCopy() );
    std::string line;
    if ( !std::getline(is,line) || trim(line) != endftsl_magic )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path<<"' has invalid magic");

    std::map<std::string,std::string> fields;
    unsigned lineno = 1;
    while ( std::getline(is,line) ) {
      ++lineno;
      line = trim(line);
      if ( line.empty() || line.front()=='#' )
        continue;
      auto eqpos = line.find('=');
      if ( eqpos == std::string::npos )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path<<"' line "<<lineno
                        <<" should be 'key = value'");
      auto key = trim( line.substr(0,eqpos) );
      auto value = trim( line.substr(eqpos+1) );
      // meta.* lines carry provenance only and are ignored by the loader.
      if ( key.rfind("meta.",0)==0 )
        continue;
      fields[key] = value;
    }

    {
      auto svIt = fields.find("schema_version");
      if ( svIt == fields.end() )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' is missing required field schema_version");
      if ( svIt->second != "1" )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' has unsupported schema_version '"<<svIt->second<<"'");
    }
    auto unitsIt = fields.find("units");
    if ( unitsIt == fields.end() )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' is missing required field units");
    if ( unitsIt->second != "angstrom_eV_barn_K" )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' has unsupported units '"<<unitsIt->second<<"'");

    PackData out;
    auto backendIt = fields.find("backend");
    if ( backendIt == fields.end() )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path<<"' is missing backend");
    out.backend = backendIt->second;
    if ( out.backend != "endf_direct" )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' has unsupported backend '"<<out.backend<<"'");

    auto reprIt = fields.find("sab_representation");
    if ( reprIt != fields.end() )
      out.sabRepresentation = reprIt->second;
    if ( out.sabRepresentation != "sab" && out.sabRepresentation != "scaled_sym_sab" )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' has unsupported sab_representation '"
                      <<out.sabRepresentation<<"'");

    out.temperatureK = parseRequiredDouble( fields, "temperature_K" );
    out.boundXS = parseRequiredDouble( fields, "bound_xs_barn" );
    out.elementMassAMU = parseRequiredDouble( fields, "element_mass_amu" );

    // Incoherent elastic block (MF7/MT2 LTHR=2/3) -- optional, all-or-nothing.
    if ( fields.count("elastic_scale") ) {
      out.elasticScale = parseRequiredDouble( fields, "elastic_scale" );
      if ( out.elasticScale <= 0.0 )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' has non-positive elastic_scale");
    }
    const bool hasElasticMSD = fields.count("elastic_msd_a2") != 0;
    const bool hasElasticXS = fields.count("elastic_incoherent_xs_barn") != 0;
    if ( hasElasticMSD != hasElasticXS )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' must provide elastic_msd_a2 and "
                      "elastic_incoherent_xs_barn together");
    if ( hasElasticMSD ) {
      out.elasticMSD = parseRequiredDouble( fields, "elastic_msd_a2" );
      out.elasticIncohXS = parseRequiredDouble( fields, "elastic_incoherent_xs_barn" );
      if ( out.elasticMSD <= 0.0 )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' has non-positive elastic_msd_a2");
      if ( out.elasticIncohXS < 0.0 )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' has negative elastic_incoherent_xs_barn");
    }

    // Coherent elastic block (MF7/MT2 LTHR=1/3) -- optional, all-or-nothing.
    const bool hasCohEdges = fields.count("coh_edges_ev") != 0;
    const bool hasCohS = fields.count("coh_cumulative_s") != 0;
    if ( hasCohEdges != hasCohS )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' must provide coh_edges_ev and coh_cumulative_s together");
    if ( hasCohEdges ) {
      out.cohEdgesEV = parseDoubles( fields["coh_edges_ev"], "coh_edges_ev" );
      out.cohCumS = parseDoubles( fields["coh_cumulative_s"], "coh_cumulative_s" );
      if ( out.cohEdgesEV.empty()
           || out.cohEdgesEV.size() != out.cohCumS.size() )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' has coh_edges_ev / coh_cumulative_s of unequal "
                        "or zero length");
      // EndfCohElasScatter relies on strictly-ascending positive finite edges
      // (upper_bound staircase; sigma = S(E)/E would diverge as E->0 if the
      // first edge were non-positive) and finite non-decreasing non-negative
      // cumulative S (m_cumS[n-1]/E is returned verbatim); a hand-edited pack
      // violating any of these would give a silently wrong sigma/mu. Reject it
      // at load. NC::safe_str2dbl deliberately accepts the literals
      // "nan"/"inf" and NaN passes every ordinary comparison, so element 0 --
      // which the pairwise loop below never inspects -- gets its own NaN-safe
      // !(a>b)-form checks, and every element is finiteness-checked.
      if ( !( out.cohEdgesEV[0] > 0.0 ) || !std::isfinite( out.cohEdgesEV[0] ) )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' coh_edges_ev must start with a positive finite "
                        "energy (index 0)");
      if ( !( out.cohCumS[0] >= 0.0 ) || !std::isfinite( out.cohCumS[0] ) )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' coh_cumulative_s must start with a non-negative "
                        "finite value (index 0)");
      for ( std::size_t i = 1; i < out.cohEdgesEV.size(); ++i ) {
        if ( !( out.cohEdgesEV[i] > out.cohEdgesEV[i-1] )
             || !std::isfinite( out.cohEdgesEV[i] ) )
          NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                          <<"' coh_edges_ev is not finite and strictly "
                          "ascending (index "<<i<<")");
        if ( !( out.cohCumS[i] >= out.cohCumS[i-1] )
             || !std::isfinite( out.cohCumS[i] ) )
          NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                          <<"' coh_cumulative_s is non-finite or decreases "
                          "(index "<<i<<")");
      }
    }

    // Require the inelastic grids explicitly: std::map::operator[] would otherwise
    // insert an empty string for a missing field, yielding empty grids that pass the
    // 0*0==0 size check and produce an ill-defined kernel far downstream.
    for ( const char * key : { "alpha_grid", "beta_grid", "sab_values" } )
      if ( !fields.count(key) )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                        <<"' is missing required field "<<key);
    out.alphaGrid = parseDoubles( fields["alpha_grid"], "alpha_grid" );
    out.betaGrid = parseDoubles( fields["beta_grid"], "beta_grid" );
    out.sabValues = parseDoubles( fields["sab_values"], "sab_values" );

    if ( out.alphaGrid.empty() || out.betaGrid.empty() || out.sabValues.empty() )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path
                      <<"' has an empty alpha_grid/beta_grid/sab_values");
    if ( out.alphaGrid.size() * out.betaGrid.size() != out.sabValues.size() )
      NCRYSTAL_THROW2(BadInput,"ENDFTSL pack '"<<path<<"' has "
                      "sab_values size which does not equal alpha_grid x beta_grid");
    return out;
  }

  std::vector<std::string> parseCustomSectionForPackPaths( const NC::Info& info )
  {
    if ( info.countCustomSections( custom_section_name ) != 1 )
      NCRYSTAL_THROW2(BadInput,"Exactly one @CUSTOM_"<<custom_section_name
                      <<" section is required");

    std::vector<std::string> packPaths;
    for ( const auto& line : info.getCustomSection( custom_section_name ) ) {
      if ( line.empty() )
        continue;
      if ( line.size()>=2 && line.at(0)=="pack" ) {
        // Join tokens after the keyword so an absolute path containing spaces
        // (common on macOS) is not rejected by the bare 2-token check.
        std::string p = line.at(1);
        for ( std::size_t i = 2; i < line.size(); ++i )
          p += " " + line.at(i);
        packPaths.push_back( p );
      } else {
        NCRYSTAL_THROW2(BadInput,"Unknown @CUSTOM_"<<custom_section_name
                        <<" entry. Expected: pack <path>");
      }
    }

    if ( packPaths.empty() )
      NCRYSTAL_THROW2(BadInput,"@CUSTOM_"<<custom_section_name
                      <<" section must contain at least one: pack <path>");
    return packPaths;
  }

  std::vector<PackData> loadPacks( const std::vector<std::string>& packPaths )
  {
    std::vector<PackData> packs;
    packs.reserve( packPaths.size() );
    for ( const auto& packPath : packPaths )
      packs.push_back( loadPack( packPath ) );
    return packs;
  }

  void requireTemperatureMatch( const NC::Info& info,
                                const std::vector<PackData>& packs )
  {
    // Each pack carries a SAB precomputed at exactly one bake temperature. The
    // base NCMAT carries no @TEMPERATURE (that is NCMAT v7+; the exporter emits
    // v5), so NCrystal defaults the material to 293.15 K unless the caller passes
    // ;temp=. Without this guard a mismatched request (e.g. ;temp=500 against a
    // 296 K pack, or an omitted ;temp against a non-293.15 K pack) would silently
    // sample the pack's precomputed law at the WRONG temperature. The plugin does
    // not interpolate across temperatures, so a mismatch is a hard error.
    if ( !info.hasTemperature() )
      return;
    const double reqT = info.getTemperature().dbl();
    for ( const auto& pack : packs ) {
      const double tolK = 1.0e-3 + 1.0e-5 * pack.temperatureK;
      if ( std::abs( reqT - pack.temperatureK ) > tolK )
        NCRYSTAL_THROW2(BadInput,"ENDFTSL pack was baked at "<<pack.temperatureK
                        <<" K but the requested material temperature is "<<reqT
                        <<" K. Load the material at the pack temperature (e.g. "
                        "'<material>.ncmat;temp="<<pack.temperatureK<<"'); the "
                        "ENDFTSL plugin samples the precomputed law and does not "
                        "interpolate across temperatures.");
    }
  }

  NC::ProcImpl::ProcPtr createPluginProcess( const NC::Info& info,
                                             std::vector<PackData>&& packs,
                                             bool includeInelastic,
                                             bool includeIncoherentElastic,
                                             bool includeCoherentElastic )
  {
    requireTemperatureMatch( info, packs );
    NC::ProcImpl::ProcComposition::ComponentList components;
    for ( auto& pack : packs ) {
      if ( includeCoherentElastic && pack.providesCoherentElastic() ) {
        components.push_back(
          { 1.0, NC::makeSO<NCPluginNamespace::EndfCohElasScatter>(
              NC::VectD{ pack.cohEdgesEV }, NC::VectD{ pack.cohCumS } ) } );
      }
      if ( includeInelastic ) {
        NC::ScatKnlData skd;
        skd.knltype = ( pack.sabRepresentation == "scaled_sym_sab"
                        ? NC::ScatKnlData::KnlType::SCALED_SYM_SAB
                        : NC::ScatKnlData::KnlType::SAB );
        skd.betaGridOptimised = true;
        skd.temperature = NC::Temperature{pack.temperatureK};
        skd.boundXS = NC::SigmaBound{pack.boundXS};
        skd.elementMassAMU = NC::AtomMass{pack.elementMassAMU};
        skd.alphaGrid = std::move(pack.alphaGrid);
        skd.betaGrid = std::move(pack.betaGrid);
        skd.sab = std::move(pack.sabValues);

        auto sabData = NC::SABUtils::transformKernelToStdFormat( std::move(skd) );
        components.push_back( { 1.0, NC::makeSO<NC::SABScatter>( std::move(sabData) ) } );
      }
      if ( includeIncoherentElastic && pack.providesIncoherentElastic() ) {
        components.push_back(
          { 1.0, NC::makeSO<NC::ElIncScatter>(
              NC::VectD{ pack.elasticMSD },
              NC::VectD{ pack.elasticIncohXS },
              NC::VectD{ pack.elasticScale } ) } );
      }
    }
    if ( components.empty() )
      NCRYSTAL_THROW(BadInput,"ENDFTSL plugin was requested but no plugin "
                     "scatter component was enabled");
    return NC::ProcImpl::ProcComposition::consumeAndCombine(
      std::move(components), NC::ProcessType::Scatter );
  }

}

bool NCP::PhysicsModel::isApplicable( const NC::Info& info )
{
  return info.countCustomSections( custom_section_name ) > 0;
}

bool NCP::PhysicsModel::providesIncoherentElastic( const NC::Info& info )
{
  for ( const auto& pack : loadPacks( parseCustomSectionForPackPaths( info ) ) ) {
    if ( pack.providesIncoherentElastic() )
      return true;
  }
  return false;
}

bool NCP::PhysicsModel::providesCoherentElastic( const NC::Info& info )
{
  for ( const auto& pack : loadPacks( parseCustomSectionForPackPaths( info ) ) ) {
    if ( pack.providesCoherentElastic() )
      return true;
  }
  return false;
}

NCP::PhysicsModel NCP::PhysicsModel::createFromInfo( const NC::Info& info,
                                                     bool includeInelastic,
                                                     bool includeIncoherentElastic,
                                                     bool includeCoherentElastic )
{
  return PhysicsModel( info, includeInelastic, includeIncoherentElastic,
                       includeCoherentElastic );
}

NCP::PhysicsModel::PhysicsModel( const NC::Info& info,
                                 bool includeInelastic,
                                 bool includeIncoherentElastic,
                                 bool includeCoherentElastic )
  // NC::ProcImpl::ProcPtr (shared_obj) is non-null and not default-constructible,
  // so it must be built in the initializer list, not assigned in the body.
  : m_proc( createPluginProcess( info,
                                 loadPacks( parseCustomSectionForPackPaths( info ) ),
                                 includeInelastic,
                                 includeIncoherentElastic,
                                 includeCoherentElastic ) )
{
}

double NCP::PhysicsModel::calcCrossSection( NC::CachePtr& cache,
                                            double neutron_ekin ) const
{
  return m_proc->crossSectionIsotropic( cache, NC::NeutronEnergy{neutron_ekin} ).dbl();
}

NCP::PhysicsModel::ScatEvent
NCP::PhysicsModel::sampleScatteringEvent( NC::CachePtr& cache, NC::RNG& rng,
                                          double neutron_ekin ) const
{
  auto outcome = m_proc->sampleScatterIsotropic( cache, rng,
                                                 NC::NeutronEnergy{neutron_ekin} );
  return { outcome.ekin.dbl(), outcome.mu.dbl() };
}
