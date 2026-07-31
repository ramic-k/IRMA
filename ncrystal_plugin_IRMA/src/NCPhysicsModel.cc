#include "NCPhysicsModel.hh"

#include "NCrystal/core/NCException.hh"
#include "NCrystal/factories/NCFactImpl.hh"
#include "NCrystal/internal/elincscatter/NCElIncScatter.hh"
#include "NCrystal/internal/extd_utils/NCOrientUtils.hh"
#include "NCrystal/internal/extd_utils/NCPlaneProvider.hh"
#include "NCrystal/internal/powderbragg/NCPowderBragg.hh"
#include "NCrystal/internal/sab/NCScatKnlData.hh"
#include "NCrystal/internal/sab/NCSABUtils.hh"
#include "NCrystal/internal/sabscatter/NCSABScatter.hh"
#include "NCrystal/internal/utils/NCMath.hh"
#include "NCrystal/internal/utils/NCString.hh"
#include "NCrystal/interfaces/NCProcImpl.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <map>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

namespace {

  constexpr const char * irma_magic = "IRMAPACK_TEXT_V1";
  constexpr const char * custom_section_name = "IRMA";

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
                        <<"' in IRMA pack field "<<fieldname);
      out.push_back(value);
    }
    return out;
  }

  std::vector<std::string> parseStrings( const std::string& text )
  {
    std::vector<std::string> out;
    std::istringstream is(text);
    std::string token;
    while ( is >> token )
      out.push_back(token);
    return out;
  }

  double parseRequiredDouble( const std::map<std::string,std::string>& fields,
                              const std::string& key )
  {
    auto it = fields.find(key);
    if ( it == fields.end() )
      NCRYSTAL_THROW2(BadInput,"IRMA pack is missing required field "<<key);
    double value = 0.0;
    if ( !NC::safe_str2dbl( it->second, value ) )
      NCRYSTAL_THROW2(BadInput,"IRMA pack field "<<key
                      <<" has invalid numeric value '"<<it->second<<"'");
    return value;
  }

  bool parseOptionalBool( const std::map<std::string,std::string>& fields,
                          const std::string& key,
                          bool defaultValue )
  {
    auto it = fields.find(key);
    if ( it == fields.end() )
      return defaultValue;
    auto value = trim( it->second );
    for ( auto& c : value )
      c = static_cast<char>( std::tolower( static_cast<unsigned char>(c) ) );
    if ( value == "1" || value == "true" || value == "yes" || value == "on" )
      return true;
    if ( value == "0" || value == "false" || value == "no" || value == "off" )
      return false;
    NCRYSTAL_THROW2(BadInput,"IRMA pack field "<<key
                    <<" has invalid boolean value '"<<it->second<<"'");
  }

  bool tensorIsPositiveSemidefinite( const NC::VectD& tensors,
                                     std::size_t offset )
  {
    constexpr double tol = 1.0e-14;
    for ( unsigned i = 0; i < 9; ++i ) {
      if ( !std::isfinite( tensors[offset+i] ) )
        return false;
    }
    if ( std::abs( tensors[offset+1] - tensors[offset+3] ) > tol )
      return false;
    if ( std::abs( tensors[offset+2] - tensors[offset+6] ) > tol )
      return false;
    if ( std::abs( tensors[offset+5] - tensors[offset+7] ) > tol )
      return false;
    const double a = tensors[offset];
    const double b = tensors[offset+1];
    const double c = tensors[offset+2];
    const double d = tensors[offset+4];
    const double e = tensors[offset+5];
    const double f = tensors[offset+8];
    const double det = a*d*f + 2.0*b*c*e - a*e*e - d*c*c - f*b*b;
    return a >= -tol
           && d >= -tol
           && f >= -tol
           && a*d - b*b >= -tol
           && a*f - c*c >= -tol
           && d*f - e*e >= -tol
           && det >= -tol;
  }

  struct PackData {
    std::string backend;
    std::string sabRepresentation = "sab";
    double temperatureK = 0.0;
    double boundXS = 0.0;
    double elementMassAMU = 0.0;
    double elasticMSD = -1.0;
    double elasticIncohXS = -1.0;
    double elasticScale = 1.0;
    bool elasticCoherent = true;
    NC::VectD elasticUTensors;
    std::vector<std::string> elasticUSymbols;
    NC::VectD elasticUFracPositions;
    // Per-tensor-site neutron data (the IRMA config's b_coh / sigma_inc). The
    // coherent F(hkl) + incoherent DW read THESE, not NCrystal's atom DB, so the
    // plugin honors exactly what IRMA specified. Coherent scat-len is in sqrt(barn)
    // (NCrystal coherentScatLen() unit = b_coh_fm/10); incoherent xs is in barn.
    NC::VectD elasticUCohScatLen;
    NC::VectD elasticUIncohXS;
    // 'isotropic' (default): trace/3 collapse into NCrystal's ElIncScatter.
    // 'directional': sample the orientation-averaged <exp(-Q^2 uhat.U.uhat)>
    // per site (mirrors irma.core.incoherent_dw).
    std::string incoherentElasticMode = "isotropic";
    NC::VectD alphaGrid;
    NC::VectD betaGrid;
    NC::VectD sabValues;

    bool providesIncoherentElastic() const
    {
      return ( elasticMSD > 0.0 && elasticIncohXS >= 0.0 )
             || !elasticUTensors.empty();
    }

    bool providesCoherentElastic() const
    {
      return elasticCoherent && ( elasticMSD > 0.0 || !elasticUTensors.empty() );
    }

    bool providesTensorCoherentElastic() const
    {
      return !elasticUTensors.empty();
    }
  };

  PackData loadPack( const std::string& path )
  {
    auto textData = NC::FactImpl::createTextData( path );
    std::istringstream is( textData->rawDataCopy() );
    std::string line;
    if ( !std::getline(is,line) || trim(line) != irma_magic )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' has invalid magic");

    std::map<std::string,std::string> fields;
    unsigned lineno = 1;
    while ( std::getline(is,line) ) {
      ++lineno;
      line = trim(line);
      if ( line.empty() || line.front()=='#' )
        continue;
      auto eqpos = line.find('=');
      if ( eqpos == std::string::npos )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' line "<<lineno
                        <<" should be 'key = value'");
      auto key = trim( line.substr(0,eqpos) );
      auto value = trim( line.substr(eqpos+1) );
      // meta.* lines carry provenance only and are ignored by the loader.
      if ( key.rfind("meta.",0)==0 )
        continue;
      fields[key] = value;
    }

    {
      // schema_version 2 is the current and only supported schema. (v1 was a
      // pre-release scaffold layout that IRMA never shipped a real pack for.)
      // Use find(), not operator[], so a missing key gives a clear "missing
      // field" error instead of silently inserting an empty string.
      auto svIt = fields.find("schema_version");
      if ( svIt == fields.end() )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' is missing required field schema_version");
      if ( svIt->second != "2" )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has unsupported schema_version '"<<svIt->second<<"'");
    }
    auto unitsIt = fields.find("units");
    if ( unitsIt == fields.end() )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                      <<"' is missing required field units");
    if ( unitsIt->second != "angstrom_meV_barn_K" )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                      <<"' has unsupported units '"<<unitsIt->second<<"'");

    PackData out;
    auto backendIt = fields.find("backend");
    if ( backendIt == fields.end() )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' is missing backend");
    out.backend = backendIt->second;

    if ( out.backend != "precomputed_sab" )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                      <<"' has unsupported backend '"<<out.backend<<"'");

    auto reprIt = fields.find("sab_representation");
    if ( reprIt != fields.end() )
      out.sabRepresentation = reprIt->second;
    if ( out.sabRepresentation != "sab" && out.sabRepresentation != "scaled_sym_sab" )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                      <<"' has unsupported sab_representation '"
                      <<out.sabRepresentation<<"'");

    out.temperatureK = parseRequiredDouble( fields, "temperature_K" );
    out.boundXS = parseRequiredDouble( fields, "bound_xs_barn" );
    out.elementMassAMU = parseRequiredDouble( fields, "element_mass_amu" );

    if ( fields.count("elastic_scale") ) {
      out.elasticScale = parseRequiredDouble( fields, "elastic_scale" );
      if ( out.elasticScale <= 0.0 )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has non-positive elastic_scale");
    }
    const bool hasElasticMSD = fields.count("elastic_msd_a2") != 0;
    const bool hasElasticXS = fields.count("elastic_incoherent_xs_barn") != 0;
    if ( hasElasticMSD != hasElasticXS )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                      <<"' must provide elastic_msd_a2 and "
                      "elastic_incoherent_xs_barn together");
    if ( hasElasticMSD ) {
      out.elasticMSD = parseRequiredDouble( fields, "elastic_msd_a2" );
      out.elasticIncohXS = parseRequiredDouble( fields, "elastic_incoherent_xs_barn" );
      if ( out.elasticMSD <= 0.0 )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has non-positive elastic_msd_a2");
      if ( out.elasticIncohXS < 0.0 )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has negative elastic_incoherent_xs_barn");
    }
    out.elasticCoherent = parseOptionalBool( fields, "elastic_coherent", true );
    if ( fields.count("elastic_u_tensors_a2") ) {
      out.elasticUTensors = parseDoubles( fields["elastic_u_tensors_a2"],
                                          "elastic_u_tensors_a2" );
      if ( out.elasticUTensors.size() % 9 != 0 )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has elastic_u_tensors_a2 length which is not "
                        "a multiple of 9");
      const auto nsites = out.elasticUTensors.size() / 9;
      out.elasticUSymbols = parseStrings( fields["elastic_u_symbols"] );
      out.elasticUFracPositions = parseDoubles(
        fields["elastic_u_frac_positions"],
        "elastic_u_frac_positions" );
      if ( out.elasticUSymbols.size() != nsites )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has elastic_u_symbols length which does not "
                        "match elastic_u_tensors_a2 site count");
      if ( out.elasticUFracPositions.size() != 3 * nsites )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has elastic_u_frac_positions length which does not "
                        "equal 3 x elastic_u_tensors_a2 site count");
      // Per-site neutron data is mandatory alongside the tensors: the coherent
      // F(hkl) + incoherent DW use it INSTEAD of NCrystal's atom DB (no silent
      // fallback), so the plugin reproduces exactly the IRMA-config b_coh/sigma_inc.
      if ( !fields.count("elastic_u_coherent_scatlen_sqrtbarn")
           || !fields.count("elastic_u_incoherent_xs_barn") )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' has elastic_u_tensors_a2 but "
                        "is missing elastic_u_coherent_scatlen_sqrtbarn and/or "
                        "elastic_u_incoherent_xs_barn (required per-site neutron data)");
      out.elasticUCohScatLen = parseDoubles(
        fields["elastic_u_coherent_scatlen_sqrtbarn"],
        "elastic_u_coherent_scatlen_sqrtbarn" );
      out.elasticUIncohXS = parseDoubles(
        fields["elastic_u_incoherent_xs_barn"], "elastic_u_incoherent_xs_barn" );
      if ( out.elasticUCohScatLen.size() != nsites )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has elastic_u_coherent_scatlen_sqrtbarn length which "
                        "does not match elastic_u_tensors_a2 site count");
      if ( out.elasticUIncohXS.size() != nsites )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has elastic_u_incoherent_xs_barn length which does not "
                        "match elastic_u_tensors_a2 site count");
      // Finiteness of the per-site neutron data: NC::safe_str2dbl deliberately
      // accepts the literals "nan"/"inf", and NaN passes every ordinary
      // comparison (a bare `< 0.0` test included) as well as PowderBragg's own
      // plane-weight guard, so a non-finite b_coh/sigma_inc would propagate to
      // a NaN Bragg cross section / Debye-Waller factor instead of a load
      // error. Reject it here. (b_coh may legitimately be negative, e.g. 1H.)
      for ( const auto bcoh : out.elasticUCohScatLen )
        if ( !std::isfinite( bcoh ) )
          NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' has non-finite "
                          "elastic_u_coherent_scatlen_sqrtbarn");
      for ( const auto sigmaInc : out.elasticUIncohXS )
        if ( !( sigmaInc >= 0.0 ) || !std::isfinite( sigmaInc ) )
          NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' has negative or "
                          "non-finite elastic_u_incoherent_xs_barn");
      for ( std::size_t isite = 0; isite < nsites; ++isite ) {
        if ( !tensorIsPositiveSemidefinite( out.elasticUTensors, 9 * isite ) )
          NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                          <<"' has elastic_u_tensors_a2 tensor which is not "
                          "finite symmetric positive semidefinite");
      }
    }

    if ( fields.count("incoherent_elastic_mode") ) {
      out.incoherentElasticMode = fields["incoherent_elastic_mode"];
      if ( out.incoherentElasticMode != "isotropic"
           && out.incoherentElasticMode != "directional" )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has unsupported incoherent_elastic_mode '"
                        <<out.incoherentElasticMode<<"'");
    }
    if ( out.incoherentElasticMode == "directional"
         && out.elasticUTensors.empty() )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                      <<"' has incoherent_elastic_mode = directional but no "
                      "elastic_u_tensors_a2 (the mode requires per-site "
                      "displacement tensors)");

    // Require the inelastic grids explicitly (mirrors the sister ENDFTSL loader):
    // std::map::operator[] would otherwise default-insert an empty string for a
    // missing field, and three empty grids would pass a bare 0*0==0 size check and
    // produce an ill-defined kernel far downstream.
    for ( const char * key : { "alpha_grid", "beta_grid", "sab_values" } ) {
      auto it = fields.find(key);
      if ( it == fields.end() )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' is missing required field "<<key);
      if ( it->second.empty() )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' has empty required field "<<key);
    }
    out.alphaGrid = parseDoubles( fields["alpha_grid"], "alpha_grid" );
    out.betaGrid = parseDoubles( fields["beta_grid"], "beta_grid" );
    out.sabValues = parseDoubles( fields["sab_values"], "sab_values" );

    // The SAB interpolation/sampling machinery assumes sorted finite grids of at
    // least two points and a full finite non-negative kernel; a hand-edited or
    // truncated pack violating any of these would otherwise surface as a cryptic
    // downstream NCrystal error or a silently wrong law. Reject it at load.
    {
      const struct { const char * name; const NC::VectD& grid; } grids[] = {
        { "alpha_grid", out.alphaGrid }, { "beta_grid", out.betaGrid } };
      for ( const auto& g : grids ) {
        if ( g.grid.size() < 2 )
          NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' field "<<g.name
                          <<" must contain at least two points");
        for ( std::size_t i = 0; i < g.grid.size(); ++i ) {
          if ( !std::isfinite( g.grid[i] ) )
            NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' field "<<g.name
                            <<" has a non-finite value (index "<<i<<")");
          if ( i > 0 && !( g.grid[i] > g.grid[i-1] ) )
            NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' field "<<g.name
                            <<" is not strictly increasing (index "<<i<<")");
        }
      }
    }
    if ( out.alphaGrid.size() * out.betaGrid.size() != out.sabValues.size() )
      NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path<<"' has "
                      "sab_values size which does not equal alpha_grid x beta_grid");
    for ( std::size_t i = 0; i < out.sabValues.size(); ++i ) {
      if ( !std::isfinite( out.sabValues[i] ) || out.sabValues[i] < 0.0 )
        NCRYSTAL_THROW2(BadInput,"IRMA pack '"<<path
                        <<"' field sab_values has a non-finite or negative value "
                        "(index "<<i<<")");
    }
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

  // Setup-cost cache (review CPP-8): for one material creation NCrystal runs
  // query() (providesIncoherentElastic + providesCoherentElastic) and then
  // produce() (the same pair again + createFromInfo), and each of those calls
  // used to re-read and re-tokenise every referenced pack from scratch --
  // measured as a 2.7x setup slowdown on PMMA. Parsed packs are therefore
  // cached here, keyed by the Info instance's UniqueID: NCrystal unique ids
  // come from a process-global monotonic counter and are never reused (the
  // UniqueID holder is move-only for exactly that reason), so a key can never
  // alias a different material. Thread-safety: factories can be invoked
  // concurrently, so every touch of the list happens under the mutex; the
  // parse itself runs unlocked, and a concurrent miss on the same key at worst
  // parses the same immutable file twice (the loser adopts the winner's
  // entry). Entries are shared_ptr-owned, so a pointer handed out stays valid
  // regardless of later eviction, and the MRU bound keeps retained memory
  // finite after the associated materials are gone.
  using PackListPtr = std::shared_ptr<const std::vector<PackData>>;

  PackListPtr getPacksForInfo( const NC::Info& info )
  {
    static std::mutex cache_mutex;
    static std::vector<std::pair<std::uint64_t,PackListPtr>> cache;
    constexpr std::size_t cache_capacity = 4;
    const std::uint64_t key = info.getUniqueID().value;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      for ( auto it = cache.begin(); it != cache.end(); ++it ) {
        if ( it->first == key ) {
          auto entry = *it;
          cache.erase( it );
          cache.push_back( entry );   // refresh MRU position
          return entry.second;
        }
      }
    }
    auto parsed = std::make_shared<const std::vector<PackData>>(
      loadPacks( parseCustomSectionForPackPaths( info ) ) );
    std::lock_guard<std::mutex> lock(cache_mutex);
    for ( const auto& entry : cache ) {
      if ( entry.first == key )
        return entry.second;
    }
    if ( cache.size() >= cache_capacity )
      cache.erase( cache.begin() );
    cache.emplace_back( key, parsed );
    return parsed;
  }

  bool hasUniformReferenceMSD( const NC::Info& info )
  {
    if ( !info.hasHKLInfo() || !info.hasStructureInfo() || !info.hasAtomMSD() )
      return false;
    const auto& atomInfos = info.getAtomInfos();
    if ( atomInfos.empty() )
      return false;
    const double referenceMSD = atomInfos.front().msd().value();
    for ( const auto& atomInfo : atomInfos ) {
      if ( !atomInfo.msd().has_value() )
        return false;
      if ( std::abs( atomInfo.msd().value() - referenceMSD ) > 1.0e-14 )
        return false;
    }
    return true;
  }

  double uniformReferenceMSD( const NC::Info& info )
  {
    if ( !hasUniformReferenceMSD( info ) )
      NCRYSTAL_THROW(BadInput,"Pack-owned coherent elastic currently requires "
                     "a crystalline material with uniform isotropic reference MSD");
    return info.getAtomInfos().front().msd().value();
  }

  struct TensorSite {
    NC::Vector pos;
    double coherentScatLen = 0.0;
    double incoherentXS = 0.0;
    std::array<double,9> tensor{};
  };

  double tensorDebyeWallerExponent( const std::array<double,9>& tensor,
                                    const NC::Vector& gvec )
  {
    const double gx = gvec[0];
    const double gy = gvec[1];
    const double gz = gvec[2];
    const double ugx = tensor[0] * gx + tensor[1] * gy + tensor[2] * gz;
    const double ugy = tensor[3] * gx + tensor[4] * gy + tensor[5] * gz;
    const double ugz = tensor[6] * gx + tensor[7] * gy + tensor[8] * gz;
    return 0.5 * ( gx * ugx + gy * ugy + gz * ugz );
  }

  double wrappedFractionalDelta( double a, double b )
  {
    auto delta = std::abs( a - b );
    delta -= std::floor(delta);
    return delta > 0.5 ? 1.0 - delta : delta;
  }

  bool sameFractionalPosition( const NC::Vector& lhs,
                               const NC::Vector& rhs )
  {
    constexpr double tol = 1.0e-6;
    return wrappedFractionalDelta(lhs[0],rhs[0]) < tol
           && wrappedFractionalDelta(lhs[1],rhs[1]) < tol
           && wrappedFractionalDelta(lhs[2],rhs[2]) < tol;
  }

  std::string elementSymbolForAtomInfo( const NC::AtomInfo& atomInfo )
  {
    const auto& atomData = atomInfo.atomData();
    if ( !atomData.isElement() )
      NCRYSTAL_THROW(BadInput,"Pack-owned coherent elastic tensor path requires "
                     "NCrystal atom data with element symbols for site matching");
    return atomData.elementName();
  }

  struct PackTensorSite {
    std::string symbol;
    NC::Vector pos;
    std::array<double,9> tensor{};
    double cohScatLen = 0.0;   // sqrt(barn), from the pack (IRMA config b_coh)
    double sigmaInc = 0.0;     // barn, from the pack (IRMA config sigma_inc)
    bool used = false;
  };

  std::vector<PackTensorSite> tensorPackSites( const PackData& pack )
  {
    std::vector<PackTensorSite> sites;
    const auto nsites = pack.elasticUTensors.size() / 9;
    sites.reserve(nsites);
    for ( std::size_t isite = 0; isite < nsites; ++isite ) {
      std::array<double,9> tensor{};
      for ( unsigned i = 0; i < 9; ++i )
        tensor[i] = pack.elasticUTensors[9*isite+i];
      sites.push_back( { pack.elasticUSymbols.at(isite),
                         NC::Vector( pack.elasticUFracPositions[3*isite],
                                     pack.elasticUFracPositions[3*isite+1],
                                     pack.elasticUFracPositions[3*isite+2] ),
                         tensor,
                         pack.elasticUCohScatLen.at(isite),
                         pack.elasticUIncohXS.at(isite),
                         false } );
    }
    return sites;
  }

  std::vector<TensorSite> tensorSitesFromInfo( const NC::Info& info,
                                               const PackData& pack )
  {
    auto packSites = tensorPackSites( pack );
    std::vector<TensorSite> sites;
    // The coherent scattering length + incoherent xs come from the PACK (the IRMA
    // config's b_coh / sigma_inc), NOT atomInfo.atomData(): the plugin must honor
    // exactly what IRMA specified, which can differ from NCrystal's atom DB. NCrystal
    // sites are still matched to pack sites by (symbol, fractional position).
    for ( const auto& atomInfo : info.getAtomInfos() ) {
      const auto symbol = elementSymbolForAtomInfo( atomInfo );
      for ( const auto& rawPos : atomInfo.unitCellPositions() ) {
        const auto pos = rawPos.as<NC::Vector>();
        std::size_t matchIndex = packSites.size();
        for ( std::size_t i = 0; i < packSites.size(); ++i ) {
          if ( packSites[i].used )
            continue;
          if ( packSites[i].symbol == symbol
               && sameFractionalPosition( packSites[i].pos, pos ) ) {
            matchIndex = i;
            break;
          }
        }
        if ( matchIndex == packSites.size() )
          NCRYSTAL_THROW2(BadInput,"IRMA pack elastic_u_tensors_a2 has no "
                          "tensor matching NCrystal site "<<symbol<<" at "
                          "fractional position ("<<pos[0]<<", "<<pos[1]
                          <<", "<<pos[2]<<")");
        packSites[matchIndex].used = true;
        sites.push_back( { pos, packSites[matchIndex].cohScatLen,
                           packSites[matchIndex].sigmaInc,
                           packSites[matchIndex].tensor } );
      }
    }
    if ( sites.empty() )
      NCRYSTAL_THROW(BadInput,"Pack-owned coherent elastic tensor path requires atom positions");
    for ( const auto& packSite : packSites ) {
      if ( !packSite.used )
        NCRYSTAL_THROW2(BadInput,"IRMA pack elastic_u_tensors_a2 contains "
                        "unused tensor for "<<packSite.symbol
                        <<" at fractional position ("<<packSite.pos[0]<<", "
                        <<packSite.pos[1]<<", "<<packSite.pos[2]<<")");
    }
    return sites;
  }

  double tensorMeanSquareDisplacement( const std::array<double,9>& tensor )
  {
    return ( tensor[0] + tensor[4] + tensor[8] ) / 3.0;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Directional incoherent elastic (incoherent_elastic_mode = directional).
  //
  // The powder incoherent-elastic cross section of one site carries the
  // orientation average of the Debye-Waller exponential,
  //
  //   f(t) = < exp(-t uhat.U.uhat) >_uhat,   t = Q^2,
  //
  // which this plugin represents as a finite mixture of exponentials
  //
  //   f(t) = sum_j w_j exp(-p_j t),  sum_j w_j = 1,  p_j >= 0,
  //
  // built from the eigenvalues u1<=u2<=u3 of U [Ang^2]:
  //   * isotropic  (u3-u1 <= tol): one component, p = tr(U)/3       (exact)
  //   * uniaxial   (a degenerate pair): 24-point Gauss-Legendre in c^2 along
  //     the unique axis, p_j = uperp + (upar - uperp) c_j^2
  //   * triaxial: the 24x24 product rule over one octant.
  //
  // The same branch constants and quadrature literals live in
  // irma/core/incoherent_dw.py (which additionally has closed erf/Dawson
  // forms for the uniaxial cases); the Python reference oracle checks this
  // implementation against those closed forms. The mixture is exact for the
  // isotropic branch and quadrature-accurate otherwise: the GL rule
  // under-resolves f only once the factor has decayed to ~1e-30 of its Q=0
  // value (Q of a few hundred 1/Ang for crystalline U tensors), far beyond
  // any physical contribution to the cross section. The mixture makes both
  // the cross section and the mu-sampling fully analytic:
  //
  //   sigma(E)  = sum_i sigma_i sum_j w_ij (1-exp(-p_ij T))/(p_ij T),
  //   T = 4 k^2 = Q^2_max,   mu = 1 - t/(2 k^2),
  //
  // with t sampled from the component-weighted truncated exponentials, so the
  // sampler is exactly consistent with the cross section by construction.
  //
  // Component selection (review PH-4): the rejection sampler offers two
  // proposals -- q(i) ~ w_i (t uniform on [0,T]) with envelope mass
  // T sum w_i, and q(i) ~ w_i/p_i (t ~ Exp(p_i), accept t<=T) with envelope
  // mass sum w_i/p_i. The branch is chosen by comparing those ACTUAL
  // envelope masses (the smaller mass = the higher mean acceptance
  // G(T)/mass). An arithmetic-mean criterion (pbar*T >= 1) used to pick the
  // wrong branch for wide p-spreads: for p = {1e-7, 0.1} Ang^2 at T = 20 it
  // chose the w/p proposal with mean acceptance 3e-6, exhausted the attempt
  // cap for ~75% of events, and silently fell back to a uniform-mu law.
  // Exhaustion now falls back to an EXACT O(n) scan over the per-component
  // integrals (never an approximate law), so every accepted tensor state is
  // sampled from the requested distribution regardless of conditioning.
  ////////////////////////////////////////////////////////////////////////////

  // 24-point Gauss-Legendre nodes/weights on [0,1] -- the SAME 17-digit
  // literals as irma.core.incoherent_dw.GL24_NODES01/GL24_WEIGHTS01.
  constexpr std::array<double,24> gl24_nodes01 = {
    0.0024063900014893447, 0.012635722014345263, 0.030862723998633601,
    0.056792236497799464, 0.089999007013048526, 0.12993790421072282,
    0.17595317403151223, 0.22728926430558022, 0.28310324618697746,
    0.3424786601519183, 0.40444056626319186, 0.46797155356869719,
    0.53202844643130276, 0.5955594337368082, 0.6575213398480817,
    0.71689675381302254, 0.77271073569441984, 0.82404682596848777,
    0.87006209578927718, 0.91000099298695147, 0.94320776350220048,
    0.96913727600136634, 0.98736427798565474, 0.99759360999851066 };
  constexpr std::array<double,24> gl24_weights01 = {
    0.0061706148999933919, 0.014265694314466842, 0.02213871940870971,
    0.029649292457718267, 0.036673240705540136, 0.043095080765976713,
    0.048809326052057046, 0.053722135057982866, 0.057752834026862855,
    0.060835236463901758, 0.062918728173414248, 0.063969097673376149,
    0.063969097673376149, 0.062918728173414248, 0.060835236463901758,
    0.057752834026862855, 0.053722135057982866, 0.048809326052057046,
    0.043095080765976713, 0.036673240705540136, 0.029649292457718267,
    0.02213871940870971, 0.014265694314466842, 0.0061706148999933919 };

  // Same relative degeneracy tolerance as irma.core.incoherent_dw, so the
  // Python and C++ sides always classify a tensor into the same branch.
  constexpr double dw_degeneracy_rtol = 1.0e-8;

  std::array<double,3> symEigenvaluesAscending( const std::array<double,9>& m )
  {
    // Trigonometric closed form for a symmetric 3x3 (the tensors were already
    // validated symmetric PSD at parse time; rounding negatives clip to 0).
    const double a = m[0], b = m[4], c = m[8];
    const double d = 0.5 * ( m[1] + m[3] );
    const double e = 0.5 * ( m[5] + m[7] );
    const double f = 0.5 * ( m[2] + m[6] );
    std::array<double,3> eig;
    const double p1 = d*d + e*e + f*f;
    if ( p1 == 0.0 ) {
      eig = { a, b, c };
    } else {
      const double q = ( a + b + c ) / 3.0;
      const double p2 = (a-q)*(a-q) + (b-q)*(b-q) + (c-q)*(c-q) + 2.0*p1;
      const double p = std::sqrt( p2 / 6.0 );
      const double invp = 1.0 / p;
      // r = det( (M - q I)/p ) / 2
      const double aa = (a-q)*invp, bb = (b-q)*invp, cc = (c-q)*invp;
      const double dd = d*invp, ee = e*invp, ff = f*invp;
      double r = 0.5 * ( aa*(bb*cc - ee*ee) - dd*(dd*cc - ee*ff)
                         + ff*(dd*ee - bb*ff) );
      r = NC::ncclamp( r, -1.0, 1.0 );
      const double phi = std::acos( r ) / 3.0;
      constexpr double twopi3 = 2.0943951023931953;  // 2*pi/3
      const double e1 = q + 2.0 * p * std::cos( phi );
      const double e3 = q + 2.0 * p * std::cos( phi + 2.0 * twopi3 );
      eig = { e3, 3.0*q - e1 - e3, e1 };
    }
    std::sort( eig.begin(), eig.end() );
    for ( auto& v : eig )
      v = std::max( 0.0, v );
    return eig;
  }

  // One site's mixture-of-exponentials representation of f(t).
  void appendDirectionalComponents( const std::array<double,3>& eig,
                                    double sigma_scaled,
                                    NC::VectD& weights, NC::VectD& pvals )
  {
    const double u1 = eig[0], u2 = eig[1], u3 = eig[2];
    if ( !(sigma_scaled > 0.0) )
      return;
    if ( u3 <= 0.0 ) {
      weights.push_back( sigma_scaled );
      pvals.push_back( 0.0 );
      return;
    }
    const double tol = dw_degeneracy_rtol * u3;
    if ( u3 - u1 <= tol ) {
      weights.push_back( sigma_scaled );
      pvals.push_back( ( u1 + u2 + u3 ) / 3.0 );
      return;
    }
    const bool oblate = ( u2 - u1 <= tol );   // unique LARGE axis (e.g. graphite c)
    const bool prolate = ( u3 - u2 <= tol );  // unique SMALL axis
    if ( oblate || prolate ) {
      const double uperp = oblate ? 0.5*(u1+u2) : 0.5*(u2+u3);
      const double upar = oblate ? u3 : u1;
      for ( std::size_t j = 0; j < gl24_nodes01.size(); ++j ) {
        const double csq = gl24_nodes01[j] * gl24_nodes01[j];
        weights.push_back( sigma_scaled * gl24_weights01[j] );
        pvals.push_back( uperp + ( upar - uperp ) * csq );
      }
      return;
    }
    constexpr double halfpi = 1.5707963267948966;
    for ( std::size_t j = 0; j < gl24_nodes01.size(); ++j ) {
      const double csq = gl24_nodes01[j] * gl24_nodes01[j];
      for ( std::size_t k = 0; k < gl24_nodes01.size(); ++k ) {
        const double phi = halfpi * gl24_nodes01[k];
        const double cos2 = std::cos( phi ) * std::cos( phi );
        const double p = u3 * csq
          + ( 1.0 - csq ) * ( u1 * cos2 + u2 * ( 1.0 - cos2 ) );
        weights.push_back( sigma_scaled * gl24_weights01[j] * gl24_weights01[k] );
        pvals.push_back( p );
      }
    }
  }

  // Directional incoherent-elastic mixture, O(log N) per call.
  //
  // The mixture can have thousands of components (each triaxial site
  // contributes a full GL24xGL24 octant), but the component rates p_i are
  // fixed; only T = Q^2_max = 4k^2 varies with neutron energy. Instead of
  // re-evaluating all component integrals b_i(T) = w_i(1-exp(-p_i T))/p_i
  // per call:
  //   - the cross section uses a dense log-T table of G(T) = sum_i b_i(T)
  //     built once in the constructor (relative interpolation error
  //     ~1e-5; exact analytic limits below/above the tabulated range),
  //   - the component pick uses an EXACT two-proposal rejection sampler:
  //     propose from the fixed cumulative of w_i/p_i (saturated weights,
  //     efficient at large p*T) or of w_i (small-p*T limit), then accept
  //     with (1-exp(-p_i T)) resp. (1-exp(-p_i T))/(p_i T), both in
  //     (0,1]. Accepted picks follow b_i(T) exactly; acceptance stays
  //     above ~0.6 with the branch switch at pbar*T = 1.
  class DirectionalElIncScatter final : public NC::ProcImpl::ScatterIsotropicMat {
  public:
    const char * name() const noexcept override
    {
      return "IRMADirectionalElIncScatter";
    }

    DirectionalElIncScatter( NC::VectD&& weights, NC::VectD&& pvals )
    {
      nc_assert_always( weights.size() == pvals.size() && !weights.empty() );
      m_wsum = 0.0;
      double swp = 0.0, swp2 = 0.0;
      for ( std::size_t i = 0; i < weights.size(); ++i ) {
        const double w = weights[i], p = pvals[i];
        m_wsum += w;
        if ( !(w > 0.0) )
          continue;
        if ( p > 0.0 ) {
          m_p.push_back( p );
          m_cum_w.push_back( ( m_cum_w.empty() ? 0.0 : m_cum_w.back() ) + w );
          m_cum_W.push_back( ( m_cum_W.empty() ? 0.0 : m_cum_W.back() ) + w / p );
          swp += w * p;
          swp2 += w * p * p;
        } else {
          m_w0sum += w;   // flat component: b = w*T, mu uniform
        }
      }
      m_swp = swp;
      m_swp2 = swp2;
      if ( m_p.empty() )
        return;
      m_wpsum = m_cum_w.back();
      m_Wpsum = m_cum_W.back();
      // log-T table of G(T) between the exact small-T and saturated limits
      double pmin = m_p.front(), pmax = m_p.front();
      for ( const auto p : m_p ) {
        pmin = std::min( pmin, p );
        pmax = std::max( pmax, p );
      }
      m_lnTlo = std::log( 1e-4 / pmax );
      const double lnThi = std::log( 50.0 / pmin );
      m_dlnT = ( lnThi - m_lnTlo ) / ( n_gtab - 1 );
      m_lnG.reserve( n_gtab );
      for ( std::size_t j = 0; j < n_gtab; ++j ) {
        const double T = std::exp( m_lnTlo + j * m_dlnT );
        double g = 0.0;
        for ( std::size_t i = 0; i < m_p.size(); ++i ) {
          const double dw = m_cum_w[i] - ( i ? m_cum_w[i-1] : 0.0 );
          g += dw * componentIntegral( m_p[i], T );
        }
        m_lnG.push_back( std::log( g ) );
      }
    }

    NC::CrossSect crossSectionIsotropic( NC::CachePtr&,
                                         NC::NeutronEnergy ekin ) const override
    {
      const double T = 4.0 * NC::ekin2ksq( ekin.dbl() );  // Q^2_max [1/Ang^2]
      if ( !(T > 0.0) )
        return NC::CrossSect{ m_wsum };
      return NC::CrossSect{ m_w0sum + gOfT( T ) / T };
    }

    NC::ScatterOutcomeIsotropic
    sampleScatterIsotropic( NC::CachePtr&, NC::RNG& rng,
                            NC::NeutronEnergy ekin ) const override
    {
      const double ksq = NC::ekin2ksq( ekin.dbl() );
      const double T = 4.0 * ksq;
      if ( !(T > 0.0) )
        return NC::ScatterOutcomeIsotropic::noScat( ekin );
      const double gp = gOfT( T );
      const double btot = m_w0sum * T + gp;
      if ( !(btot > 0.0) )
        return NC::ScatterOutcomeIsotropic::noScat( ekin );
      double p = 0.0;
      // Branch guard (review CPP-7): NC::RNG::generate() is contractually in
      // (0,1] and CAN return exactly 1.0. With m_p empty, gOfT()==0 makes
      // btot == m_w0sum*T exactly, so a `>=` test at rng==1.0 entered this
      // branch and ran cum.back()/m_p.size()-1 on EMPTY vectors (UB). The
      // explicit empty() guard removes that path outright, and `>` keeps the
      // p>0 branch probability at gp/btot for rng uniform on (0,1] (the
      // boundary has measure zero; when m_w0sum==0, rng*btot>0 always holds).
      if ( !m_p.empty() && rng.generate() * btot > m_w0sum * T ) {
        // p>0 set: exact rejection with the smaller-envelope proposal
        // (review PH-4: compare the ACTUAL envelope masses, sum(w/p) vs
        // T*sum(w) -- an arithmetic pbar*T criterion picked the wrong
        // branch for wide p-spreads and starved the acceptance).
        const bool high = ( m_Wpsum <= m_wpsum * T );
        const NC::VectD& cum = high ? m_cum_W : m_cum_w;
        const double cmax = cum.back();
        bool accepted = false;
        for ( unsigned it = 0; it < 1000; ++it ) {
          const double r = rng.generate() * cmax;
          const std::size_t i = std::min<std::size_t>(
              m_p.size() - 1,
              static_cast<std::size_t>(
                  std::lower_bound( cum.begin(), cum.end(), r ) - cum.begin() ) );
          const double pT = m_p[i] * T;
          const double one_m_exp = ( pT < 1e-12 ) ? pT : -std::expm1( -pT );
          const double acc = high ? one_m_exp
                                  : ( pT < 1e-12 ? 1.0 : one_m_exp / pT );
          if ( rng.generate() < acc ) {
            p = m_p[i];
            accepted = true;
            break;
          }
        }
        if ( !accepted ) {
          // Exact O(n) fallback (review PH-4): select the component from
          // the exact per-component integrals b_i = dw_i (1-e^{-p_i T})/p_i.
          // The old code left p = 0 here, which the truncated-exponential
          // step below reads as the FLAT law -- a silent wrong-distribution
          // fallback for poorly conditioned mixtures.
          double btot_exact = 0.0;
          for ( std::size_t i = 0; i < m_p.size(); ++i ) {
            const double dw = m_cum_w[i] - ( i ? m_cum_w[i-1] : 0.0 );
            btot_exact += dw * componentIntegral( m_p[i], T );
          }
          double r = rng.generate() * btot_exact;
          p = m_p.back();
          for ( std::size_t i = 0; i < m_p.size(); ++i ) {
            const double dw = m_cum_w[i] - ( i ? m_cum_w[i-1] : 0.0 );
            r -= dw * componentIntegral( m_p[i], T );
            if ( r <= 0.0 ) {
              p = m_p[i];
              break;
            }
          }
        }
      }
      // Truncated exponential on [0,T]: t = -ln(1 - u(1-exp(-pT)))/p.
      const double u = rng.generate();
      double t;
      if ( p * T < 1e-12 ) {
        t = u * T;
      } else {
        t = -std::log1p( u * std::expm1( -p * T ) ) / p;
      }
      const double mu = NC::ncclamp( 1.0 - t / ( 2.0 * ksq ), -1.0, 1.0 );
      return { ekin, NC::CosineScatAngle{ mu } };
    }

  private:
    // G(T) = sum_{p_i>0} w_i (1-exp(-p_i T))/p_i via the log-log table,
    // with exact limits outside the tabulated range.
    double gOfT( double T ) const
    {
      if ( m_p.empty() )
        return 0.0;
      const double lnT = std::log( T );
      if ( lnT <= m_lnTlo )
        return T * ( m_wpsum - 0.5 * T * m_swp + T * T * m_swp2 / 6.0 );
      const double f = ( lnT - m_lnTlo ) / m_dlnT;
      if ( f >= n_gtab - 1 )
        return m_Wpsum;
      const std::size_t j = static_cast<std::size_t>( f );
      const double frac = f - j;
      return std::exp( m_lnG[j] * ( 1.0 - frac ) + m_lnG[j+1] * frac );
    }

    // integral_0^T exp(-p t) dt = (1 - exp(-p T))/p, stable for p T -> 0.
    static double componentIntegral( double p, double T )
    {
      const double pT = p * T;
      if ( pT < 1e-12 )
        return T * ( 1.0 - 0.5 * pT );
      return -std::expm1( -pT ) / p;
    }

    static constexpr std::size_t n_gtab = 800;
    NC::VectD m_p;        // p>0 components only
    NC::VectD m_cum_w;    // cumulative w_i        (low-pT proposal)
    NC::VectD m_cum_W;    // cumulative w_i/p_i    (high-pT proposal)
    NC::VectD m_lnG;      // ln G on the log-T grid
    double m_lnTlo = 0.0, m_dlnT = 1.0;
    double m_wsum = 0.0, m_w0sum = 0.0;
    double m_wpsum = 0.0, m_Wpsum = 0.0;
    double m_swp = 0.0, m_swp2 = 0.0;
  };

  NC::ProcImpl::ProcPtr
  createDirectionalIncoherentElasticProcess( const NC::Info& info,
                                             const PackData& pack )
  {
    if ( !info.hasAtomInfo() )
      NCRYSTAL_THROW(BadInput,"Pack-owned incoherent elastic tensor path "
                     "requires atom positions");
    const auto sites = tensorSitesFromInfo( info, pack );
    const double siteScale = pack.elasticScale / double( sites.size() );
    NC::VectD weights, pvals;
    for ( const auto& site : sites )
      appendDirectionalComponents( symEigenvaluesAscending( site.tensor ),
                                   site.incoherentXS * siteScale,
                                   weights, pvals );
    if ( weights.empty() ) {
      // All sites have zero incoherent cross section: an exactly-zero
      // component keeps the process well-formed (and the composition happy).
      weights.push_back( 0.0 );
      pvals.push_back( 0.0 );
    }
    return NC::makeSO<DirectionalElIncScatter>( std::move(weights),
                                                std::move(pvals) );
  }

  double tensorFSquaredForHKL( const std::vector<TensorSite>& sites,
                               const NC::RotMatrix& reciprocalLattice,
                               const NC::HKL& hkl )
  {
    constexpr double twopi = 6.283185307179586476925286766559;
    const NC::Vector hklVector( hkl.h, hkl.k, hkl.l );
    const auto gvec = reciprocalLattice * hklVector;
    double real = 0.0;
    double imag = 0.0;
    for ( std::size_t isite = 0; isite < sites.size(); ++isite ) {
      const auto& site = sites[isite];
      const double dw = std::exp( -tensorDebyeWallerExponent( site.tensor, gvec ) );
      const double phase = twopi * ( hkl.h * site.pos[0]
                                     + hkl.k * site.pos[1]
                                     + hkl.l * site.pos[2] );
      const double factor = site.coherentScatLen * dw;
      real += factor * std::cos( phase );
      imag += factor * std::sin( phase );
    }
    return real * real + imag * imag;
  }

  NC::ProcImpl::ProcPtr createTensorCoherentElasticProcess( const NC::Info& info,
                                                            const PackData& pack )
  {
    if ( !info.hasHKLInfo() || !info.hasStructureInfo() || !info.hasAtomInfo() )
      NCRYSTAL_THROW(BadInput,"Pack-owned coherent elastic tensor path requires "
                     "crystalline structure, atom positions, and HKL data");
    NC::ExpandHKLHelper hklExpander( info );
    if ( !hklExpander.canExpand( info.hklInfoType() ) )
      NCRYSTAL_THROW(BadInput,"Pack-owned coherent elastic tensor path requires "
                     "expandable HKL entries");

    const auto sites = tensorSitesFromInfo( info, pack );
    const auto reciprocalLattice = NC::getReciprocalLatticeRot( info.getStructureInfo() );
    NC::PowderBragg::VectDFM planes;
    planes.reserve( info.hklList().size() );
    for ( const auto& hklInfo : info.hklList() ) {
      if ( hklInfo.dspacing <= 0.0 )
        NCRYSTAL_THROW(CalcError,"HKL entry has non-positive dspacing");
      auto expandedHKLs = hklExpander.expand( hklInfo );
      double fsqSum = 0.0;
      unsigned nExpanded = 0;
      for ( const auto& hkl : expandedHKLs ) {
        fsqSum += tensorFSquaredForHKL( sites,
                                        reciprocalLattice,
                                        hkl );
        ++nExpanded;
      }
      if ( nExpanded == 0 )
        continue;
      const double meanFSq = fsqSum / nExpanded;
      planes.emplace_back( hklInfo.dspacing, meanFSq * hklInfo.multiplicity );
    }
    return NC::makeSO<NC::PowderBragg>( info.getStructureInfo(), std::move(planes) );
  }

  NC::ProcImpl::ProcPtr createCoherentElasticProcess( const NC::Info& info,
                                                      const PackData& pack )
  {
    if ( !pack.elasticUTensors.empty() )
      return createTensorCoherentElasticProcess( info, pack );

    constexpr double twopi = 6.283185307179586476925286766559;
    const double referenceMSD = uniformReferenceMSD( info );
    const double deltaMSD = pack.elasticMSD - referenceMSD;
    NC::PowderBragg::VectDFM planes;
    planes.reserve( info.hklList().size() );
    for ( const auto& hkl : info.hklList() ) {
      if ( hkl.fsquared < 0.0 )
        NCRYSTAL_THROW(CalcError,"HKL entry has negative fsquared");
      if ( hkl.dspacing <= 0.0 )
        NCRYSTAL_THROW(CalcError,"HKL entry has non-positive dspacing");
      const double gmag = twopi / hkl.dspacing;
      const double dwRatio = std::exp( -gmag * gmag * deltaMSD );
      planes.emplace_back( hkl.dspacing, hkl.fsquared * hkl.multiplicity * dwRatio );
    }
    return NC::makeSO<NC::PowderBragg>( info.getStructureInfo(), std::move(planes) );
  }

  NC::ProcImpl::ProcPtr createTensorIncoherentElasticProcess( const NC::Info& info,
                                                              const PackData& pack )
  {
    if ( pack.incoherentElasticMode == "directional" )
      return createDirectionalIncoherentElasticProcess( info, pack );
    if ( !info.hasAtomInfo() )
      NCRYSTAL_THROW(BadInput,"Pack-owned incoherent elastic tensor path requires atom positions");
    const auto sites = tensorSitesFromInfo( info, pack );
    const double siteScale = pack.elasticScale / double(sites.size());
    NC::VectD msd;
    NC::VectD xs;
    NC::VectD scale;
    msd.reserve( sites.size() );
    xs.reserve( sites.size() );
    scale.reserve( sites.size() );
    for ( const auto& site : sites ) {
      msd.push_back( tensorMeanSquareDisplacement( site.tensor ) );
      xs.push_back( site.incoherentXS );
      scale.push_back( siteScale );
    }
    return NC::makeSO<NC::ElIncScatter>( msd, xs, scale );
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
        NCRYSTAL_THROW2(BadInput,"IRMA pack was baked at "<<pack.temperatureK
                        <<" K but the requested material temperature is "<<reqT
                        <<" K. Load the material at the pack temperature (e.g. "
                        "'<material>.ncmat;temp="<<pack.temperatureK<<"'); the IRMA "
                        "plugin samples the precomputed law and does not "
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
      if ( includeCoherentElastic && pack.providesCoherentElastic() )
        components.push_back( { 1.0, createCoherentElasticProcess( info, pack ) } );
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
        if ( !pack.elasticUTensors.empty() ) {
          components.push_back( { 1.0, createTensorIncoherentElasticProcess( info, pack ) } );
          continue;
        }
        components.push_back(
          { 1.0, NC::makeSO<NC::ElIncScatter>(
              NC::VectD{ pack.elasticMSD },
              NC::VectD{ pack.elasticIncohXS },
              NC::VectD{ pack.elasticScale } ) } );
      }
    }
    if ( components.empty() )
      NCRYSTAL_THROW(BadInput,"IRMA plugin was requested but no plugin "
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
  for ( const auto& pack : *getPacksForInfo( info ) ) {
    if ( pack.providesIncoherentElastic() )
      return true;
  }
  return false;
}

bool NCP::PhysicsModel::providesCoherentElastic( const NC::Info& info )
{
  const bool uniformReferenceMSD = hasUniformReferenceMSD( info );
  for ( const auto& pack : *getPacksForInfo( info ) ) {
    if ( pack.providesCoherentElastic()
         && ( pack.providesTensorCoherentElastic() || uniformReferenceMSD ) )
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
  // createPluginProcess consumes the pack arrays (std::move into ScatKnlData),
  // so it gets a mutable COPY of the cached parse -- copying the numeric
  // vectors is far cheaper than re-tokenising the pack text.
  : m_proc( createPluginProcess( info,
                                 std::vector<PackData>( *getPacksForInfo( info ) ),
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
