#include "NCPluginFactory.hh"
#include "NCPhysicsModel.hh"

namespace NCPluginNamespace {

  class PluginScatter final : public NC::ProcImpl::ScatterIsotropicMat {
  public:
    const char * name() const noexcept override { return NCPLUGIN_NAME_CSTR "Model"; }

    explicit PluginScatter( PhysicsModel&& pm ) : m_pm(std::move(pm)) {}

    NC::CrossSect
    crossSectionIsotropic( NC::CachePtr& cache,
                           NC::NeutronEnergy ekin ) const override
    {
      return NC::CrossSect{ m_pm.calcCrossSection( cache, ekin.dbl() ) };
    }

    NC::ScatterOutcomeIsotropic
    sampleScatterIsotropic( NC::CachePtr& cache,
                            NC::RNG& rng,
                            NC::NeutronEnergy ekin ) const override
    {
      auto outcome = m_pm.sampleScatteringEvent( cache, rng, ekin.dbl() );
      return { NC::NeutronEnergy{outcome.ekin_final},
               NC::CosineScatAngle{outcome.mu} };
    }

  private:
    PhysicsModel m_pm;
  };

}

namespace {

  bool inelasticRequested( const NC::FactImpl::ScatterRequest& cfg )
  {
    const auto& inelas = cfg.get_inelas();
    return !( inelas=="none"
              || inelas=="0"
              || inelas=="false"
              || inelas=="sterile" );
  }

}

const char * NCP::PluginFactory::name() const noexcept
{
  return NCPLUGIN_NAME_CSTR "Factory";
}

NC::Priority
NCP::PluginFactory::query( const NC::FactImpl::ScatterRequest& cfg ) const
{
  if ( !PhysicsModel::isApplicable( cfg.info() ) )
    return NC::Priority::Unable;

  if ( inelasticRequested( cfg ) )
    return NC::Priority{999};

  if ( cfg.get_incoh_elas() && PhysicsModel::providesIncoherentElastic( cfg.info() ) )
    return NC::Priority{999};

  if ( cfg.get_coh_elas() && PhysicsModel::providesCoherentElastic( cfg.info() ) )
    return NC::Priority{999};

  return NC::Priority::Unable;
}

NC::ProcImpl::ProcPtr
NCP::PluginFactory::produce( const NC::FactImpl::ScatterRequest& cfg ) const
{
  const bool includeInelastic = inelasticRequested( cfg );
  const bool includeIncohElastic = (
    cfg.get_incoh_elas() && PhysicsModel::providesIncoherentElastic( cfg.info() ) );
  const bool includeCohElastic = (
    cfg.get_coh_elas() && PhysicsModel::providesCoherentElastic( cfg.info() ) );

  auto sc_ourmodel = NC::makeSO<PluginScatter>(
    PhysicsModel::createFromInfo( cfg.info(),
                                  includeInelastic,
                                  includeIncohElastic,
                                  includeCohElastic ) );

  auto cfg_std = includeInelastic ? cfg.modified("inelas=0") : cfg;
  if ( includeIncohElastic )
    cfg_std = cfg_std.modified("incoh_elas=0");
  if ( includeCohElastic )
    cfg_std = cfg_std.modified("coh_elas=0");

  auto sc_std = globalCreateScatter( cfg_std );

  return combineProcs( sc_std, sc_ourmodel );
}
