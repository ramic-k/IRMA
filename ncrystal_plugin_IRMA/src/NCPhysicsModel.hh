#ifndef NCPlugin_PhysicsModel_hh
#define NCPlugin_PhysicsModel_hh

#include "NCrystal/NCPluginBoilerplate.hh"
#include "NCrystal/interfaces/NCProcImpl.hh"

#include <memory>

namespace NCPluginNamespace {

  class PhysicsModel final : public NC::MoveOnly {
  public:
    static bool isApplicable( const NC::Info& );
    static bool providesIncoherentElastic( const NC::Info& );
    static bool providesCoherentElastic( const NC::Info& );
    static PhysicsModel createFromInfo( const NC::Info&, bool includeInelastic,
                                        bool includeIncoherentElastic,
                                        bool includeCoherentElastic );

    explicit PhysicsModel( const NC::Info&, bool includeInelastic,
                           bool includeIncoherentElastic,
                           bool includeCoherentElastic );

    double calcCrossSection( NC::CachePtr&, double neutron_ekin ) const;

    struct ScatEvent { double ekin_final, mu; };
    ScatEvent sampleScatteringEvent( NC::CachePtr&, NC::RNG&, double neutron_ekin ) const;

  private:
    // Hold the underlying process directly, NOT an NC::Scatter (which owns its
    // own RNG): NCrystal hands us a per-call CachePtr + RNG for thread-safety
    // and reproducibility, and we thread them straight through to the process.
    NC::ProcImpl::ProcPtr m_proc;
  };

}

#endif
