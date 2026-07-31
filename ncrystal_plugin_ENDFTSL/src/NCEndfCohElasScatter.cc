#include "NCEndfCohElasScatter.hh"

#include "NCrystal/internal/utils/NCMath.hh"

#include <algorithm>

namespace NCP = NCPluginNamespace;

NCP::EndfCohElasScatter::EndfCohElasScatter( NC::VectD edges_eV, NC::VectD cumS )
  : m_edges( std::move(edges_eV) ),
    m_cumS( std::move(cumS) )
{
  nc_assert_always( m_edges.size() == m_cumS.size() && !m_edges.empty() );
}

NC::CrossSect
NCP::EndfCohElasScatter::crossSectionIsotropic( NC::CachePtr&,
                                                NC::NeutronEnergy ekin ) const
{
  const double E = ekin.dbl();
  // number of edges with E_i <= E (histogram / staircase, INT=1)
  const std::size_t n =
    std::upper_bound( m_edges.begin(), m_edges.end(), E ) - m_edges.begin();
  if ( n == 0 )
    return NC::CrossSect{ 0.0 };          // below the first Bragg edge
  return NC::CrossSect{ m_cumS[n-1] / E };  // sigma_coh = S(E)/E
}

NC::ScatterOutcomeIsotropic
NCP::EndfCohElasScatter::sampleScatterIsotropic( NC::CachePtr&,
                                                 NC::RNG& rng,
                                                 NC::NeutronEnergy ekin ) const
{
  const double E = ekin.dbl();
  const std::size_t n =
    std::upper_bound( m_edges.begin(), m_edges.end(), E ) - m_edges.begin();
  if ( n == 0 )
    return { ekin, NC::CosineScatAngle{ 1.0 } };   // xs=0 here -> forward, never reached

  const double Stot = m_cumS[n-1];
  const double u = rng.generate() * Stot;
  // pick edge i by its cumulative jump weight j_i = cumS[i]-cumS[i-1]:
  // smallest i with cumS[i] > u.
  std::size_t i =
    std::upper_bound( m_cumS.begin(), m_cumS.begin() + n, u ) - m_cumS.begin();
  if ( i >= n )
    i = n - 1;

  double mu = 1.0 - 2.0 * m_edges[i] / E;   // corrected Bragg cosine (NOT 1 - E_i/E)
  mu = std::max( -1.0, std::min( 1.0, mu ) );
  return { ekin, NC::CosineScatAngle{ mu } };   // elastic: E' = E
}
