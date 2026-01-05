'use client';

interface MetricsDisplayProps {
  volatilityData?: any[];
  trendingData?: any[];
  regimesData?: any[];
  microstructureData?: any[];
}

export default function MetricsDisplay({ 
  volatilityData, 
  trendingData, 
  regimesData,
  microstructureData 
}: MetricsDisplayProps) {
  return (
    <div className="space-y-6">
      {/* Volatility Metrics */}
      {volatilityData && volatilityData.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold mb-4">Volatility Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {volatilityData.slice(0, 8).map((vol, idx) => (
              <div key={idx} className="bg-gray-900/50 p-4 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <span className="font-semibold text-blue-300">{vol.symbol}</span>
                  <span className={`text-sm px-2 py-1 rounded ${
                    vol.realized_volatility > 0.3 
                      ? 'bg-red-900/50 text-red-300' 
                      : vol.realized_volatility < 0.15
                      ? 'bg-green-900/50 text-green-300'
                      : 'bg-yellow-900/50 text-yellow-300'
                  }`}>
                    {(vol.realized_volatility * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="space-y-1 text-sm text-gray-400">
                  <div className="flex justify-between">
                    <span>Short Term:</span>
                    <span>{(vol.short_term_vol * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Long Term:</span>
                    <span>{(vol.long_term_vol * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Skew:</span>
                    <span className={vol.volatility_skew > 0 ? 'text-green-400' : 'text-red-400'}>
                      {vol.volatility_skew.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Leverage Effect:</span>
                    <span className={vol.leverage_effect < 0 ? 'text-green-400' : 'text-red-400'}>
                      {vol.leverage_effect.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trending Metrics */}
      {trendingData && trendingData.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold mb-4">Trending Analysis</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 border-b border-gray-700">
                  <th className="pb-3">Symbol</th>
                  <th className="pb-3">Trend Strength</th>
                  <th className="pb-3">Hurst Exponent</th>
                  <th className="pb-3">Fractional D</th>
                  <th className="pb-3">Stationary</th>
                  <th className="pb-3">ADF Stat</th>
                  <th className="pb-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {trendingData.map((trend, idx) => (
                  <tr key={idx} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-3 font-medium">{trend.symbol}</td>
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              trend.trend_strength > 0.7 
                                ? 'bg-green-500' 
                                : trend.trend_strength < 0.3
                                ? 'bg-red-500'
                                : 'bg-yellow-500'
                            }`}
                            style={{ width: `${Math.min(trend.trend_strength * 100, 100)}%` }}
                          />
                        </div>
                        <span className="text-sm">{(trend.trend_strength * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <span className={trend.hurst_exponent > 0.65 
                        ? 'text-green-400' 
                        : trend.hurst_exponent < 0.35 
                        ? 'text-red-400' 
                        : 'text-yellow-400'
                      }>
                        {trend.hurst_exponent.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-3">{trend.fractional_d.toFixed(3)}</td>
                    <td className="py-3">
                      <span className={`px-2 py-1 rounded text-xs ${
                        trend.is_stationary 
                          ? 'bg-green-900/50 text-green-300' 
                          : 'bg-red-900/50 text-red-300'
                      }`}>
                        {trend.is_stationary ? 'Stationary' : 'Non-Stationary'}
                      </span>
                    </td>
                    <td className="py-3">{trend.adf_statistic.toFixed(3)}</td>
                    <td className="py-3">
                      {trend.hurst_exponent > 0.65 && trend.trend_strength > 0.7 ? (
                        <span className="text-green-400 font-semibold">Strong Trend</span>
                      ) : trend.hurst_exponent < 0.35 && trend.trend_strength < 0.3 ? (
                        <span className="text-red-400 font-semibold">Mean Reverting</span>
                      ) : (
                        <span className="text-yellow-400 font-semibold">Random Walk</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Regime Metrics */}
      {regimesData && regimesData.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold mb-4">Market Regimes</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {regimesData.map((regime, idx) => (
              <div key={idx} className="bg-gray-900/50 p-4 rounded-lg">
                <div className="flex justify-between items-center mb-3">
                  <span className="font-semibold text-purple-300">{regime.symbol}</span>
                  <span className="text-sm text-gray-400">
                    {regime.regime_statistics ? Object.keys(regime.regime_statistics).length : 0} regimes
                  </span>
                </div>
                {regime.regime_statistics && (
                  <div className="space-y-2">
                    {Object.entries(regime.regime_statistics).map(([regimeName, stats]: [string, any]) => (
                      <div key={regimeName} className="text-sm">
                        <div className="flex justify-between mb-1">
                          <span className="text-gray-400">{regimeName}</span>
                          <span className="font-medium">{(stats.percentage || 0).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-800 rounded-full h-1.5">
                          <div 
                            className="h-1.5 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                            style={{ width: `${stats.percentage || 0}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>μ: {(stats.mean_return * 100).toFixed(2)}%</span>
                          <span>σ: {(stats.std_return * 100).toFixed(2)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Microstructure Metrics */}
      {microstructureData && microstructureData.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold mb-4">Microstructure Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {microstructureData.map((micro, idx) => (
              <div key={idx} className="bg-gray-900/50 p-4 rounded-lg">
                <div className="flex justify-between items-center mb-3">
                  <span className="font-semibold text-blue-300">{micro.symbol}</span>
                  <span className={`text-sm px-2 py-1 rounded ${
                    micro.amihud_illiquidity > 1e-6 
                      ? 'bg-red-900/50 text-red-300' 
                      : 'bg-green-900/50 text-green-300'
                  }`}>
                    Illiquidity: {(micro.amihud_illiquidity * 1e6).toFixed(2)}
                  </span>
                </div>
                <div className="space-y-2 text-sm text-gray-400">
                  <div className="flex justify-between">
                    <span>Mean Volume:</span>
                    <span>{(micro.volume_profile?.mean_volume / 1e6).toFixed(2)}M</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Vol Skew:</span>
                    <span>{micro.volume_profile?.volume_skew?.toFixed(2) || '0.00'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Realized Vol:</span>
                    <span>{(micro.volatility_metrics?.realized_volatility * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Term Structure:</span>
                    <span className={
                      micro.volatility_metrics?.term_structure > 1.2 
                        ? 'text-red-400' 
                        : micro.volatility_metrics?.term_structure < 0.8
                        ? 'text-green-400'
                        : 'text-yellow-400'
                    }>
                      {micro.volatility_metrics?.term_structure?.toFixed(2) || '1.00'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}