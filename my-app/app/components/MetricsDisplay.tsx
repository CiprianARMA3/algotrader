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
    <div className="space-y-20 pb-20">
      {/* 01. Volatility Matrix */}
      {volatilityData && volatilityData.length > 0 && (
        <section>
          <h3 className="text-[10px] font-black uppercase tracking-[0.5em] text-gray-600 mb-8 flex items-center gap-4">
            <span className="w-12 h-[1px] bg-gray-800"></span>
            01_VOLATILITY_VECTORS
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-0 border-t border-l border-gray-900">
            {volatilityData.map((vol, idx) => {
              const currentVol = Array.isArray(vol.realized_volatility) 
                ? vol.realized_volatility.slice(-1)[0] 
                : 0;
              
              return (
                <div key={idx} className="p-6 border-r border-b border-gray-900 hover:bg-white hover:text-black transition-all">
                  <div className="flex justify-between items-baseline mb-6">
                    <span className="text-xl font-black uppercase">{vol.symbol}</span>
                    <span className="text-xs font-mono">{(currentVol * 100).toFixed(2)}%</span>
                  </div>
                  <div className="space-y-3 text-[10px] uppercase font-bold tracking-widest opacity-60">
                    <div className="flex justify-between">
                      <span>Term_Structure</span>
                      <span>{vol.term_structure?.toFixed(3) || '0.000'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Skew_Asym</span>
                      <span>{vol.volatility_skew?.toFixed(3) || '0.000'}</span>
                    </div>
                    <div className="flex justify-between border-t border-gray-800 pt-2 mt-2">
                      <span>Jump_Intensity</span>
                      <span>{vol.jump_intensity?.toFixed(4) || '0.000'}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 02. Microstructure & Toxicity */}
      {microstructureData && microstructureData.length > 0 && (
        <section>
          <h3 className="text-[10px] font-black uppercase tracking-[0.5em] text-gray-600 mb-8 flex items-center gap-4">
            <span className="w-12 h-[1px] bg-gray-800"></span>
            02_MICROSTRUCTURE_ANOMALIES
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-0 border-t border-l border-gray-900">
            {microstructureData.map((micro, idx) => (
              <div key={idx} className="p-8 border-r border-b border-gray-900 flex justify-between">
                <div>
                  <h4 className="text-2xl font-black uppercase mb-2">{micro.symbol}</h4>
                  <p className="text-[10px] text-gray-500 uppercase tracking-widest">Adverse_Selection_Score</p>
                </div>
                <div className="text-right space-y-4">
                  <div>
                    <div className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">VPIN_Toxicity</div>
                    <div className="text-xl font-black">{(micro.vpin_toxicity || 0).toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Kyle_Lambda</div>
                    <div className="text-xs font-mono">{(micro.kyles_lambda || 0).toExponential(2)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 03. Spectral Analysis */}
      {trendingData && trendingData.length > 0 && (
        <section>
          <h3 className="text-[10px] font-black uppercase tracking-[0.5em] text-gray-600 mb-8 flex items-center gap-4">
            <span className="w-12 h-[1px] bg-gray-800"></span>
            03_SPECTRAL_STATIONARITY
          </h3>
          <div className="overflow-x-auto border border-gray-900">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white text-black uppercase text-[9px] font-black tracking-widest">
                  <th className="p-4">Symbol</th>
                  <th className="p-4">Trend_Vector</th>
                  <th className="p-4">Hurst_Exp</th>
                  <th className="p-4">Frac_Diff</th>
                  <th className="p-4">Stationary</th>
                  <th className="p-4">Status_Logic</th>
                </tr>
              </thead>
              <tbody className="text-[10px] font-bold uppercase tracking-tighter">
                {trendingData.map((trend, idx) => (
                  <tr key={idx} className="border-b border-gray-900 hover:bg-gray-950 transition-colors">
                    <td className="p-4 border-r border-gray-900 font-black text-xs">{trend.symbol}</td>
                    <td className="p-4 border-r border-gray-900">
                      <div className="flex items-center gap-4">
                        <div className="flex-1 bg-gray-900 h-1">
                          <div className="bg-white h-full" style={{ width: `${(trend.trend_strength || 0) * 100}%` }}></div>
                        </div>
                        <span className="w-8">{(trend.trend_strength || 0).toFixed(2)}</span>
                      </div>
                    </td>
                    <td className="p-4 border-r border-gray-900 font-mono">{trend.hurst_exponent?.toFixed(4)}</td>
                    <td className="p-4 border-r border-gray-900 font-mono">{trend.fractional_d?.toFixed(4)}</td>
                    <td className="p-4 border-r border-gray-900">
                      {trend.is_stationary ? '[ TRUE ]' : '[ FALSE ]'}
                    </td>
                    <td className="p-4 font-black">
                      {trend.hurst_exponent > 0.6 ? 'EQUILIBRIUM_TREND' : trend.hurst_exponent < 0.4 ? 'MEAN_REVERTING' : 'BROWNIAN_MOTION'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
