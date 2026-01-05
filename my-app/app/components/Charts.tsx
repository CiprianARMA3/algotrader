'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

interface ChartsProps {
  volatilityData?: any[];
  trendingData?: any[];
  cointegrationData?: any[];
}

export default function Charts({ volatilityData, trendingData, cointegrationData }: ChartsProps) {
  const volChartData = volatilityData?.map((v) => ({
    name: v.symbol,
    volatility: Array.isArray(v.realized_volatility) 
      ? v.realized_volatility.slice(-1)[0] * 100 
      : 0,
    garch_vol: Array.isArray(v.garch_volatility)
      ? v.garch_volatility.slice(-1)[0] * 100
      : 0
  })) || [];

  const trendChartData = trendingData?.map((t) => ({
    name: t.symbol,
    trendStrength: t.trend_strength || 0,
    hurstExponent: t.hurst_exponent || 0.5,
    fractionalD: t.fractional_d || 0.5
  })) || [];

  const cointChartData = cointegrationData?.slice(0, 10).map((c) => ({
    name: Array.isArray(c.pair) ? c.pair.join('/') : 'UNKNOWN',
    halfLife: c.half_life || 0,
    hurst: c.hurst_exponent || 0.5,
    pValue: c.cointegration_pvalue || 1
  })) || [];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-black border border-gray-800 p-3 text-[10px] uppercase font-bold">
          <p className="border-b border-gray-900 pb-1 mb-2 tracking-widest">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: 'white' }}>
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
      {/* Volatility - B&W */}
      <div className="border border-gray-900 p-6">
        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] mb-8 border-l-4 border-white pl-4">Matrix_Vol_Dispersion</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={volChartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#111" vertical={false} />
            <XAxis dataKey="name" stroke="#333" fontSize={10} tickLine={false} axisLine={false} />
            <YAxis stroke="#333" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#0a0a0a' }} />
            <Legend iconType="square" align="right" verticalAlign="top" wrapperStyle={{ fontSize: '8px', textTransform: 'uppercase', letterSpacing: '2px', paddingBottom: '20px' }} />
            <Bar dataKey="volatility" fill="#FFFFFF" name="REALIZED_V" />
            <Bar dataKey="garch_vol" fill="#333333" name="GARCH_V" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Trending - B&W Scatter */}
      <div className="border border-gray-900 p-6">
        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] mb-8 border-l-4 border-white pl-4">Spectral_Trend_Correlation</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#111" />
            <XAxis dataKey="hurstExponent" type="number" name="HURST" stroke="#333" domain={[0, 1]} fontSize={10} tickLine={false} axisLine={false} />
            <YAxis dataKey="trendStrength" type="number" name="STRENGTH" stroke="#333" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Scatter name="Assets" data={trendChartData} fill="#FFF" shape="cross" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Cointegration - B&W */}
      <div className="border border-gray-900 p-6">
        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] mb-8 border-l-4 border-white pl-4">Equilibrium_Half_Life</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={cointChartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#111" vertical={false} />
            <XAxis dataKey="name" stroke="#333" fontSize={10} tickLine={false} axisLine={false} />
            <YAxis stroke="#333" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#0a0a0a' }} />
            <Bar dataKey="halfLife" fill="#FFF" name="HL_DAYS" />
            <Bar dataKey="hurst" fill="#222" name="HURST_E" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Numerical Matrix Section */}
      <div className="border border-gray-900 p-6">
        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] mb-8 border-l-4 border-white pl-4">Data_Stream_Output</h3>
        <div className="space-y-6">
          {trendChartData.slice(0, 4).map((item) => (
            <div key={item.name} className="space-y-2">
              <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest">
                <span>{item.name}</span>
                <span className="text-gray-500">Stability: {item.hurstExponent.toFixed(3)}</span>
              </div>
              <div className="w-full bg-gray-900 h-[2px]">
                <div 
                  className="bg-white h-full transition-all duration-1000"
                  style={{ width: `${Math.min(item.trendStrength * 100, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
