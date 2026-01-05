'use client';

import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

interface ChartsProps {
  volatilityData?: any[];
  trendingData?: any[];
  cointegrationData?: any[];
}

export default function Charts({ volatilityData, trendingData, cointegrationData }: ChartsProps) {
  // Prepare volatility chart data
  const volChartData = volatilityData?.map((v, idx) => ({
    name: v.symbol,
    volatility: Array.isArray(v.realized_volatility) 
      ? v.realized_volatility.slice(-1)[0] * 100 
      : 0,
    garch_vol: Array.isArray(v.garch_volatility)
      ? v.garch_volatility.slice(-1)[0] * 100
      : 0
  })) || [];

  // Prepare trending data
  const trendChartData = trendingData?.map((t, idx) => ({
    name: t.symbol,
    trendStrength: t.trend_strength || 0,
    hurstExponent: t.hurst_exponent || 0.5,
    fractionalD: t.fractional_d || 0.5
  })) || [];

  // Prepare cointegration data
  const cointChartData = cointegrationData?.slice(0, 10).map((c, idx) => ({
    name: c.pair.join('/'),
    halfLife: c.half_life || 0,
    hurst: c.hurst_exponent || 0.5,
    pValue: c.cointegration_pvalue || 1
  })) || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Volatility Comparison Chart */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Volatility Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={volChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="name" stroke="#888" fontSize={12} />
            <YAxis stroke="#888" fontSize={12} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Volatility']}
            />
            <Legend />
            <Bar dataKey="volatility" fill="#3b82f6" name="Realized Vol" />
            <Bar dataKey="garch_vol" fill="#8b5cf6" name="GARCH Vol" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Trending Analysis Chart */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Trending Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis 
              dataKey="hurstExponent" 
              type="number"
              name="Hurst Exponent"
              stroke="#888"
              domain={[0, 1]}
            />
            <YAxis 
              dataKey="trendStrength" 
              type="number"
              name="Trend Strength"
              stroke="#888"
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              formatter={(value, name) => [Number(value).toFixed(3), name]}
            />
            <Legend />
            <Scatter 
              name="Instruments" 
              data={trendChartData} 
              fill="#10b981"
              shape="circle"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Cointegration Analysis */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Cointegration Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={cointChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="name" stroke="#888" fontSize={10} angle={-45} textAnchor="end" />
            <YAxis stroke="#888" fontSize={12} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              formatter={(value) => [Number(value).toFixed(2), 'Value']}
            />
            <Legend />
            <Bar dataKey="halfLife" fill="#f59e0b" name="Half Life (days)" />
            <Bar dataKey="hurst" fill="#ec4899" name="Hurst Exponent" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Market Regime */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Market Metrics</h3>
        <div className="space-y-4">
          {trendChartData.slice(0, 5).map((item) => (
            <div key={item.name} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-300">{item.name}</span>
                <span className={`font-semibold ${
                  item.trendStrength > 0.7 ? 'text-green-400' :
                  item.trendStrength < 0.3 ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  Trend: {(item.trendStrength * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                  style={{ width: `${Math.min(item.trendStrength * 100, 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-400">
                <span>Hurst: {item.hurstExponent.toFixed(3)}</span>
                <span>Fractional D: {item.fractionalD.toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}