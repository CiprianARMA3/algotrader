'use client';

import { AnalysisResponse } from '../types/api';
import MetricsDisplay from './MetricsDisplay';
import Charts from './Charts';

interface DashboardProps {
  data: AnalysisResponse;
}

export default function Dashboard({ data }: DashboardProps) {
  const { 
    instruments,
    cointegration,
    volatility,
    regimes,
    trending,
    microstructure,
    execution_recommendations 
  } = data;

  return (
    <div className="space-y-8">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 p-6 rounded-xl border border-blue-800/50">
          <h3 className="text-lg font-semibold text-blue-300">Instruments</h3>
          <p className="text-3xl font-bold mt-2">{instruments.length}</p>
          <p className="text-sm text-gray-400 mt-1">Analyzed</p>
        </div>
        
        <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 p-6 rounded-xl border border-purple-800/50">
          <h3 className="text-lg font-semibold text-purple-300">Cointegrated Pairs</h3>
          <p className="text-3xl font-bold mt-2">{cointegration?.length || 0}</p>
          <p className="text-sm text-gray-400 mt-1">Trading opportunities</p>
        </div>
        
        <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 p-6 rounded-xl border border-green-800/50">
          <h3 className="text-lg font-semibold text-green-300">Avg Volatility</h3>
          <p className="text-3xl font-bold mt-2">
            {volatility && volatility.length > 0 
              ? `${(volatility.reduce((acc, v) => acc + (v.realized_volatility || 0), 0) / volatility.length * 100).toFixed(1)}%`
              : '0%'}
          </p>
          <p className="text-sm text-gray-400 mt-1">Annualized</p>
        </div>
        
        <div className="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 p-6 rounded-xl border border-yellow-800/50">
          <h3 className="text-lg font-semibold text-yellow-300">Market Regime</h3>
          <p className="text-3xl font-bold mt-2 capitalize">
            {execution_recommendations?.market_regime?.replace('_', ' ') || 'Neutral'}
          </p>
          <p className="text-sm text-gray-400 mt-1">Current state</p>
        </div>
      </div>

      {/* Recommendations */}
      {execution_recommendations && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700">
          <h2 className="text-2xl font-bold mb-4">Execution Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <span className="text-sm text-gray-400">Market Regime</span>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                execution_recommendations.market_regime?.includes('bull') 
                  ? 'bg-green-900/50 text-green-300' 
                  : execution_recommendations.market_regime?.includes('bear')
                  ? 'bg-red-900/50 text-red-300'
                  : 'bg-yellow-900/50 text-yellow-300'
              }`}>
                {execution_recommendations.market_regime}
              </div>
            </div>
            
            <div className="space-y-2">
              <span className="text-sm text-gray-400">Volatility</span>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                execution_recommendations.volatility_environment === 'high'
                  ? 'bg-red-900/50 text-red-300'
                  : execution_recommendations.volatility_environment === 'low'
                  ? 'bg-green-900/50 text-green-300'
                  : 'bg-yellow-900/50 text-yellow-300'
              }`}>
                {execution_recommendations.volatility_environment}
              </div>
            </div>
            
            <div className="space-y-2">
              <span className="text-sm text-gray-400">Position Sizing</span>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                execution_recommendations.position_sizing === 'reduced'
                  ? 'bg-red-900/50 text-red-300'
                  : 'bg-green-900/50 text-green-300'
              }`}>
                {execution_recommendations.position_sizing}
              </div>
            </div>
            
            <div className="space-y-2">
              <span className="text-sm text-gray-400">Execution Timing</span>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                execution_recommendations.execution_timing === 'aggressive'
                  ? 'bg-green-900/50 text-green-300'
                  : execution_recommendations.execution_timing === 'conservative'
                  ? 'bg-red-900/50 text-red-300'
                  : 'bg-yellow-900/50 text-yellow-300'
              }`}>
                {execution_recommendations.execution_timing}
              </div>
            </div>
          </div>
          
          {/* Recommended Strategies */}
          {execution_recommendations.recommended_strategies && 
           execution_recommendations.recommended_strategies.length > 0 && (
            <div className="mt-4">
              <span className="text-sm text-gray-400">Recommended Strategies:</span>
              <div className="flex flex-wrap gap-2 mt-2">
                {execution_recommendations.recommended_strategies.map((strategy, idx) => (
                  <span key={idx} className="px-3 py-1 bg-blue-900/50 text-blue-300 rounded-full text-sm">
                    {strategy}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          {/* Risk Warnings */}
          {execution_recommendations.risk_warnings && 
           execution_recommendations.risk_warnings.length > 0 && (
            <div className="mt-4">
              <span className="text-sm text-red-400">Risk Warnings:</span>
              <ul className="mt-2 space-y-1">
                {execution_recommendations.risk_warnings.map((warning, idx) => (
                  <li key={idx} className="text-sm text-red-300">• {warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Charts Section */}
      <Charts 
        volatilityData={volatility}
        trendingData={trending}
        cointegrationData={cointegration}
      />

      {/* Metrics Display */}
      <MetricsDisplay 
        volatilityData={volatility}
        trendingData={trending}
        regimesData={regimes}
        microstructureData={microstructure}
      />
    </div>
  );
}