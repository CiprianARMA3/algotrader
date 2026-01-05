'use client';

import { AnalysisResponse } from '../types/api';
import MetricsDisplay from './MetricsDisplay';
import dynamic from 'next/dynamic';

const Charts = dynamic(() => import('./Charts'), { ssr: false });

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
    <div className="space-y-12">
      {/* Summary Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-0 border-t border-l border-gray-800">
        <div className="p-6 border-r border-b border-gray-800 group hover:bg-white hover:text-black transition-all">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 group-hover:text-black">Assets_Scanned</h3>
          <p className="text-4xl font-black">{instruments.length}</p>
          <div className="mt-2 text-[10px] font-mono text-gray-600 group-hover:text-black italic">Primary_Node_Active</div>
        </div>
        
        <div className="p-6 border-r border-b border-gray-800 group hover:bg-white hover:text-black transition-all">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 group-hover:text-black">Equilibrium_Pairs</h3>
          <p className="text-4xl font-black">{cointegration?.length || 0}</p>
          <div className="mt-2 text-[10px] font-mono text-gray-600 group-hover:text-black italic">Cointegration_Confirmed</div>
        </div>
        
        <div className="p-6 border-r border-b border-gray-800 group hover:bg-white hover:text-black transition-all">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 group-hover:text-black">Avg_Realized_Vol</h3>
          <p className="text-4xl font-black">
            {volatility && volatility.length > 0 
              ? `${(volatility.reduce((acc, v) => {
                  const currentVol = Array.isArray(v.realized_volatility) 
                    ? v.realized_volatility.slice(-1)[0] 
                    : 0;
                  return acc + currentVol;
                }, 0) / volatility.length * 100).toFixed(1)}%`
              : '0.0%'}
          </p>
          <div className="mt-2 text-[10px] font-mono text-gray-600 group-hover:text-black italic">Standard_Deviation_252D</div>
        </div>
        
        <div className="p-6 border-r border-b border-gray-800 group hover:bg-white hover:text-black transition-all">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 group-hover:text-black">Dominant_Flow</h3>
          <p className="text-xl font-black truncate uppercase">
            {execution_recommendations?.dominant_instrument || 'Calculating...'}
          </p>
          <div className="mt-2 text-[10px] font-mono text-gray-600 group-hover:text-black italic">
            {execution_recommendations?.lead_lag_score 
              ? `TE_SCORE: ${execution_recommendations.lead_lag_score.toFixed(4)}`
              : 'Transfer_Entropy_Inert'}
          </div>
        </div>
      </div>

      {/* Execution Logic - Stark Terminal Style */}
      {execution_recommendations && (
        <div className="border border-gray-800 p-8">
          <div className="flex justify-between items-start mb-8">
            <h2 className="text-xs font-black uppercase tracking-[0.4em] border-b-2 border-white pb-2">Execution_Logic_Output</h2>
            <span className="text-[10px] text-gray-600 font-mono">SYS_AUTH: STARK_REASONING</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
            <div className="space-y-3">
              <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Market_Regime</span>
              <div className="text-sm font-black border-l-2 border-white pl-3 py-1 uppercase">
                {execution_recommendations.market_regime || 'Neutral_State'}
              </div>
            </div>
            
            <div className="space-y-3">
              <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Vol_Env</span>
              <div className="text-sm font-black border-l-2 border-white pl-3 py-1 uppercase">
                {execution_recommendations.volatility_environment || 'Medium_Risk'}
              </div>
            </div>
            
            <div className="space-y-3">
              <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Sizing_Profile</span>
              <div className="text-sm font-black border-l-2 border-white pl-3 py-1 uppercase">
                {execution_recommendations.position_sizing || 'Normal_Allocation'}
              </div>
            </div>
            
            <div className="space-y-3">
              <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Execution_Bias</span>
              <div className="text-sm font-black border-l-2 border-white pl-3 py-1 uppercase">
                {execution_recommendations.execution_timing || 'Standard_Order'}
              </div>
            </div>
          </div>
          
          {/* Detailed Strategies Table Feel */}
          <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-12 pt-8 border-t border-gray-900">
            <div>
              <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest mb-4 block">Recommended_Vectors:</span>
              <div className="flex flex-wrap gap-3">
                {execution_recommendations.recommended_strategies?.map((strategy, idx) => (
                  <span key={idx} className="px-3 py-1 border border-gray-700 text-white text-[10px] font-bold uppercase tracking-tighter">
                    {strategy}
                  </span>
                )) || <span className="text-gray-700 italic text-[10px]">No_Active_Signals</span>}
              </div>
            </div>
            
            <div>
              <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest mb-4 block">Risk_Anomalies:</span>
              <div className="space-y-2">
                {execution_recommendations.risk_warnings?.map((warning, idx) => (
                  <div key={idx} className="text-[10px] font-bold text-white uppercase flex items-center gap-3">
                    <span className="w-1 h-1 bg-white"></span>
                    {warning}
                  </div>
                )) || <div className="text-gray-700 italic text-[10px]">Baseline_Noise_Only</div>}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Visual Data Matrices */}
      <Charts 
        volatilityData={volatility}
        trendingData={trending}
        cointegrationData={cointegration}
      />

      {/* Raw Metric Tables */}
      <MetricsDisplay 
        volatilityData={volatility}
        trendingData={trending}
        regimesData={regimes}
        microstructureData={microstructure}
      />
    </div>
  );
}
