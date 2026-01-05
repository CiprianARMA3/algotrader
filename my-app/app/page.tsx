'use client';

import { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import { AnalysisResponse, InstrumentRequest, AnalysisType, TimeFrame } from './types/api';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || '';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>([
    'AAPL', 'MSFT', 'GOOGL', 'EURUSD=X', 'GBPUSD=X'
  ]);
  const [timeframe, setTimeframe] = useState<string>('1d');
  const [lookbackDays, setLookbackDays] = useState<number>(365);

  const performAnalysis = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const request: InstrumentRequest = {
        symbols: selectedInstruments,
        timeframe: timeframe as TimeFrame,
        lookback_days: lookbackDays
      };

      const analysisRequest = {
        instruments: request,
        analysis_types: [AnalysisType.FULL_ANALYSIS],
        parameters: {}
      };

      const response = await fetch(`${BACKEND_URL}/api/v1/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisRequest),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText || response.statusText}`);
      }

      const data: AnalysisResponse = await response.json();
      setAnalysisData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    performAnalysis();
  }, []);

  return (
    <main className="min-h-screen bg-black text-white font-mono p-4 md:p-8 selection:bg-white selection:text-black">
      <div className="max-w-7xl mx-auto border border-gray-800 p-6">
        {/* Header - Stark B&W */}
        <header className="mb-12 border-b border-gray-800 pb-8 flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold tracking-tighter uppercase">
              Institutional Terminal / v1.0
            </h1>
            <p className="text-gray-500 mt-2 text-sm uppercase tracking-widest">
              Quantitative Analysis & Risk Modeling Framework
            </p>
          </div>
          <div className="text-right text-xs text-gray-600">
            <div>STATUS: {isLoading ? 'PROCESSING' : 'READY'}</div>
            <div>CONN: {BACKEND_URL ? 'REMOTE' : 'LOCAL'}</div>
            <div>TS: {new Date().toISOString()}</div>
          </div>
        </header>

        {/* Controls - Minimalist Table Feel */}
        <div className="mb-12 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="col-span-2">
              <label className="block text-xs font-bold text-gray-500 uppercase mb-3 tracking-widest">
                Portfolio Basket
              </label>
              <div className="flex flex-wrap gap-2">
                {selectedInstruments.map((symbol) => (
                  <span
                    key={symbol}
                    className="px-3 py-1 bg-white text-black text-xs font-bold flex items-center gap-2 hover:bg-gray-200 transition-colors"
                  >
                    {symbol}
                    <button
                      onClick={() => setSelectedInstruments(prev => prev.filter(s => s !== symbol))}
                      className="hover:text-red-600 transition-colors"
                    >
                      ×
                    </button>
                  </span>
                ))}
                <button className="px-3 py-1 border border-dashed border-gray-700 text-gray-500 text-xs hover:border-white hover:text-white transition-all">
                  + ADD_SYMBOL
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase mb-3 tracking-widest">Interval</label>
                <select
                  value={timeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                  className="w-full bg-black border border-gray-800 px-3 py-2 text-xs uppercase focus:border-white outline-none transition-colors cursor-pointer"
                >
                  <option value="1d">1_Day</option>
                  <option value="1h">1_Hour</option>
                  <option value="15m">15_Min</option>
                  <option value="1wk">1_Week</option>
                </select>
              </div>
              
              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase mb-3 tracking-widest">Window</label>
                <select
                  value={lookbackDays}
                  onChange={(e) => setLookbackDays(Number(e.target.value))}
                  className="w-full bg-black border border-gray-800 px-3 py-2 text-xs uppercase focus:border-white outline-none transition-colors cursor-pointer"
                >
                  <option value={30}>30_D</option>
                  <option value={90}>90_D</option>
                  <option value={180}>180_D</option>
                  <option value={365}>1_YR</option>
                  <option value={730}>2_YR</option>
                </select>
              </div>
            </div>
          </div>

          <div className="flex justify-end pt-4 border-t border-gray-900">
            <button
              onClick={performAnalysis}
              disabled={isLoading}
              className="px-8 py-3 bg-white text-black text-xs font-black uppercase tracking-widest hover:bg-gray-200 disabled:opacity-20 transition-all active:scale-95"
            >
              {isLoading ? 'RUNNING_MODELS...' : 'EXECUTE_ANALYSIS'}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-12 p-4 border border-red-900 bg-red-950/10 text-red-500 text-xs">
            <span className="font-bold uppercase mr-2">[CRITICAL_ERROR]:</span> {error}
          </div>
        )}

        {/* Dashboard Content */}
        <div className="min-h-[400px]">
          {isLoading ? (
            <div className="flex flex-col justify-center items-center h-64 space-y-4">
              <div className="w-12 h-1 border-t-2 border-white animate-pulse"></div>
              <div className="text-[10px] text-gray-500 tracking-[0.3em] uppercase animate-pulse">
                Synthesizing Statistical Matrices
              </div>
            </div>
          ) : (
            analysisData && <Dashboard data={analysisData} />
          )}
        </div>

        {/* Bottom Data Legend - High Contrast */}
        <footer className="mt-20 border-t border-gray-800 pt-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-12 text-[10px]">
            <div className="space-y-4">
              <h3 className="font-bold text-white uppercase tracking-widest border-b border-gray-900 pb-2">Econometrics</h3>
              <p className="text-gray-500 leading-relaxed">
                Cointegration, Johansen Multivariate tests, and PCA Eigenportfolios are used to identify structural equilibrium and residual anomalies.
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="font-bold text-white uppercase tracking-widest border-b border-gray-900 pb-2">Microstructure</h3>
              <p className="text-gray-500 leading-relaxed">
                VPIN and Kyle's Lambda quantify toxic flow and market depth to assess liquidity risk and adverse selection.
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="font-bold text-white uppercase tracking-widest border-b border-gray-900 pb-2">Signals</h3>
              <p className="text-gray-500 leading-relaxed">
                Fractional Differentiation maintains memory while ensuring stationarity. Wavelet Denoising separates signal from Gaussian noise.
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="font-bold text-white uppercase tracking-widest border-b border-gray-900 pb-2">Modeling</h3>
              <p className="text-gray-500 leading-relaxed">
                Bayesian Structural Time Series and GJR-GARCH models manage time-varying volatility and regime-switching probabilities.
              </p>
            </div>
          </div>
          <div className="mt-12 text-center text-gray-700 text-[8px] tracking-[0.5em] uppercase">
            Data provided by institutional feeds. Proprietary Framework v1.0.26
          </div>
        </footer>
      </div>
    </main>
  );
}
