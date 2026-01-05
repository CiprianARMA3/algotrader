'use client';

import { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import { AnalysisResponse, InstrumentRequest, AnalysisType } from './types/api';

// Cloudflare tunnel URL (replace with your actual URL)
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
        timeframe: timeframe as any,
        lookback_days: lookbackDays
      };

      const analysisRequest = {
        instruments: request,
        analysis_types: ['full'],
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
        throw new Error(`API error: ${response.statusText}`);
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

  const quickAnalyze = async (symbol: string) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/quick-analysis/${symbol}?timeframe=${timeframe}&lookback_days=${lookbackDays}`);
      const data = await response.json();
      console.log('Quick analysis:', data);
      // Update UI with quick analysis results
    } catch (err) {
      console.error('Quick analysis error:', err);
    }
  };

  // Fetch initial data on component mount
  useEffect(() => {
    performAnalysis();
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
            Algorithmic Trading Analytics Dashboard
          </h1>
          <p className="text-gray-400 mt-2">
            Advanced quantitative frameworks for institutional trading analysis
          </p>
        </header>

        {/* Controls */}
        <div className="bg-gray-800 rounded-xl p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Selected Instruments ({selectedInstruments.length})
              </label>
              <div className="flex flex-wrap gap-2">
                {selectedInstruments.map((symbol) => (
                  <span
                    key={symbol}
                    className="px-3 py-1 bg-gray-700 rounded-full text-sm flex items-center gap-2"
                  >
                    {symbol}
                    <button
                      onClick={() => setSelectedInstruments(prev => prev.filter(s => s !== symbol))}
                      className="text-gray-400 hover:text-red-400"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
            
            <div className="flex gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Timeframe</label>
                <select
                  value={timeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                  className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
                >
                  <option value="1d">Daily</option>
                  <option value="1h">Hourly</option>
                  <option value="15m">15 Minute</option>
                  <option value="1wk">Weekly</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Lookback</label>
                <select
                  value={lookbackDays}
                  onChange={(e) => setLookbackDays(Number(e.target.value))}
                  className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
                >
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                  <option value={180}>180 days</option>
                  <option value={365}>1 year</option>
                  <option value={730}>2 years</option>
                </select>
              </div>
              
              <div className="flex items-end">
                <button
                  onClick={performAnalysis}
                  disabled={isLoading}
                  className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-semibold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Analyzing...' : 'Run Analysis'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-8 p-4 bg-red-900/50 border border-red-700 rounded-lg">
            <p className="text-red-300">Error: {error}</p>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        )}

        {/* Dashboard Content */}
        {analysisData && !isLoading && (
          <Dashboard data={analysisData} />
        )}

        {/* Info Section */}
        <div className="mt-12 p-6 bg-gradient-to-r from-gray-800 to-gray-900 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">Analysis Methodologies</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-gray-800/50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-400 mb-2">Econometrics</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Cointegration & Johansen Test</li>
                <li>• PCA Eigenportfolios</li>
                <li>• Vector Error Correction Models</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-400 mb-2">Signal Processing</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Fractional Differentiation</li>
                <li>• Wavelet Transforms</li>
                <li>• Kalman Filter Estimation</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg">
              <h3 className="font-semibold text-green-400 mb-2">Volatility</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• GARCH Family Models</li>
                <li>• Realized Volatility</li>
                <li>• Regime Detection</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg">
              <h3 className="font-semibold text-yellow-400 mb-2">Machine Learning</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Hidden Markov Models</li>
                <li>• Transfer Entropy</li>
                <li>• Change Point Detection</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}