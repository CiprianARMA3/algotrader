'use client';

import { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import { useMarketAnalysis } from './hooks/useMarketAnalysis';

export default function Home() {
  const { 
    isLoading, 
    analysisData, 
    marketStatus, 
    error, 
    performAnalysis, 
    fetchMarketStatus, 
    quickScan 
  } = useMarketAnalysis();

  const [selectedInstruments, setSelectedInstruments] = useState<string[]>([
    'AAPL', 'MSFT', 'GOOGL', 'EURUSD=X', 'GBPUSD=X'
  ]);
  const [timeframe, setTimeframe] = useState<string>('1d');
  const [lookbackDays, setLookbackDays] = useState<number>(365);
  const [logs, setLogs] = useState<string[]>(['[SYSTEM]: Initializing Core...']);

  const addLog = (msg: string) => setLogs(prev => [msg, ...prev].slice(0, 10));

  const handleAnalysis = async () => {
    addLog(`[EXECUTE]: Analyzing ${selectedInstruments.length} assets...`);
    await performAnalysis(selectedInstruments, timeframe, lookbackDays);
    addLog('[SUCCESS]: Matrix synthesized.');
  };

  const handleQuickScan = async (symbol: string) => {
    addLog(`[SCAN]: Deep probing ${symbol}...`);
    const data = await quickScan(symbol, timeframe, lookbackDays);
    if (data) {
      alert(`QUICK_SCAN: ${symbol}\nPrice: ${data.basic_metrics.current_price}\nVolatility: ${(data.basic_metrics.volatility * 100).toFixed(2)}%\nTrend: ${(data.basic_metrics.trend_strength * 100).toFixed(0)}%`);
      addLog(`[DATA]: Probe successful for ${symbol}.`);
    }
  };

  useEffect(() => {
    handleAnalysis();
    fetchMarketStatus();
  }, []);

  return (
    <main className="min-h-screen bg-black text-white font-mono selection:bg-white selection:text-black flex flex-col lg:flex-row">
      {/* Sidebar - Terminal Logs */}
      <aside className="w-full lg:w-64 border-r border-gray-900 p-6 space-y-8 order-2 lg:order-1">
        <div>
          <h2 className="text-[10px] font-black uppercase tracking-[0.3em] mb-6 text-gray-600">System_Trace</h2>
          <div className="space-y-4">
            {logs.map((log, i) => (
              <div key={i} className="text-[9px] font-bold leading-relaxed break-all opacity-80 animate-in fade-in slide-in-from-left duration-500">
                {log}
              </div>
            ))}
          </div>
        </div>
        
        <div className="pt-8 border-t border-gray-900">
          <h2 className="text-[10px] font-black uppercase tracking-[0.3em] mb-4 text-gray-600">Network_Core</h2>
          <div className="space-y-2 text-[9px] uppercase font-bold">
            <div className="flex justify-between"><span>Node:</span> <span className="text-white">FRA-1</span></div>
            <div className="flex justify-between"><span>Lat:</span> <span className="text-white">12ms</span></div>
            <div className="flex justify-between"><span>Enc:</span> <span className="text-white">AES-256</span></div>
          </div>
        </div>
      </aside>

      {/* Main Terminal Feed */}
      <section className="flex-1 p-4 md:p-8 order-1 lg:order-2 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <header className="mb-12 border-b border-gray-900 pb-8 flex flex-col md:flex-row justify-between items-end gap-4">
            <div>
              <h1 className="text-3xl font-black tracking-tighter uppercase">Institutional_Terminal_v1.0</h1>
              <p className="text-gray-600 mt-2 text-[10px] uppercase tracking-[0.4em]">High_Performance_Quantitative_Engine</p>
            </div>
            <div className="text-right text-[10px] text-gray-500 space-y-1 font-bold uppercase tracking-tighter">
              <div className="flex gap-4">
                <span>VIX: <span className="text-white">{marketStatus?.market_indicators?.VIX?.current?.toFixed(2) || '0.00'}</span></span>
                <span>BREADTH: <span className="text-white">{marketStatus?.market_breadth?.advance_decline_ratio?.toFixed(2) || '0.00'}</span></span>
              </div>
              <div className="opacity-40">{new Date().toISOString()}</div>
            </div>
          </header>

          {/* Controls */}
          <div className="mb-12 space-y-10">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-12">
              <div className="xl:col-span-2">
                <label className="text-[10px] font-black text-gray-600 uppercase mb-4 block tracking-[0.3em]">Portfolio_Basket</label>
                <div className="flex flex-wrap gap-2">
                  {selectedInstruments.map((symbol) => (
                    <button
                      key={symbol}
                      onClick={() => handleQuickScan(symbol)}
                      className="px-3 py-1 bg-white text-black text-[10px] font-black uppercase flex items-center gap-3 hover:invert transition-all"
                    >
                      {symbol}
                      <span 
                        onClick={(e) => { e.stopPropagation(); setSelectedInstruments(p => p.filter(s => s !== symbol)); }}
                        className="opacity-30 hover:opacity-100"
                      >×</span>
                    </button>
                  ))}
                  <button 
                    onClick={() => { const s = prompt('Symbol:'); if(s) setSelectedInstruments(p => [...p, s.toUpperCase()]); }}
                    className="px-3 py-1 border border-dashed border-gray-800 text-gray-600 text-[10px] font-black uppercase hover:border-white hover:text-white transition-all"
                  >+ ADD_UNIT</button>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="text-[10px] font-black text-gray-600 uppercase mb-4 block tracking-[0.3em]">Interval</label>
                  <select value={timeframe} onChange={e => setTimeframe(e.target.value)} className="w-full bg-black border border-gray-900 px-3 py-2 text-[10px] uppercase font-black focus:border-white outline-none">
                    <option value="1d">1_DAY</option><option value="1h">1_HOUR</option><option value="15m">15_MIN</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-600 uppercase mb-4 block tracking-[0.3em]">Window</label>
                  <select value={lookbackDays} onChange={e => setLookbackDays(Number(e.target.value))} className="w-full bg-black border border-gray-900 px-3 py-2 text-[10px] uppercase font-black focus:border-white outline-none">
                    <option value={30}>30_D</option><option value={180}>180_D</option><option value={365}>1_YR</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-6 border-t border-gray-950">
              <button
                onClick={handleAnalysis}
                disabled={isLoading}
                className="px-12 py-4 bg-white text-black text-[10px] font-black uppercase tracking-[0.5em] hover:invert disabled:opacity-10 transition-all active:scale-95 shadow-[0_0_20px_rgba(255,255,255,0.1)]"
              >
                {isLoading ? 'CALCULATING_MATRIX...' : 'EXECUTE_SYNC'}
              </button>
            </div>
          </div>

          {/* Main Feed */}
          {error && <div className="mb-12 p-4 border border-red-900 bg-red-950/10 text-red-500 text-[10px] font-black uppercase tracking-widest"><span className="bg-red-500 text-black px-2 mr-3">FAIL</span> {error}</div>}
          
          <div className="min-h-[600px] animate-in fade-in duration-1000">
            {isLoading ? (
              <div className="flex flex-col justify-center items-center h-96 space-y-6">
                <div className="w-24 h-[1px] bg-white animate-pulse"></div>
                <div className="text-[9px] text-gray-600 tracking-[0.5em] uppercase animate-pulse">Neural_Sync_Active</div>
              </div>
            ) : (
              analysisData && <Dashboard data={analysisData} />
            )}
          </div>

          {/* Footer Legend */}
          <footer className="mt-32 pt-16 border-t border-gray-900 grid grid-cols-1 md:grid-cols-4 gap-12 opacity-40 hover:opacity-100 transition-opacity duration-1000">
            {['Econometrics', 'Microstructure', 'Signals', 'Modeling'].map((h, i) => (
              <div key={i} className="space-y-4">
                <h3 className="text-[9px] font-black uppercase tracking-widest border-b border-gray-900 pb-2">{h}</h3>
                <p className="text-[8px] leading-relaxed uppercase">Advanced_Quantitative_Module_v1.0.26_Loaded</p>
              </div>
            ))}
          </footer>
        </div>
      </section>
    </main>
  );
}
