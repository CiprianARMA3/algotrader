import { useState, useCallback } from 'react';
import { AnalysisResponse, InstrumentRequest, AnalysisType, TimeFrame } from '../types/api';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || '';

export function useMarketAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [marketStatus, setMarketStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMarketStatus = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/market-status`);
      if (response.ok) {
        const data = await response.json();
        setMarketStatus(data);
      }
    } catch (err) {
      console.error('Core_Market_Status_Err:', err);
    }
  }, []);

  const performAnalysis = useCallback(async (selectedInstruments: string[], timeframe: string, lookbackDays: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const request: InstrumentRequest = {
        symbols: selectedInstruments,
        timeframe: timeframe as TimeFrame,
        lookback_days: lookbackDays
      };

      const response = await fetch(`${BACKEND_URL}/api/v1/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruments: request,
          analysis_types: [AnalysisType.FULL_ANALYSIS],
          parameters: {}
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Terminal_Sync_Fail (${response.status}): ${text}`);
      }

      const data: AnalysisResponse = await response.json();
      setAnalysisData(data);
      await fetchMarketStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown_System_Fault');
    } finally {
      setIsLoading(false);
    }
  }, [fetchMarketStatus]);

  const quickScan = useCallback(async (symbol: string, timeframe: string, lookbackDays: number) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/quick-analysis/${symbol}?timeframe=${timeframe}&lookback_days=${lookbackDays}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      console.error('Quick_Scan_Fault:', err);
    }
    return null;
  }, []);

  return {
    isLoading,
    analysisData,
    marketStatus,
    error,
    performAnalysis,
    fetchMarketStatus,
    quickScan
  };
}
