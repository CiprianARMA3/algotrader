'use client';

import React, { useEffect, useRef } from 'react';

interface TradingViewChartProps {
  symbol: string;
}

declare global {
  interface Window {
    TradingView: any;
  }
}

export default function TradingViewChart({ symbol }: TradingViewChartProps) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.type = 'text/javascript';
    script.async = true;
    script.onload = () => {
      if (window.TradingView && container.current) {
        // Map symbols if needed (e.g., EURUSD=X to FX_IDC:EURUSD)
        let tvSymbol = symbol;
        if (symbol.includes('=X')) {
          tvSymbol = `FX_IDC:${symbol.replace('=X', '')}`;
        } else if (['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'].includes(symbol)) {
          tvSymbol = `NASDAQ:${symbol}`;
        }

        new window.TradingView.widget({
          "autosize": true,
          "symbol": tvSymbol,
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": false,
          "container_id": container.current.id,
          "backgroundColor": "rgba(0, 0, 0, 1)",
          "gridColor": "rgba(255, 255, 255, 0.06)",
          "width": "100%",
          "height": 500
        });
      }
    };
    document.head.appendChild(script);

    return () => {
      // Clean up script on unmount if necessary
    };
  }, [symbol]);

  return (
    <div className="border border-gray-900 bg-black p-1">
      <div className="flex justify-between items-center px-4 py-2 border-b border-gray-900">
        <span className="text-[10px] font-black uppercase tracking-[0.3em]">Live_Terminal_Feed: {symbol}</span>
        <span className="text-[8px] text-gray-600 font-mono">SOURCE: TRADINGVIEW_CORE</span>
      </div>
      <div id={`tv_chart_${symbol}`} ref={container} className="w-full h-[500px]" />
    </div>
  );
}
