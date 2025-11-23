import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";

const DEFAULT_COINS = ["bitcoin", "ethereum", "bnb"];

const fetchPrices = async (coins: string[]) => {
  const ids = coins.join(",");
  const url = `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd&include_last_updated_at=true`;

  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch data");

  return res.json();
};

const Index: React.FC = () => {
  const [coins, setCoins] = useState<string[]>(DEFAULT_COINS);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["prices", coins],
    queryFn: () => fetchPrices(coins),
    refetchInterval: 10000,
  });

  return (
    <div style={{ padding: 24 }}>
      <h1>CryptoAnalyzer — Market Snapshot</h1>

      <button onClick={() => refetch()} style={{ marginBottom: 12 }}>
        Refresh
      </button>

      <div style={{ marginBottom: 12 }}>
        <input
          placeholder="Add coin e.g. solana"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const coin = (e.target as HTMLInputElement).value.trim().toLowerCase();
              if (coin && !coins.includes(coin)) {
                setCoins((c) => [...c, coin]);
              }
              (e.target as HTMLInputElement).value = "";
            }
          }}
        />
      </div>

      {isLoading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>Error fetching market data</p>}

      {data && (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>Coin</th>
              <th style={{ textAlign: "right", borderBottom: "1px solid #ccc" }}>Price (USD)</th>
              <th style={{ textAlign: "right", borderBottom: "1px solid #ccc" }}>Updated</th>
            </tr>
          </thead>
          <tbody>
            {coins.map((coin) => (
              <tr key={coin}>
                <td style={{ padding: 8 }}>{coin}</td>
                <td style={{ padding: 8, textAlign: "right" }}>
                  {data[coin]?.usd ? `$${data[coin].usd.toLocaleString()}` : "—"}
                </td>
                <td style={{ padding: 8, textAlign: "right" }}>
                  {data[coin]?.last_updated_at
                    ? new Date(data[coin].last_updated_at * 1000).toLocaleTimeString()
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default Index;
