import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown } from "lucide-react";

interface CryptoCardProps {
  name: string;
  symbol: string;
  price: number;
  change24h: number;
  sentiment?: string;
}

export const CryptoCard = ({ name, symbol, price, change24h, sentiment }: CryptoCardProps) => {
  const isPositive = change24h >= 0;
  
  return (
    <Card className="p-6 bg-card border-border hover:shadow-[var(--shadow-glow)] transition-all duration-300 hover:scale-105">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold text-foreground">{name}</h3>
          <p className="text-sm text-muted-foreground">{symbol}</p>
        </div>
        {isPositive ? (
          <TrendingUp className="text-success w-6 h-6" />
        ) : (
          <TrendingDown className="text-danger w-6 h-6" />
        )}
      </div>
      
      <div className="space-y-2">
        <p className="text-3xl font-bold text-foreground">
          ${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </p>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${isPositive ? 'text-success' : 'text-danger'}`}>
            {isPositive ? '+' : ''}{change24h.toFixed(2)}%
          </span>
          <span className="text-xs text-muted-foreground">24h</span>
        </div>
        
        {sentiment && (
          <div className="mt-3 pt-3 border-t border-border">
            <p className="text-xs text-muted-foreground">Sentiment</p>
            <p className="text-sm font-medium text-primary">{sentiment}</p>
          </div>
        )}
      </div>
    </Card>
  );
};
