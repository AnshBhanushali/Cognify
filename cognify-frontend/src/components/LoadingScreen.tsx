import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-ai-network.jpg";

interface LoadingScreenProps {
  onComplete: () => void;
}

export const LoadingScreen = ({ onComplete }: LoadingScreenProps) => {
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsComplete(true); // ✅ just mark complete, don't auto-navigate
          return 100;
        }
        return prev + 2;
      });
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-hero flex items-center justify-center relative overflow-hidden">
      {/* Background glow effect */}
      <div className="absolute inset-0 bg-gradient-glow opacity-50" />
      
      {/* Hero image with overlay */}
      <div className="absolute inset-0 opacity-20">
        <img 
          src={heroImage} 
          alt="AI Network" 
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-hero/60" />
      </div>

      <div className="relative z-10 text-center max-w-4xl mx-auto px-6">
        {/* Main heading */}
        <div className="animate-fade-in">
          <h1 className="text-6xl md:text-8xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-8 animate-float">
            CognifyAI
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-2xl mx-auto">
            Intelligent data labeling powered by advanced AI embeddings and human-in-the-loop validation
          </p>
        </div>

        {/* Progress bar */}
        <div className="w-full max-w-md mx-auto mb-8 animate-scale-in">
          <div className="bg-muted/30 h-2 rounded-full overflow-hidden backdrop-blur-sm">
            <div 
              className="h-full bg-gradient-primary transition-all duration-300 ease-out rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Initializing AI models... {progress}%
          </p>
        </div>

        {/* Features showcase */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12 animate-fade-in">
          <div className="bg-gradient-card/30 backdrop-blur-sm border border-border rounded-lg p-6">
            <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              🧠
            </div>
            <h3 className="text-lg font-semibold mb-2">CLIP Embeddings</h3>
            <p className="text-sm text-muted-foreground">
              Advanced vision-language understanding
            </p>
          </div>
          
          <div className="bg-gradient-card/30 backdrop-blur-sm border border-border rounded-lg p-6">
            <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              🔍
            </div>
            <h3 className="text-lg font-semibold mb-2">Smart Search</h3>
            <p className="text-sm text-muted-foreground">
              Vector similarity matching in ChromaDB
            </p>
          </div>
          
          <div className="bg-gradient-card/30 backdrop-blur-sm border border-border rounded-lg p-6">
            <div className="w-12 h-12 bg-primary-glow/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              👥
            </div>
            <h3 className="text-lg font-semibold mb-2">Human-in-Loop</h3>
            <p className="text-sm text-muted-foreground">
              Collaborative AI-human labeling workflow
            </p>
          </div>
        </div>

        {/* CTA Button */}
        {isComplete && (
          <div className="animate-scale-in">
            <Button 
              onClick={onComplete}   // ✅ user triggers navigation
              size="lg"
              className="bg-gradient-primary hover:shadow-glow transition-all duration-300 animate-pulse-glow"
            >
              Start Labeling →
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};
