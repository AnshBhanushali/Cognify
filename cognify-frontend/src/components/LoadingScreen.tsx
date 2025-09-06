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
          setIsComplete(true);
          return 100;
        }
        return prev + 2;
      });
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* HERO SECTION WITH BACKGROUND IMAGE */}
      <section className="relative flex flex-col items-center justify-center text-center px-6 py-24 min-h-screen">
        {/* Background */}
        <div className="absolute inset-0">
          <img
            src={heroImage}
            alt="AI Background"
            className="w-full h-full object-cover"
          />
          {/* Dark overlay */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/60 to-gray-900/90" />
        </div>

        {/* Content */}
        <div className="relative z-10 space-y-8 max-w-4xl mx-auto">
          <h1 className="text-6xl md:text-8xl font-bold bg-gradient-primary bg-clip-text text-transparent animate-float">
            CognifyAI
          </h1>
          <p className="text-xl md:text-2xl text-gray-200 max-w-2xl mx-auto">
            Smarter, faster, and cleaner data labeling — powered by AI
          </p>

          {/* Progress bar / system status */}
          <div className="w-full max-w-md mx-auto mb-6 animate-scale-in">
            <div className="bg-white/20 h-3 rounded-full overflow-hidden backdrop-blur-sm">
              <div
                className="h-full bg-gradient-primary transition-all duration-300 ease-out rounded-full"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-gray-200 mt-2">
              Loading AI models... {progress}%
            </p>
          </div>

          {/* CTA Button */}
          {isComplete && (
            <Button
              onClick={onComplete}
              size="lg"
              className="bg-gradient-primary hover:shadow-glow transition-all duration-300 animate-pulse-glow px-10 py-4 text-lg"
            >
              Start Labeling →
            </Button>
          )}
        </div>
      </section>

      {/* TRANSITION BACKGROUND SECTION */}
      <div className="flex-1 bg-gradient-to-b from-gray-900 via-gray-800 to-gray-100 px-6 py-20 space-y-32">
        
        {/* About Section */}
        <section className="max-w-5xl mx-auto text-center space-y-6">
          <h2 className="text-3xl font-bold text-gray-100">Why CognifyAI?</h2>
          <p className="text-lg text-gray-300 max-w-3xl mx-auto">
            Traditional data labeling is slow, expensive, and inconsistent.
            CognifyAI accelerates your workflow using cutting-edge embeddings and
            intelligent automation — while giving you control over every label.
          </p>
        </section>

        {/* How to Use Section */}
        <section className="max-w-6xl mx-auto space-y-8 text-center">
          <h2 className="text-3xl font-bold text-gray-100">How to Use</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-left">
            <div className="bg-gray-50/80 dark:bg-gray-800/70 rounded-xl p-6 border border-gray-200/30 shadow-md">
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">1. Upload</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Upload your images or record a quick voice instruction.
              </p>
            </div>
            <div className="bg-gray-50/80 dark:bg-gray-800/70 rounded-xl p-6 border border-gray-200/30 shadow-md">
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">2. AI Suggests</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Our models instantly suggest the most relevant labels.
              </p>
            </div>
            <div className="bg-gray-50/80 dark:bg-gray-800/70 rounded-xl p-6 border border-gray-200/30 shadow-md">
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">3. Save</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Confirm, edit, and save labels directly into your dataset.
              </p>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="max-w-6xl mx-auto space-y-8 text-center">
          <h2 className="text-3xl font-bold text-gray-100">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-gray-50/80 dark:bg-gray-800/70 rounded-xl p-6 border border-gray-200/30 shadow-md">
              <span className="text-3xl">🧠</span>
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Embeddings</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                CLIP embeddings map vision and text into the same space.
              </p>
            </div>
            <div className="bg-gray-50/80 dark:bg-gray-800/70 rounded-xl p-6 border border-gray-200/30 shadow-md">
              <span className="text-3xl">🔍</span>
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Vector Search</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                ChromaDB finds similar items instantly with vector similarity.
              </p>
            </div>
            <div className="bg-gray-50/80 dark:bg-gray-800/70 rounded-xl p-6 border border-gray-200/30 shadow-md">
              <span className="text-3xl">⚡</span>
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Pipeline</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Optimized workflows let you process hundreds of labels in seconds.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
