import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { 
  Check, 
  X, 
  Edit, 
  Save, 
  Brain, 
  Clock, 
  Target,
  ChevronLeft,
  Download
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface LabelSuggestion {
  label: string;
  confidence: number;
  source: 'chromadb' | 'clip' | 'gpt4v';
  embedding?: number[];
}

interface ResultsPageProps {
  imageFile?: File;
  imagePreview?: string;
  audioBlob?: Blob;
  onBack: () => void;
  onSave: (finalLabel: string, metadata: any) => void;
}

export const ResultsPage = ({ 
  imageFile, 
  imagePreview, 
  audioBlob, 
  onBack, 
  onSave 
}: ResultsPageProps) => {
  const [suggestions, setSuggestions] = useState<LabelSuggestion[]>([]);

  useEffect(() => {
    if (suggestions.length > 0) {
      setSelectedLabel(suggestions[0].label);
    }
  }, [suggestions]);
  
useEffect(() => {
  const fetchSuggestions = async () => {
    if (imageFile) {
      const form = new FormData();
      form.append("file", imageFile);
      const res = await fetch("http://localhost:8000/upload/image", {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      // Adapt backend response into your LabelSuggestion format
      setSuggestions([
        {
          label: data.predicted_label || data.id, // 👈 prefer predicted_label
          confidence: 1.0,
          source: "clip",
          embedding: data.embedding,
        },
      ]);
      
    } else if (audioBlob) {
      const form = new FormData();
      form.append("file", new File([audioBlob], "voice.wav", { type: "audio/wav" }));
      const res = await fetch("http://localhost:8000/upload/audio", {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setSuggestions([
        {
          label: data.is_broken
            ? `Broken: ${data.broken_words.join(", ")}`
            : data.summary || data.transcript,
          confidence: data.avg_confidence,
          source: "gpt4v",
        },
      ]);
    }
  };

  fetchSuggestions();
}, [imageFile, audioBlob]);

  
  const [selectedLabel, setSelectedLabel] = useState<string>(suggestions[0]?.label || "");
  const [isEditing, setIsEditing] = useState(false);
  const [customLabel, setCustomLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  
  const { toast } = useToast();

  const handleAcceptSuggestion = (label: string) => {
    setSelectedLabel(label);
    setIsEditing(false);
    setCustomLabel("");
  };

  const handleCustomLabel = () => {
    if (customLabel.trim()) {
      setSelectedLabel(customLabel.trim());
      setIsEditing(false);
    }
  };

  const handleSave = async () => {
    if (!selectedLabel.trim()) {
      toast({
        title: "Label required",
        description: "Please select or enter a label before saving",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    
    // Simulate processing
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const metadata = {
      timestamp: new Date().toISOString(),
      filename: imageFile?.name || 'audio_input',
      confidence: suggestions.find(s => s.label === selectedLabel)?.confidence || 1.0,
      notes: notes.trim(),
      hasAudio: !!audioBlob,
      hasImage: !!imageFile
    };
    
    onSave(selectedLabel, metadata);
    setIsProcessing(false);
    
    toast({
      title: "Label saved successfully",
      description: "Added to your dataset with embedding",
    });
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'chromadb': return 'bg-primary/20 text-primary';
      case 'clip': return 'bg-accent/20 text-accent';
      case 'gpt4v': return 'bg-purple-500/20 text-purple-400';
      default: return 'bg-muted/20 text-muted-foreground';
    }
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'chromadb': return '🗄️';
      case 'clip': return '🧠';
      case 'gpt4v': return '🤖';
      default: return '📊';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8 animate-fade-in">
          <Button 
            onClick={onBack}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <ChevronLeft className="w-4 h-4" />
            Back
          </Button>
          <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            AI Label Suggestions
          </h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Preview */}
          <div className="space-y-6">
            {/* Image Preview */}
            {imagePreview && (
              <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-primary" />
                    Uploaded Image
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={imagePreview} 
                    alt="Uploaded content" 
                    className="w-full h-64 object-contain rounded-lg bg-muted/20"
                  />
                  <div className="mt-4 text-sm text-muted-foreground">
                    <p>Filename: {imageFile?.name}</p>
                    <p>Size: {((imageFile?.size || 0) / 1024).toFixed(1)} KB</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Audio Input */}
            {audioBlob && (
              <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-accent" />
                    Voice Instructions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-muted/30 rounded-lg p-4 text-center">
                    <div className="w-12 h-12 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-3">
                      🎵
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Audio transcription will be processed by speech-to-text
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Duration: ~{Math.round((audioBlob.size / 8000))}s
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Processing Stats */}
            <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-purple-400" />
                  Processing Stats
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Embedding Time</p>
                    <p className="font-semibold">0.3s</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">ChromaDB Search</p>
                    <p className="font-semibold">0.1s</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Similar Labels</p>
                    <p className="font-semibold">{suggestions.length}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Top Confidence</p>
                    <p className="font-semibold">
  {suggestions.length > 0 ? `${(suggestions[0].confidence * 100).toFixed(0)}%` : "N/A"}
</p>

                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Suggestions and Actions */}
          <div className="space-y-6">
            {/* AI Suggestions */}
            <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary" />
                  AI Suggestions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {suggestions.map((suggestion, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg border transition-all duration-300 cursor-pointer ${
                      selectedLabel === suggestion.label
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/50'
                    }`}
                    onClick={() => handleAcceptSuggestion(suggestion.label)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{suggestion.label}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={getSourceColor(suggestion.source)}>
                          {getSourceIcon(suggestion.source)} {suggestion.source.toUpperCase()}
                        </Badge>
                        {selectedLabel === suggestion.label && (
                          <Check className="w-4 h-4 text-primary" />
                        )}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex-1 bg-muted/30 rounded-full h-2 mr-3">
                        <div 
                          className="bg-gradient-primary h-2 rounded-full transition-all duration-500"
                          style={{ width: `${suggestion.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {(suggestion.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Custom Label */}
            <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Edit className="w-5 h-5 text-accent" />
                  Custom Label
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isEditing ? (
                  <div className="space-y-3">
                    <Input
                      placeholder="Enter custom label..."
                      value={customLabel}
                      onChange={(e) => setCustomLabel(e.target.value)}
                      className="bg-muted/30 border-border"
                    />
                    <div className="flex gap-2">
                      <Button 
                        onClick={handleCustomLabel}
                        size="sm"
                        className="bg-gradient-primary hover:shadow-glow transition-all duration-300"
                      >
                        <Save className="w-4 h-4 mr-1" />
                        Save
                      </Button>
                      <Button 
                        onClick={() => setIsEditing(false)}
                        variant="outline"
                        size="sm"
                      >
                        <X className="w-4 h-4 mr-1" />
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      Create your own label if none of the suggestions fit
                    </p>
                    <Button 
                      onClick={() => setIsEditing(true)}
                      variant="outline"
                      className="w-full"
                    >
                      <Edit className="w-4 h-4 mr-2" />
                      Create Custom Label
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Notes */}
            <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
              <CardHeader>
                <CardTitle>Additional Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <Textarea
                  placeholder="Add any additional context or notes..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="bg-muted/30 border-border min-h-[100px]"
                />
              </CardContent>
            </Card>

            {/* Final Selection */}
            {selectedLabel && (
              <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
                <CardHeader>
                  <CardTitle className="text-primary">Selected Label</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg border border-primary/20">
                    <span className="font-semibold text-lg">{selectedLabel}</span>
                    <Check className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex gap-3 mt-6">
                    <Button 
                      onClick={handleSave}
                      disabled={isProcessing}
                      className="flex-1 bg-gradient-primary hover:shadow-glow transition-all duration-300"
                    >
                      {isProcessing ? (
                        "Processing..."
                      ) : (
                        <>
                          <Download className="w-4 h-4 mr-2" />
                          Save to Dataset
                        </>
                      )}
                    </Button>
                    <Button 
                      variant="outline"
                      onClick={() => setSelectedLabel("")}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};