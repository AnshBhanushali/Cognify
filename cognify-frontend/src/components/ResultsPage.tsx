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
  const [selectedLabel, setSelectedLabel] = useState<string>(suggestions[0]?.label || "");
  const [isEditing, setIsEditing] = useState(false);
  const [customLabel, setCustomLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  // Stats + IDs + embeddings
  type Stats = {
    embedding_time: number;
    chroma_time: number;
    similar_labels: number;
    top_confidence: number;
  } | null;

  const [stats, setStats] = useState<Stats>(null);
  const [imageId, setImageId] = useState<string>("");
  const [embedding, setEmbedding] = useState<number[]>([]);

  // Audio-related state
  const [audioId, setAudioId] = useState<string>("");
  const [audioEmbedding, setAudioEmbedding] = useState<number[]>([]);
  const [transcript, setTranscript] = useState<string>("");
  const [brokenWords, setBrokenWords] = useState<string[]>([]);
  const [diarization, setDiarization] = useState<any[]>([]);
  const [retrievedLabel, setRetrievedLabel] = useState<string | null>(null);

  const { toast } = useToast();

  useEffect(() => {
    if (suggestions.length > 0) {
      setSelectedLabel(suggestions[0].label);
    }
  }, [suggestions]);

  // Local state for preview
const [previewUrl, setPreviewUrl] = useState<string | null>(null);

useEffect(() => {
  if (imageFile) {
    const url = URL.createObjectURL(imageFile);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url); // cleanup
  }
}, [imageFile]);

  
useEffect(() => {
  const fetchSuggestions = async () => {
    if (imageFile && !audioBlob) {
      // IMAGE MODE
      const form = new FormData();
      form.append("file", imageFile);
      const res = await fetch("http://localhost:8000/upload/image", {
        method: "POST",
        body: form,
      });
      const data = await res.json();

      setImageId(data.id);
      setEmbedding(data.embedding || []);
      setStats(data.stats || null);

      const mapped =
        (data.suggestions || []).length > 0
          ? data.suggestions.map((s: any) => ({
              label: s.label,
              confidence: s.score ?? 0,
              source: "clip" as const,
            }))
          : [
              {
                label: data.predicted_label || "unknown",
                confidence: 1.0,
                source: "chromadb" as const,
                embedding: data.embedding,
              },
            ];
      setSuggestions(mapped);

    } else if (audioBlob && !imageFile) {
      // AUDIO MODE
      const form = new FormData();
      form.append("file", new File([audioBlob], "voice.wav", { type: "audio/wav" }));
      const res = await fetch("http://localhost:8000/upload/audio", {
        method: "POST",
        body: form,
      });
      const data = await res.json();

      setAudioId(data.id);
      setAudioEmbedding(data.audio_embedding || []);
      setTranscript(data.transcript || "");
      setBrokenWords(data.broken_words || []);
      setDiarization(data.diarization || []);
      setRetrievedLabel(data.retrieved_label || null);

      setSuggestions([
        {
          label: data.is_broken
            ? `Broken: ${data.broken_words.join(", ")}`
            : data.summary || data.transcript,
          confidence: data.avg_confidence,
          source: "gpt4v",
        },
      ]);
    } else {
      // reset if neither
      setSuggestions([]);
    }
  };

  fetchSuggestions();
}, [imageFile, audioBlob]);


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
  
    if (imageFile) {
      await fetch("http://localhost:8000/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: imageId || crypto.randomUUID(),
          label: selectedLabel,
          embedding,
        }),
      });
    } else if (audioBlob) {
      await fetch("http://localhost:8000/confirm_audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_id: audioId || crypto.randomUUID(),
          label: selectedLabel,
          embedding: audioEmbedding,
        }),
      });
    }
  
    setIsProcessing(false);
  
    toast({
      title: "Label saved successfully",
      description: "Added to your dataset with embedding",
    });
  
    const metadata = {
      timestamp: new Date().toISOString(),
      filename: imageFile?.name || "audio_input",
      confidence:
        suggestions.find((s) => s.label === selectedLabel)?.confidence ??
        (stats?.top_confidence ?? 1.0),
      notes: notes.trim(),
      hasAudio: !!audioBlob,
      hasImage: !!imageFile,
    };
    onSave(selectedLabel, metadata);
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
            {/* Image Preview */}
{previewUrl && (
  <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Target className="w-5 h-5 text-primary" />
        Uploaded Image
      </CardTitle>
    </CardHeader>
    <CardContent>
      <img 
        src={previewUrl} 
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
                <CardContent className="space-y-4">
                  <div className="bg-muted/30 rounded-lg p-4 text-sm space-y-2">
                    <p><strong>Transcript:</strong> {transcript || "—"}</p>
                    <p><strong>Broken Words:</strong> {brokenWords.length > 0 ? brokenWords.join(", ") : "—"}</p>
                    <p><strong>Retrieved Label:</strong> {retrievedLabel || "—"}</p>
                    <p><strong>Speakers:</strong> {diarization.map((d, i) => `S${d.speaker}[${d.start}-${d.end}]`).join(", ") || "—"}</p>
                  </div>
                  <a
                    href={`http://localhost:8000/audio/${audioId}/repair`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-green-600 text-white px-4 py-2 rounded-lg inline-block"
                  >
                    Download Repaired Audio
                  </a>
                </CardContent>
              </Card>
            )}

            {/* Processing Stats */}
            <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-primary" />
                  Processing Stats
                </CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Embedding Time</div>
                  <div className="font-medium">{stats ? `${stats.embedding_time.toFixed(3)}s` : "—"}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">ChromaDB Search</div>
                  <div className="font-medium">{stats ? `${stats.chroma_time.toFixed(3)}s` : "—"}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Similar Labels</div>
                  <div className="font-medium">{stats ? stats.similar_labels : "—"}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Top Confidence</div>
                  <div className="font-medium">{stats ? `${(stats.top_confidence * 100).toFixed(1)}%` : "—"}</div>
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
                    <div className="mt-6 flex gap-4 justify-end">
                      {/* Save Button */}
                      <button
                        onClick={handleSave}
                        className="bg-gradient-to-r from-purple-600 to-blue-500 text-white px-4 py-2 rounded-lg"
                      >
                        Save to Dataset
                      </button>

                      {/* Download JSON */}
                      <a
                        href="http://localhost:8000/dataset/download/json"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="bg-blue-600 text-white px-4 py-2 rounded-lg"
                      >
                        Download JSON
                      </a>

                      {/* Download CSV */}
                      <a
                        href="http://localhost:8000/dataset/download/csv"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="bg-green-600 text-white px-4 py-2 rounded-lg"
                      >
                        Download CSV
                      </a>
                    </div>

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
