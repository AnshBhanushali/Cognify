import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, Mic, MicOff, Image as ImageIcon, FileX } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface UploadInterfaceProps {
  onImageUpload: (file: File) => void;
  onAudioUpload: (blob: Blob) => void;
  onNext: () => void;
}

export const UploadInterface = ({ onImageUpload, onAudioUpload, onNext }: UploadInterfaceProps) => {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  
  const { toast } = useToast();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file",
        variant: "destructive",
      });
      return;
    }
  
    // Preview
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  
    setUploadedImage(file);
  
    // ✅ send to backend
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/upload/image", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
  
    console.log("Image backend response:", data);
    onImageUpload(file);
  };
  

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        setAudioBlob(audioBlob);
      
        // ✅ send to backend
        const formData = new FormData();
        formData.append("file", new File([audioBlob], "recording.wav", { type: "audio/wav" }));
        const res = await fetch("http://localhost:8000/upload/audio", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
      
        console.log("Audio backend response:", data);
        onAudioUpload(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };
      
      
      toast({
        title: "Recording started",
        description: "Speak your labeling instructions",
      });
    } catch (error) {
      toast({
        title: "Recording failed",
        description: "Please check microphone permissions",
        variant: "destructive",
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      toast({
        title: "Recording completed",
        description: "Audio captured successfully",
      });
    }
  };

  const clearImage = () => {
    setUploadedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const clearAudio = () => {
    setAudioBlob(null);
  };

  return (
    <div className="min-h-screen bg-gradient-hero flex items-center justify-center p-6">
      <div className="w-full max-w-4xl mx-auto">
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">
            Upload & Label
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Upload an image or record voice instructions to begin the AI-powered labeling process
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Image Upload Section */}
          <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
            <CardContent className="p-8">
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                <ImageIcon className="w-6 h-6 text-primary" />
                Image Upload
              </h2>
              
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-primary bg-primary/10' 
                    : 'border-border hover:border-primary/50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {imagePreview ? (
                  <div className="space-y-4">
                    <img 
                      src={imagePreview} 
                      alt="Preview" 
                      className="max-w-full h-48 object-contain mx-auto rounded-lg"
                    />
                    <div className="flex gap-2 justify-center">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={clearImage}
                        className="flex items-center gap-2"
                      >
                        <FileX className="w-4 h-4" />
                        Clear
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Upload className="w-12 h-12 text-muted-foreground mx-auto" />
                    <div>
                      <p className="text-lg font-medium mb-2">Drop your image here</p>
                      <p className="text-sm text-muted-foreground mb-4">
                        or click to browse files
                      </p>
                      <Button 
                        onClick={() => fileInputRef.current?.click()}
                        className="bg-gradient-primary hover:shadow-glow transition-all duration-300"
                      >
                        Choose File
                      </Button>
                    </div>
                  </div>
                )}
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                className="hidden"
              />
            </CardContent>
          </Card>

          {/* Voice Recording Section */}
          <Card className="bg-gradient-card border-border/50 backdrop-blur-sm animate-scale-in">
            <CardContent className="p-8">
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                <Mic className="w-6 h-6 text-accent" />
                Voice Instructions
              </h2>
              
              <div className="text-center space-y-6">
                <div className="bg-muted/30 rounded-lg p-6">
                  {isRecording ? (
                    <div className="space-y-4">
                      <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto animate-pulse-glow">
                        <div className="w-8 h-8 bg-red-500 rounded-full animate-pulse" />
                      </div>
                      <p className="text-lg font-medium">Recording in progress...</p>
                      <p className="text-sm text-muted-foreground">
                        Speak your labeling instructions clearly
                      </p>
                    </div>
                  ) : audioBlob ? (
                    <div className="space-y-4">
                      <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto">
                        <div className="w-8 h-8 bg-green-500 rounded-full" />
                      </div>
                      <p className="text-lg font-medium">Recording captured</p>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={clearAudio}
                        className="flex items-center gap-2"
                      >
                        <FileX className="w-4 h-4" />
                        Clear Recording
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Mic className="w-16 h-16 text-muted-foreground mx-auto" />
                      <p className="text-lg font-medium">Ready to record</p>
                      <p className="text-sm text-muted-foreground">
                        Describe what you want to label in the image
                      </p>
                    </div>
                  )}
                </div>
                
                <Button
                  onClick={isRecording ? stopRecording : startRecording}
                  size="lg"
                  className={`transition-all duration-300 ${
                    isRecording 
                      ? 'bg-red-500 hover:bg-red-600' 
                      : 'bg-gradient-primary hover:shadow-glow'
                  }`}
                >
                  {isRecording ? (
                    <>
                      <MicOff className="w-5 h-5 mr-2" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Mic className="w-5 h-5 mr-2" />
                      Start Recording
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Continue Button */}
        {(uploadedImage || audioBlob) && (
          <div className="text-center animate-scale-in">
            <Button 
              onClick={onNext}
              size="lg"
              className="bg-gradient-primary hover:shadow-glow transition-all duration-300 text-lg px-8 py-3"
            >
              Process with AI →
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};