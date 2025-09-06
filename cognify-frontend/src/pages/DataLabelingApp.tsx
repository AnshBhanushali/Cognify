import { useState } from "react";
import { LoadingScreen } from "@/components/LoadingScreen";
import { UploadInterface } from "@/components/UploadInterface";
import { ResultsPage } from "@/components/ResultsPage";

type AppState = 'loading' | 'upload' | 'results' | 'complete';

export const DataLabelingApp = () => {
  const [currentState, setCurrentState] = useState<AppState>('loading');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const handleLoadingComplete = () => {
    setCurrentState('upload');
  };

  const handleImageUpload = (file: File) => {
    setUploadedImage(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleAudioUpload = (blob: Blob) => {
    setAudioBlob(blob);
  };

  const handleProceedToResults = () => {
    setCurrentState('results');
  };

  const handleBackToUpload = () => {
    setCurrentState('upload');
  };

  const handleSaveLabel = (label: string, metadata: any) => {
    console.log('Saving label:', label, metadata);
    // Here you would typically send to your backend API
    setCurrentState('complete');
    
    // Reset after 2 seconds for demo
    setTimeout(() => {
      setCurrentState('upload');
      setUploadedImage(null);
      setImagePreview(null);
      setAudioBlob(null);
    }, 2000);
  };

  if (currentState === 'loading') {
    return <LoadingScreen onComplete={handleLoadingComplete} />;
  }

  if (currentState === 'upload') {
    return (
      <UploadInterface
        onImageUpload={handleImageUpload}
        onAudioUpload={handleAudioUpload}
        onNext={handleProceedToResults}
      />
    );
  }

  if (currentState === 'results') {
    return (
      <ResultsPage
        imageFile={uploadedImage || undefined}
        imagePreview={imagePreview || undefined}
        audioBlob={audioBlob || undefined}
        onBack={handleBackToUpload}
        onSave={handleSaveLabel}
      />
    );
  }

  if (currentState === 'complete') {
    return (
      <div className="min-h-screen bg-gradient-hero flex items-center justify-center">
        <div className="text-center animate-scale-in">
          <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
              ✓
            </div>
          </div>
          <h1 className="text-3xl font-bold text-foreground mb-4">
            Label Saved Successfully!
          </h1>
          <p className="text-muted-foreground">
            Your data has been added to the dataset with AI embeddings
          </p>
        </div>
      </div>
    );
  }

  return null;
};