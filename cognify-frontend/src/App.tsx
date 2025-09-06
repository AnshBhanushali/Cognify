import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useNavigate, Navigate } from "react-router-dom";

import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import { LoadingScreen } from "./components/LoadingScreen";
import { UploadInterface } from "./components/UploadInterface";
import { ResultsPage } from "./components/ResultsPage";
import DatasetPage from "./components/DatasetPage";

import { useState } from "react";

const queryClient = new QueryClient();

const AppRoutes = ({
  imageFile,
  setImageFile,
  audioBlob,
  setAudioBlob,
}: {
  imageFile: File | null;
  setImageFile: React.Dispatch<React.SetStateAction<File | null>>;
  audioBlob: Blob | null;
  setAudioBlob: React.Dispatch<React.SetStateAction<Blob | null>>;
}) => {
  const navigate = useNavigate();

  return (
    <Routes>
      {/* Redirect root → /loading */}
      <Route path="/" element={<Navigate to="/loading" replace />} />

      {/* Loading Screen */}
      <Route
        path="/loading"
        element={<LoadingScreen onComplete={() => navigate("/upload")} />}
      />

      {/* Upload Screen */}
      <Route
  path="/upload"
  element={
    <UploadInterface
      onImageUpload={(file) => {
        setImageFile(file);
        setAudioBlob(null); // reset audio
      }}
      onAudioUpload={(blob) => {
        setAudioBlob(blob);
        setImageFile(null); // reset image
      }}
      onNext={() => navigate("/results")}
      onBack={() => navigate("/loading")}
    />
  }
/>


      {/* Results Screen */}
      <Route
        path="/results"
        element={
          <ResultsPage
            imageFile={imageFile ?? undefined}
            audioBlob={audioBlob ?? undefined}
            onBack={() => navigate("/upload")}    // ✅ explicit back
            onSave={(label, metadata) => {
              console.log("Saved:", label, metadata);
            }}
          />
        }
      />

      {/* Dataset Page */}
      <Route path="/dataset" element={<DatasetPage />} />

      {/* Catch-all */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

const App = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <AppRoutes
            imageFile={imageFile}
            setImageFile={setImageFile}
            audioBlob={audioBlob}
            setAudioBlob={setAudioBlob}
          />
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
