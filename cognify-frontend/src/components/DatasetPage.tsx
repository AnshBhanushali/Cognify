"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip } from "recharts";

export default function DatasetPage() {
  const [dataset, setDataset] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [embeddings, setEmbeddings] = useState<{ X: number[][]; y: string[] }>({
    X: [],
    y: [],
  });

  // --- fetch dataset ---
  const fetchDataset = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/dataset");
      const data = await res.json();
      setDataset(data.items || []);
    } catch (err) {
      console.error("Failed to fetch dataset", err);
    }
    setLoading(false);
  };

  // --- retrain endpoint ---
  const handleRetrain = async () => {
    try {
      const res = await fetch("http://localhost:8000/retrain", { method: "POST" });
      const data = await res.json();
      alert(`Retraining complete: ${data.status}, samples: ${data.samples}`);
    } catch (err) {
      console.error("Retrain failed", err);
      alert("Failed to retrain model");
    }
  };

  // --- fetch embeddings ---
  const fetchEmbeddings = async () => {
    try {
      const res = await fetch("http://localhost:8000/embeddings/all");
      const data = await res.json();
      setEmbeddings(data);
    } catch (err) {
      console.error("Failed to fetch embeddings", err);
    }
  };

  useEffect(() => {
    fetchDataset();
    fetchEmbeddings();
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">📂 Your Dataset</h1>

      {/* Actions */}
      <div className="flex gap-4 mb-6">
        <Button onClick={fetchDataset}>Refresh Dataset</Button>
        <Button onClick={handleRetrain} className="bg-purple-500 text-white">
          Retrain Model
        </Button>
        <a
          href="http://localhost:8000/dataset/download/json"
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Download JSON
        </a>
        <a
          href="http://localhost:8000/dataset/download/csv"
          className="bg-green-500 text-white px-4 py-2 rounded"
        >
          Download CSV
        </a>
      </div>

      {/* Dataset table */}
      {loading ? (
        <p>Loading...</p>
      ) : dataset.length === 0 ? (
        <p className="text-gray-500">No dataset entries yet. Save some labels first!</p>
      ) : (
        <table className="border-collapse border border-gray-400 w-full">
          <thead>
            <tr className="bg-gray-100">
              <th className="border px-2 py-1">ID</th>
              <th className="border px-2 py-1">Label</th>
              <th className="border px-2 py-1">Metadata</th>
            </tr>
          </thead>
          <tbody>
            {dataset.map((item, idx) => (
              <tr key={idx}>
                <td className="border px-2 py-1">{item.id}</td>
                <td className="border px-2 py-1">{item.label}</td>
                <td className="border px-2 py-1">
                  {JSON.stringify(item.metadata)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Embedding visualization */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-4">📊 Embedding Visualization</h2>
        {embeddings.X.length > 0 ? (
          <ScatterChart width={600} height={400}>
            <XAxis dataKey="x" type="number" />
            <YAxis dataKey="y" type="number" />
            <Tooltip />
            <Scatter
              data={embeddings.X.map((vec, i) => ({
                x: vec[0],
                y: vec[1],
                label: embeddings.y[i],
              }))}
              fill="#8884d8"
            />
          </ScatterChart>
        ) : (
          <p className="text-gray-500">No embeddings available yet.</p>
        )}
      </div>
    </div>
  );
}
