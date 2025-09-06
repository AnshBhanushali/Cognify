"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

export default function DatasetPage() {
  const [dataset, setDataset] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

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

  useEffect(() => {
    fetchDataset();
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">📂 Your Dataset</h1>

      <div className="flex gap-4 mb-6">
        <Button onClick={fetchDataset}>Refresh Dataset</Button>
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
    </div>
  );
}
