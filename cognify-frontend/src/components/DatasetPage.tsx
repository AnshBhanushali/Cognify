"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

export const DatasetPage = () => {
  const [dataset, setDataset] = useState<any[]>([]);

  const fetchDataset = async () => {
    const res = await fetch("http://localhost:8000/dataset");
    const data = await res.json();
    setDataset(data.items);
  };

  useEffect(() => {
    fetchDataset();
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Your Dataset</h1>
      <Button onClick={fetchDataset} className="mb-4">Refresh</Button>

      <table className="border-collapse border border-gray-400 w-full">
        <thead>
          <tr>
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
              <td className="border px-2 py-1">{JSON.stringify(item.metadata)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-6 flex gap-4">
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
    </div>
  );
};
