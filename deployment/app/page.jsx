"use client"
import { useState } from "react";

const MODELS = [
  {
    id: "model1",
    name: "Model 1 – ResNet18 (DA only)",
    description: "Fine-tuned ResNet18 trained using data augmentation only."
  },
  {
    id: "model2",
    name: "Model 2 – ResNet18 (DA + class weights)",
    description: "ResNet18 trained with class-weighted loss to address dataset imbalance."
  },
  {
    id: "model3",
    name: "Model 3 – MobileNetV3 (friend’s model)",
    description: "MobileNetV3 model trained by teammate and exported as ONNX."
  },
];


function Greeting() {
  return (
    <h1 className="text-2xl md:text-3xl font-bold text-center mb-4">
      Hello there, welcome to the Animal Checker 🐾
    </h1>
  );
}

export default function LandingPage() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    setResult(null);
    setError("");
  };

  const handleFileChange = (e) => {
    const uploaded = e.target.files?.[0];
    if (!uploaded) return;

    setFile(uploaded);
    setResult(null);
    setError("");

    const url = URL.createObjectURL(uploaded);
    setPreviewUrl(url);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError("Please upload an image of an animal first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("modelId", selectedModel);
      formData.append("image", file);

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });


      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      // Expecting something like:
      // { topClass: "dog", probability: 0.97, allProbs: {...} }
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Something went wrong while checking the animal. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl bg-slate-800 rounded-2xl shadow-xl p-6 md:p-8">
        <Greeting />

        <p className="text-center text-slate-300 mb-6">
          Choose a model, upload an image of an animal, and we’ll tell you what it is.
        </p>

        <div className="mb-6">
          <label className="block mb-2 font-semibold">
            1. Choose a model
          </label>
          <select
            value={selectedModel}
            onChange={handleModelChange}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-400"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>

          <p className="mt-2 text-sm text-slate-400">
            {
              MODELS.find((m) => m.id === selectedModel)?.description
            }
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block mb-2 font-semibold">
              2. Upload an animal image
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-slate-200
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-lg file:border-0
                         file:text-sm file:font-semibold
                         file:bg-emerald-500 file:text-slate-900
                         hover:file:bg-emerald-400
                         cursor-pointer"
            />
            <p className="mt-1 text-xs text-slate-400">
              JPG, PNG, etc.
            </p>
          </div>

          {previewUrl && (
            <div className="flex items-center gap-4">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-32 h-32 object-cover rounded-lg border border-slate-700"
              />
              <p className="text-sm text-slate-300">
                Preview of the uploaded image.
              </p>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 inline-flex items-center justify-center px-4 py-2 rounded-lg
                       bg-emerald-500 text-slate-900 font-semibold
                       hover:bg-emerald-400 disabled:opacity-60 disabled:cursor-not-allowed
                       transition-colors"
          >
            {loading ? "Checking..." : "3. Check Animal"}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-900/60 text-red-100 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-6 p-4 rounded-lg bg-slate-900 border border-slate-700">
            <h2 className="font-semibold text-lg mb-2">Prediction</h2>
            <p className="text-slate-200">
              <span className="font-bold">Model:</span>{" "}
              {MODELS.find((m) => m.id === selectedModel)?.name}
            </p>
            <p className="text-slate-200 mt-1">
              <span className="font-bold">Top class:</span>{" "}
              {result.topClass ?? "Unknown"}
            </p>
            {result.probability != null && (
              <p className="text-slate-200">
                <span className="font-bold">Confidence:</span>{" "}
                {(result.probability * 100).toFixed(1)}%
              </p>
            )}
            {result.allProbs && (
              <div className="mt-3">
                <p className="font-semibold mb-1 text-slate-100 text-sm">
                  All class probabilities:
                </p>
                <ul className="text-xs text-slate-300 space-y-1">
                  {Object.entries(result.allProbs).map(([cls, prob]) => (
                    <li key={cls}>
                      {cls}: {(prob * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
