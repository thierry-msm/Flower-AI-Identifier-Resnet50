import React, { useState } from 'react';
import { Upload, X, Flower, Loader2, AlertCircle } from 'lucide-react';
import './index.css';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictions(null);
      setError(null);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setPredictions(null);
    setError(null);
  };

  const identifyFlower = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      // Nota: Certifique-se de que seu backend Python está rodando na porta 8000
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Falha ao conectar com o servidor API.');
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setPredictions(data.predictions);
    } catch (err) {
      console.error(err);
      setError('Erro ao identificar. O backend Python está rodando?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans">
      {/* Header */}
      <header className="bg-emerald-600 text-white p-6 shadow-lg">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <Flower className="w-8 h-8" />
          <h1 className="text-2xl font-bold">FlowerAI Identifier</h1>
        </div>
      </header>

      <main className="max-w-3xl mx-auto p-6 mt-8">
        
        {/* Intro Card */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8 border border-slate-200">
          <h2 className="text-lg font-semibold mb-2 text-slate-700">Identificação de Espécies</h2>
          <p className="text-slate-500">
            Carregue uma foto de uma flor e nossa Inteligência Artificial (ResNet50) irá identificar a qual das 102 espécies do dataset Oxford ela pertence.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          
          {/* Upload Section */}
          <div className="space-y-4">
            <div 
              className={`border-2 border-dashed rounded-xl h-80 flex flex-col items-center justify-center transition-all relative overflow-hidden bg-slate-100 ${
                previewUrl ? 'border-emerald-500' : 'border-slate-300 hover:border-emerald-400'
              }`}
            >
              {previewUrl ? (
                <>
                  <img 
                    src={previewUrl} 
                    alt="Preview" 
                    className="w-full h-full object-cover"
                  />
                  <button 
                    onClick={clearImage}
                    className="absolute top-2 right-2 bg-white/80 p-1 rounded-full hover:bg-white text-slate-600 transition-colors"
                  >
                    <X size={20} />
                  </button>
                </>
              ) : (
                <label className="cursor-pointer flex flex-col items-center w-full h-full justify-center">
                  <div className="bg-emerald-100 p-4 rounded-full mb-3 text-emerald-600">
                    <Upload size={32} />
                  </div>
                  <span className="font-medium text-slate-600">Clique para enviar foto</span>
                  <span className="text-sm text-slate-400 mt-1">JPG ou PNG</span>
                  <input 
                    type="file" 
                    className="hidden" 
                    accept="image/*"
                    onChange={handleImageChange}
                  />
                </label>
              )}
            </div>

            <button
              onClick={identifyFlower}
              disabled={!selectedImage || loading}
              className={`w-full py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all ${
                !selectedImage || loading
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-md hover:shadow-lg'
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" /> Processando...
                </>
              ) : (
                <>
                  <Flower /> Identificar Espécie
                </>
              )}
            </button>

            {error && (
              <div className="bg-red-50 text-red-600 p-4 rounded-lg flex items-center gap-2 text-sm border border-red-100">
                <AlertCircle size={16} />
                {error}
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="font-bold text-lg mb-4 text-slate-800 border-b pb-2">Resultados da Análise</h3>
            
            {!predictions ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 text-center py-10">
                <div className="bg-slate-50 p-4 rounded-full mb-3">
                  <Flower size={40} className="opacity-20" />
                </div>
                <p>Os resultados aparecerão aqui após a análise.</p>
              </div>
            ) : (
              <div className="space-y-6">
                {predictions.map((pred, index) => (
                  <div key={index} className="space-y-1">
                    <div className="flex justify-between items-end mb-1">
                      <span className={`font-medium ${index === 0 ? 'text-emerald-700 text-lg' : 'text-slate-600'}`}>
                        {pred.species}
                      </span>
                      <span className="font-bold text-slate-700">{pred.confidence}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                      <div 
                        className={`h-2.5 rounded-full transition-all duration-1000 ${
                          index === 0 ? 'bg-emerald-500' : 'bg-slate-400'
                        }`} 
                        style={{ width: `${pred.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
                
                <div className="mt-6 pt-4 border-t text-xs text-slate-400">
                  Modelo: ResNet50 (Transfer Learning)
                  <br/>
                  Dataset: Oxford Flowers 102
                </div>
              </div>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}