import React, { useEffect, useState, useCallback } from 'react';
import { getApiBaseUrl } from '../../services/config';

const Banner = ({ children, tone = 'info' }) => {
  const toneClasses = tone === 'ok'
    ? 'border-[var(--color-border)] text-[var(--color-text-muted)]'
    : tone === 'warn'
      ? 'border-[var(--color-accent)] text-[var(--color-accent)]'
      : 'border-[var(--color-border)] text-[var(--color-text-secondary)]';
  return (
    <div className={`w-full bg-[var(--color-bg-elev-1)] border ${toneClasses} px-3 py-2 text-xs`}>{children}</div>
  );
};

export default function DiagnosticsBanner() {
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState('unknown'); // ok | warn | unknown
  const [message, setMessage] = useState('');
  const [details, setDetails] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const base = await getApiBaseUrl();
      const res = await fetch(`${base}/diagnostics/gpu?test_llm=false`);
      const data = await res.json();
      setDetails(data);
      const cuda = !!data?.cuda_detected;
      const env = data?.ollama || {};
      const numGpu = env?.num_gpu;
      const gpuLayers = env?.gpu_layers;
      const likelyGpuEnabled = cuda && (numGpu === undefined || numGpu === null || String(numGpu) === '1' || String(numGpu) === '-1' || String(numGpu).toLowerCase() === 'true');
      if (likelyGpuEnabled) {
        setStatus('ok');
        setMessage('GPU available. Ollama appears configured for GPU offload.');
      } else {
        setStatus('warn');
        setMessage('GPU not detected or Ollama GPU offload disabled. Running on CPU may cause high usage.');
      }
    } catch (e) {
      setStatus('warn');
      setMessage('Unable to fetch GPU diagnostics. The app may still function, but performance insights are unavailable.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => { if (alive) await load(); })();
    const id = setInterval(() => { load(); }, 60000); // refresh every 60s
    return () => { alive = false; clearInterval(id); };
  }, [load]);

  const tone = status === 'ok' ? 'ok' : status === 'warn' ? 'warn' : 'info';

  return (
    <Banner tone={tone}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm">
            {loading ? 'Checking GPU/CPU status…' : message}
          </span>
          {details?.nvidia_smi_present !== undefined && (
            <span className="hidden sm:inline text-[var(--color-text-muted)]">
              nvidia-smi: {details.nvidia_smi_present ? 'present' : 'absent'}
            </span>
          )}
          {details?.cuda_detected !== undefined && (
            <span className="hidden sm:inline text-[var(--color-text-muted)]">
              CUDA: {details.cuda_detected ? 'detected' : 'not detected'}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2" />
      </div>
    </Banner>
  );
}
