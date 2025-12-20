// Runtime config loader for frontend
// Priority: env (REACT_APP_API_URL/VITE_API_URL) -> /config.json -> window-derived fallback.

let cachedConfig = null;
let inflightPromise = null;

export async function getConfig() {
  if (cachedConfig) return cachedConfig;
  if (inflightPromise) return inflightPromise;

  // 1) Environment variables (CRA/Vite) at build-time/runtime
  const envApi = process.env.REACT_APP_API_URL || process.env.VITE_API_URL;
  if (envApi) {
    cachedConfig = { apiBaseUrl: envApi };
    return cachedConfig;
  }

  // 2) Runtime config.json from the same origin
  inflightPromise = (async () => {
    try {
      const res = await fetch('/config.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(`Failed to load config.json: ${res.status}`);
      const data = await res.json();
      // Basic shape validation
      if (!data || typeof data.apiBaseUrl !== 'string' || !data.apiBaseUrl) {
        throw new Error('Invalid config.json contents');
      }
      cachedConfig = data;
      return cachedConfig;
    } catch (err) {
      console.warn('Runtime config load failed, falling back:', err);
      // 3) Fallback: infer API URL from window location
      try {
        const { protocol, hostname, port } = window.location;
        // If frontend on 3000, assume backend on 8000; otherwise reuse port if present
        let backendPort = '8000';
        if (port && port !== '3000' && port !== '80' && port !== '443') {
          backendPort = port;
        } else if (!port && (protocol === 'http:' || protocol === 'https:')) {
          // keep default 8000 when standard ports
          backendPort = '8000';
        }
        const fallback = `${protocol}//${hostname}:${backendPort}`;
        cachedConfig = { apiBaseUrl: fallback };
        return cachedConfig;
      } catch (e) {
        // Final hard fallback
        cachedConfig = { apiBaseUrl: 'http://localhost:8000' };
        return cachedConfig;
      }
    } finally {
      inflightPromise = null;
    }
  })();

  return inflightPromise;
}

export async function getApiBaseUrl() {
  const cfg = await getConfig();
  return cfg.apiBaseUrl;
}
