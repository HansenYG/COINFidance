// Centralized API base URL.
// Local dev: defaults to http://127.0.0.1:8000.
// Production: Vercel injects VITE_API_BASE at build time
// (set in Vercel Project Settings → Environment Variables).
export const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";
