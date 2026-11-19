/* Some embedded previews have an opaque origin and prohibit browser storage.
 * ml5's bundled speech-commands dependency reads localStorage at import time,
 * even though this workshop never uses that dependency or saves there.
 * Only in that restricted environment, provide an in-memory Storage-shaped
 * fallback. It never writes to disk and disappears on reload.
 * Normal Live Server / HTTPS browser storage is left untouched.
 */
(() => {
  // Accessing localStorage itself can throw for opaque-origin embedded previews.
  try { void window.localStorage; }
  catch {
    const memory = new Map();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        // Supply only the Storage methods ml5 touches; values last until reload.
        get length() { return memory.size; },
        key(index) { return [...memory.keys()][index] ?? null; },
        getItem(key) { return memory.get(String(key)) ?? null; },
        setItem(key, value) { memory.set(String(key), String(value)); },
        removeItem(key) { memory.delete(String(key)); },
        clear() { memory.clear(); }
      }
    });
  }
})();
