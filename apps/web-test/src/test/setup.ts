import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length(): number { return this.values.size; }
  clear(): void { this.values.clear(); }
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  key(index: number): string | null { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string): void { this.values.delete(key); }
  setItem(key: string, value: string): void { this.values.set(key, String(value)); }
}

// Node 26 exposes an incomplete experimental global localStorage unless a file
// is configured. Tests use the browser contract instead of that process global.
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: new MemoryStorage(),
});
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: window.localStorage,
});
if (typeof window.crypto.randomUUID !== "function") {
  Object.defineProperty(window.crypto, "randomUUID", {
    configurable: true,
    value: () => "00000000-0000-4000-8000-000000000777",
  });
}

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});
