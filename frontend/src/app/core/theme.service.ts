import { Injectable, signal, effect } from '@angular/core';

export type Theme = 'light' | 'dark';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly STORAGE_KEY = 'fraudshield-theme';

  // Initialise from localStorage, falling back to system preference
  private _theme = signal<Theme>(this._load());

  readonly theme     = this._theme.asReadonly();
  readonly isDark    = () => this._theme() === 'dark';

  constructor() {
    // Apply immediately and whenever theme changes
    effect(() => this._apply(this._theme()));
  }

  toggle(): void {
    this._theme.update(t => t === 'dark' ? 'light' : 'dark');
    localStorage.setItem(this.STORAGE_KEY, this._theme());
  }

  set(t: Theme): void {
    this._theme.set(t);
    localStorage.setItem(this.STORAGE_KEY, t);
  }

  private _load(): Theme {
    const stored = localStorage.getItem(this.STORAGE_KEY) as Theme | null;
    if (stored === 'dark' || stored === 'light') return stored;
    // Use system preference as default
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  private _apply(theme: Theme): void {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.setAttribute('data-theme', 'dark');
    } else {
      root.removeAttribute('data-theme');
    }
  }
}
