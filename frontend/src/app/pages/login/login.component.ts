import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth.service';
import { ThemeService } from '../../core/theme.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="login-page">

      <!-- Theme toggle top-right -->
      <button class="theme-btn" (click)="theme.toggle()" [title]="theme.isDark() ? 'Light mode' : 'Dark mode'">
        {{ theme.isDark() ? '☀️' : '🌙' }}
      </button>

      <div class="login-card">
        <div class="login-card__logo">
          <img src="/umba-logo.png" alt="Umba" class="login-logo" />
        </div>
        <h1>FraudShield</h1>
        <p class="login-card__sub">Umba Fraud Detection Platform</p>

        <form (ngSubmit)="submit()" #f="ngForm" class="login-form">
          <div class="field">
            <label>Email</label>
            <input type="email" [(ngModel)]="email" name="email" placeholder="analyst@umba.com" required />
          </div>
          <div class="field">
            <label>Password</label>
            <input type="password" [(ngModel)]="password" name="password" placeholder="••••••••" required />
          </div>

          <div class="error-msg" *ngIf="error">{{ error }}</div>

          <button type="submit" class="btn-login" [disabled]="loading">
            <span *ngIf="!loading">Sign In →</span>
            <span *ngIf="loading" class="spin">⟳</span>
          </button>
        </form>

        <div class="demo-hints">
          <p>Demo accounts — click to fill:</p>
          <div class="hint-row" *ngFor="let u of demoUsers" (click)="fill(u.email, u.pass)">
            <code>{{ u.email }}</code><span class="pass-hint">{{ u.pass }}</span>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .login-page {
      min-height: 100vh;
      background: var(--bg);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      position: relative;
    }
    .theme-btn {
      position: absolute;
      top: 20px; right: 20px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      width: 38px; height: 38px;
      cursor: pointer;
      font-size: 1.1rem;
      display: flex; align-items: center; justify-content: center;
      transition: background .15s;
      &:hover { background: var(--hover-bg); }
    }
    .login-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 40px;
      width: 100%;
      max-width: 400px;
      text-align: center;
      box-shadow: var(--shadow-lg);
    }
    .login-card__logo { margin-bottom: 12px; }
    .login-logo { width: 56px; height: 56px; object-fit: contain; border-radius: 12px; }
    h1 { margin: 0 0 4px; color: var(--text); font-size: 1.5rem; font-weight: 700; }
    .login-card__sub { color: var(--text-muted); font-size: .85rem; margin: 0 0 28px; }
    .login-form { text-align: left; }
    .field { margin-bottom: 14px; }
    .field label {
      display: block; font-size: .75rem; font-weight: 600;
      color: var(--text-muted); margin-bottom: 5px;
      text-transform: uppercase; letter-spacing: .06em;
    }
    .field input {
      width: 100%; padding: 10px 12px; border: 1px solid var(--border);
      border-radius: 8px; background: var(--input-bg); color: var(--text);
      font-size: .9rem; box-sizing: border-box;
      &:focus { outline: 2px solid var(--accent); border-color: transparent; }
    }
    .error-msg {
      background: #fef2f2; border: 1px solid #fca5a5;
      border-radius: 8px; padding: 8px 12px;
      font-size: .82rem; color: #b91c1c; margin-bottom: 12px;
    }
    .btn-login {
      width: 100%; padding: 11px; background: var(--accent); color: #fff;
      border: none; border-radius: 8px; font-size: .92rem; font-weight: 600;
      cursor: pointer; margin-top: 4px; transition: opacity .15s, transform .1s;
      letter-spacing: .01em;
      &:hover:not(:disabled) { opacity: .9; }
      &:active:not(:disabled) { transform: scale(.99); }
      &:disabled { opacity: .5; cursor: not-allowed; }
    }
    .spin { display: inline-block; animation: spin .8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .demo-hints {
      margin-top: 24px; border-top: 1px solid var(--border);
      padding-top: 14px; text-align: left;
      p { font-size: .73rem; color: var(--text-muted); margin: 0 0 8px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    }
    .hint-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 10px; border-radius: 6px; cursor: pointer; margin-bottom: 4px;
      border: 1px solid transparent; transition: all .12s;
      &:hover { background: var(--hover-bg); border-color: var(--border); }
      code { font-size: .78rem; color: var(--accent); }
    }
    .pass-hint { font-size: .76rem; color: var(--text-muted); font-family: monospace; }
  `],
})
export class LoginComponent {
  email    = '';
  password = '';
  loading  = false;
  error    = '';

  demoUsers = [
    { email: 'analyst@umba.com', pass: 'umba2026' },
    { email: 'admin@umba.com',   pass: 'admin2026' },
    { email: 'demo@umba.com',    pass: 'demo'      },
  ];

  constructor(
    private auth:  AuthService,
    public  theme: ThemeService,
    private router: Router,
  ) {}

  fill(e: string, p: string): void { this.email = e; this.password = p; }

  submit(): void {
    this.loading = true;
    this.error   = '';
    this.auth.login(this.email, this.password).subscribe({
      next: () => this.router.navigate(['/dashboard']),
      error: e  => {
        this.error   = e?.error?.detail || 'Invalid credentials';
        this.loading = false;
      },
    });
  }
}
