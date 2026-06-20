import { Component, OnInit, signal, HostListener, inject } from '@angular/core';
import { Router, RouterOutlet, RouterLink, RouterLinkActive, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService } from './core/auth.service';
import { ThemeService } from './core/theme.service';
import { filter } from 'rxjs/operators';

export interface NavGroup {
  label: string;
  icon:  string;
  paths: string[];          // all route paths belonging to this group
  primary: string;          // default route when clicking the tab
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements OnInit {

  readonly navGroups: NavGroup[] = [
    {
      label:   'Overview',
      icon:    '📊',
      primary: 'dashboard',
      paths:   ['dashboard', 'transactions'],
    },
    {
      label:   'Analytics',
      icon:    '🌐',
      primary: 'analytics-3d',
      paths:   ['analytics-3d', 'model-comparison', 'model-monitoring'],
    },
    {
      label:   'Explainability',
      icon:    '🧠',
      primary: 'explainability',
      paths:   ['explainability'],
    },
    {
      label:   'ML Pipeline',
      icon:    '⚙️',
      primary: 'training',
      paths:   ['training', 'review-queue'],
    },
  ];

  // Sub-nav for each group (shown below the main tabs when group is active)
  readonly subNav: Record<string, { path: string; label: string }[]> = {
    'dashboard': [
      { path: 'dashboard',    label: 'Dashboard'    },
      { path: 'transactions', label: 'Transactions' },
    ],
    'analytics-3d': [
      { path: 'analytics-3d',     label: 'Advanced Analytics' },
      { path: 'model-comparison', label: 'Model Comparison'   },
      { path: 'model-monitoring', label: 'Model Health'       },
    ],
    'explainability': [
      { path: 'explainability', label: 'SHAP / LIME / Feature Importance' },
    ],
    'training': [
      { path: 'training',     label: 'Pipeline & Training' },
      { path: 'review-queue', label: 'Human-in-the-Loop'   },
    ],
  };

  activeGroup = signal<string>('dashboard');
  userMenuOpen = signal(false);
  currentUrl   = signal('/dashboard');

  constructor(
    public auth:   AuthService,
    public theme:  ThemeService,
    public router: Router,
  ) {}

  ngOnInit(): void {
    // Track active group on route changes
    this.router.events.pipe(
      filter(e => e instanceof NavigationEnd),
    ).subscribe((e: any) => {
      this.currentUrl.set(e.urlAfterRedirects || e.url);
      this.updateActiveGroup(e.urlAfterRedirects || e.url);
    });
    this.updateActiveGroup(this.router.url);
  }

  private updateActiveGroup(url: string): void {
    const segment = url.replace('/', '').split('/')[0];
    const group = this.navGroups.find(g => g.paths.includes(segment));
    if (group) this.activeGroup.set(group.primary);
  }

  isGroupActive(group: NavGroup): boolean {
    const segment = this.currentUrl().replace('/', '').split('/')[0];
    return group.paths.includes(segment);
  }

  getSubNav(): { path: string; label: string }[] {
    return this.subNav[this.activeGroup()] || [];
  }

  get hasSubNav(): boolean {
    return (this.subNav[this.activeGroup()] || []).length > 1;
  }

  @HostListener('document:click', ['$event'])
  onDocClick(e: MouseEvent): void {
    const t = e.target as HTMLElement;
    if (!t.closest('.user-menu') && !t.closest('.user-btn')) {
      this.userMenuOpen.set(false);
    }
  }

  toggleUserMenu(): void { this.userMenuOpen.update(v => !v); }

  logout(): void {
    this.auth.logout().subscribe({ next: () => this.router.navigate(['/login']) });
    this.userMenuOpen.set(false);
  }

  get userName():    string { return this.auth.getUser()?.name || 'User'; }
  get userRole():    string { return this.auth.getUser()?.role || ''; }
  get userInitial(): string { return this.userName[0].toUpperCase(); }
  get showShell():   boolean { return this.auth.isAuthenticated(); }
}
