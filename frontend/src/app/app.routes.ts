import { Routes } from '@angular/router';
import { authGuard } from './core/auth.guard';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./pages/login/login.component').then(m => m.LoginComponent) },

  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },

  { path: 'dashboard',        canActivate: [authGuard], loadComponent: () => import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent) },
  { path: 'transactions',     canActivate: [authGuard], loadComponent: () => import('./pages/transactions/transactions.component').then(m => m.TransactionsComponent) },
  { path: 'review-queue',     canActivate: [authGuard], loadComponent: () => import('./pages/review-queue/review-queue.component').then(m => m.ReviewQueueComponent) },
  { path: 'model-comparison', canActivate: [authGuard], loadComponent: () => import('./pages/model-comparison/model-comparison.component').then(m => m.ModelComparisonComponent) },
  { path: 'model-monitoring', canActivate: [authGuard], loadComponent: () => import('./pages/model-monitoring/model-monitoring.component').then(m => m.ModelMonitoringComponent) },
  { path: 'explainability',   canActivate: [authGuard], loadComponent: () => import('./pages/explainability/explainability.component').then(m => m.ExplainabilityComponent) },
  { path: 'training',         canActivate: [authGuard], loadComponent: () => import('./pages/training/training.component').then(m => m.TrainingComponent) },
  { path: 'analytics-3d',     canActivate: [authGuard], loadComponent: () => import('./pages/analytics-3d/analytics-3d.component').then(m => m.Analytics3dComponent) },

  { path: '**', redirectTo: 'dashboard' },
];
