import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-stat-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="stat-card" [class]="'stat-card--' + variant">
      <div class="stat-card__left">
        <div class="stat-card__icon-wrap">
          <span class="stat-card__icon">{{ icon }}</span>
        </div>
      </div>
      <div class="stat-card__body">
        <div class="stat-card__header-row">
          <div class="stat-card__label">{{ label }}</div>
          <div class="stat-card__badge" *ngIf="badge" [class]="'badge--' + badgeVariant">{{ badge }}</div>
        </div>
        <div class="stat-card__value">{{ value }}</div>
        <div class="stat-card__sub" *ngIf="sub">{{ sub }}</div>
      </div>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100%; }

    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
      display: flex;
      align-items: flex-start;
      gap: 12px;
      height: 100%;
      box-sizing: border-box;
      transition: box-shadow .2s, transform .15s;
      min-height: 100px;

      &:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,.08);
        transform: translateY(-1px);
      }
    }

    /* Accent border per variant */
    .stat-card--danger  { border-left: 3px solid #ef4444; }
    .stat-card--warning { border-left: 3px solid #f59e0b; }
    .stat-card--success { border-left: 3px solid #10b981; }
    .stat-card--info    { border-left: 3px solid #3b82f6; }
    .stat-card--default { border-left: 3px solid var(--accent); }

    /* Icon container — fixed size circle */
    .stat-card__left { flex-shrink: 0; }
    .stat-card__icon-wrap {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: var(--hover-bg);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .stat-card__icon { font-size: 1.15rem; line-height: 1; }

    /* Body */
    .stat-card__body { flex: 1; min-width: 0; }

    .stat-card__header-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 4px;
    }

    .stat-card__label {
      font-size: .68rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: .07em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .stat-card__value {
      font-size: 1.55rem;
      font-weight: 800;
      color: var(--text);
      line-height: 1.1;
      letter-spacing: -.03em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .stat-card__sub {
      font-size: .72rem;
      color: var(--text-muted);
      margin-top: 3px;
      line-height: 1.4;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Badge */
    .stat-card__badge {
      padding: 2px 8px;
      border-radius: 999px;
      font-size: .65rem;
      font-weight: 700;
      white-space: nowrap;
      flex-shrink: 0;
      letter-spacing: .03em;
    }
    .badge--danger  { background: rgba(239,68,68,.12);  color: #ef4444; }
    .badge--success { background: rgba(16,185,129,.12); color: #10b981; }
    .badge--warning { background: rgba(245,158,11,.12); color: #f59e0b; }
    .badge--info    { background: rgba(59,130,246,.12); color: #3b82f6; }
  `],
})
export class StatCardComponent {
  @Input() label   = '';
  @Input() value: string | number = '';
  @Input() sub     = '';
  @Input() icon    = '';
  @Input() badge   = '';
  @Input() variant: 'default' | 'danger' | 'warning' | 'success' | 'info' = 'default';
  @Input() badgeVariant: 'danger' | 'success' | 'warning' | 'info' = 'info';
}
