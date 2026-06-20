import { Injectable, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface AuthUser {
  email: string;
  name:  string;
  role:  string;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private _user    = signal<AuthUser | null>(null);
  private _checked = signal(false);

  readonly user            = this._user.asReadonly();
  readonly isAuthenticated = computed(() => this._user() !== null);
  readonly isChecked       = computed(() => this._checked());

  constructor(private http: HttpClient) {}

  checkSession(): Observable<boolean> {
    return this.http
      .get<AuthUser>(`${environment.apiUrl}/auth/me`, { withCredentials: true })
      .pipe(
        tap(user => { this._user.set(user); this._checked.set(true); }),
        map(() => true),
        catchError(() => {
          this._user.set(null);
          this._checked.set(true);
          return of(false);
        }),
      );
  }

  login(email: string, password: string): Observable<AuthUser> {
    return this.http
      .post<AuthUser>(
        `${environment.apiUrl}/auth/login`,
        { email, password },
        { withCredentials: true },
      )
      .pipe(tap(user => this._user.set(user)));
  }

  logout(): Observable<any> {
    return this.http
      .post(`${environment.apiUrl}/auth/logout`, {}, { withCredentials: true })
      .pipe(tap(() => this._user.set(null)));
  }

  getUser(): AuthUser | null {
    return this._user();
  }
}
