import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const router = inject(Router);
  const withCreds = req.clone({ withCredentials: true });
  return next(withCreds).pipe(
    catchError(err => {
      if (err.status === 401) router.navigate(['/login']);
      return throwError(() => err);
    }),
  );
};
