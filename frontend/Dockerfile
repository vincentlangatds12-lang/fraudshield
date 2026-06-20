# Stage 1: Build Angular app
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install --legacy-peer-deps

COPY . .
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Copy built Angular files
COPY --from=builder /app/dist/fraud-detection/browser /usr/share/nginx/html

# nginx config for SPA (Angular routing)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
