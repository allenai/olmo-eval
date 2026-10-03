# Build context: dashboard/ (so the UI can import ../contract types).
FROM node:24-alpine AS build
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY contract/ /app/contract/
COPY ui/ /app/ui/
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/ui/dist /usr/share/nginx/html
COPY ui/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
